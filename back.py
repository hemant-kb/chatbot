# back.py
"""Backend for PDF-only LangGraph chatbot: extraction, chunking, vectorstore, graph state, streaming."""

from typing import Annotated, Iterator, List, Optional, TypedDict, Literal
import os
import shutil
import sqlite3
import gc
import time
import stat
from dotenv import load_dotenv

# LangChain / Azure wrappers
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# Vector DB / Splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Persistence / graph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages

# ----------------------------------------------------------------------------- #
# ENV
# ----------------------------------------------------------------------------- #
load_dotenv()
# DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")
# AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
# EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

# ----------------------------------------------------------------------------- #
# LLM & EMBEDDINGS
# ----------------------------------------------------------------------------- #
# llm = AzureChatOpenAI(
#     deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     temperature=0.6,
#     streaming=True,
# )

# embeddings = AzureOpenAIEmbeddings(
#     deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
# )


# LLM Configuration
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
    max_output_tokens=100
)

# llm = ChatOpenAI(
#     model="sonar",
#     base_url="https://api.perplexity.ai",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0.7,
#     max_tokens=100,
# )


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    task_type="retrieval_document"
)


# ----------------------------------------------------------------------------- #
# STORAGE & SQLITE
# ----------------------------------------------------------------------------- #
vector_stores: dict[str, Chroma] = {}
thread_filenames: dict[str, str] = {}

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Create metadata tables if missing
conn.execute(
    """CREATE TABLE IF NOT EXISTS thread_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT
    )"""
)
conn.execute(
    """CREATE TABLE IF NOT EXISTS thread_files (
        thread_id TEXT PRIMARY KEY,
        filename TEXT
    )"""
)
conn.commit()

# ----------------------------------------------------------------------------- #
# PDF EXTRACTORS (text-only, no OCR)
# ----------------------------------------------------------------------------- #
def _extract_pdf_pymupdf(path: str) -> str:
    try:
        import fitz
    except Exception:
        return ""
    try:
        parts: List[str] = []
        with fitz.open(path) as doc:
            if doc.is_encrypted:
                try:
                    doc.authenticate("")  # try empty password
                except Exception:
                    return ""
            for p in doc:
                txt = p.get_text("text") or ""
                if txt.strip():
                    parts.append(txt)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def _extract_pdf_pypdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return ""
        parts: List[str] = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
                if txt.strip():
                    parts.append(txt)
            except Exception:
                continue
        return "\n".join(parts).strip()
    except Exception:
        return ""


def _extract_text_pdf(path: str) -> str:
    """Try a sequence of extractors; return first successful non-empty string."""
    for f in (_extract_pdf_pymupdf, _extract_pdf_pypdf):
        txt = f(path)
        if txt:
            return txt
    return ""

# ----------------------------------------------------------------------------- #
# Utilities: normalize, chunking, chroma dir handling
# ----------------------------------------------------------------------------- #
def _normalize_space(s: str) -> str:
    if not s:
        return ""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


def _chunk_text_to_documents(text: str, thread_id: str, source: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = [c.strip() for c in splitter.split_text(text or "") if c and c.strip()]
    return [Document(page_content=c, metadata={"thread_id": thread_id, "source": source}) for c in chunks]


def _ensure_thread_chroma_dir(thread_id: str) -> str:
    base = "./chroma_db"
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, thread_id)
    if os.path.exists(path):
        shutil.rmtree(path, onerror=_force_remove_readonly)
    os.makedirs(path, exist_ok=True)
    return path


def _force_remove_readonly(func, path, exc_info):
    """Handler to remove read-only files on Windows during rmtree."""
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

# ----------------------------------------------------------------------------- #
# File processing (PDF-only)
# ----------------------------------------------------------------------------- #
def process_uploaded_file(file_path: str, file_type: str, thread_id: str, filename: str) -> int:
    """
    Process an uploaded file. Only PDF supported (text-based PDFs).
    Returns number of chunks created.
    """
    if file_type.lower() != "pdf":
        raise ValueError("Only PDF files are supported. Do not upload text files.")

    raw_text = _extract_text_pdf(file_path)
    if not raw_text:
        raise ValueError("Could not extract text from PDF. The PDF may be scanned or empty.")

    normalized_text = _normalize_space(raw_text)
    if not normalized_text:
        raise ValueError("File processed but contains no readable text.")

    # Chunk into documents
    chunks = _chunk_text_to_documents(normalized_text, thread_id, filename)
    if not chunks:
        raise ValueError("No chunks created. File may be too small or empty.")

    # Write to Chroma directory and create vectorstore
    persist_dir = _ensure_thread_chroma_dir(thread_id)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"thread_{thread_id}",
        persist_directory=persist_dir,
    )

    vector_stores[thread_id] = vectorstore
    thread_filenames[thread_id] = filename

    # Save filename mapping to sqlite
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO thread_files (thread_id, filename) VALUES (?, ?)",
        (thread_id, filename),
    )
    conn.commit()

    return len(chunks)


def get_thread_filename(thread_id: str) -> Optional[str]:
    cur = conn.cursor()
    row = cur.execute("SELECT filename FROM thread_files WHERE thread_id = ?", (thread_id,)).fetchone()
    return row[0] if row else None


def delete_thread_files(thread_id: str):
    """Delete vector store and files for a thread. Attempts to close client before removal."""
    # Close & cleanup memory references
    if thread_id in vector_stores:
        try:
            vs = vector_stores[thread_id]
            # Try to clear client caches and delete collection (best-effort)
            if hasattr(vs, "_client"):
                try:
                    vs._client.clear_system_cache()
                except Exception:
                    pass
            if hasattr(vs, "_collection"):
                try:
                    vs._client.delete_collection(vs._collection.name)
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning: Error closing vectorstore: {e}")
        finally:
            vector_stores.pop(thread_id, None)

    thread_filenames.pop(thread_id, None)

    # Ensure OS releases file handles (Windows)
    gc.collect()
    time.sleep(0.5)

    chroma_path = f"./chroma_db/{thread_id}"
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path, onerror=_force_remove_readonly)
        except Exception as e:
            print(f"Warning: Could not delete {chroma_path}: {e}")
            time.sleep(1)
            gc.collect()
            try:
                shutil.rmtree(chroma_path, onerror=_force_remove_readonly)
            except Exception as e2:
                print(f"Error: Failed to delete {chroma_path} after retry: {e2}")

# ----------------------------------------------------------------------------- #
# Graph state, classifier, retrieval, and streaming
# ----------------------------------------------------------------------------- #
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    retrieved_context: str
    needs_rag: bool


def stream_llm_response(messages: List[BaseMessage], context: Optional[str] = None) -> Iterator[str]:
    """Yield LLM response token-by-token (streaming)."""
    if context:
        system_msg = SystemMessage(
            content=(
                "You are a helpful assistant. Answer the user's question based on this context from their uploaded document.\n\n"
                f"{context}\n\n"
                "If the context doesn't contain relevant information, politely say so."
            )
        )
        messages_to_send = [system_msg] + messages
    else:
        messages_to_send = messages

    for chunk in llm.stream(messages_to_send):
        if chunk.content:
            yield chunk.content


def classifier_node(state: ChatState) -> dict:
    thread_id = state.get("thread_id", "default")
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    has_document = thread_id in vector_stores
    if not has_document:
        return {"needs_rag": False}
    question_lower = last_message.lower()
    greeting_words = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
    if any(word in question_lower for word in greeting_words) and len(question_lower.split()) < 5:
        return {"needs_rag": False}
    return {"needs_rag": True}


def rag_retrieval_node(state: ChatState) -> dict:
    thread_id = state.get("thread_id", "default")
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    if thread_id not in vector_stores:
        return {"retrieved_context": ""}
    retriever = vector_stores[thread_id].as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(last_message)
    if not docs:
        return {"retrieved_context": "No relevant information found."}
    context = "\n\n".join([d.page_content for d in docs if d.page_content.strip()])
    filename = thread_filenames.get(thread_id, "the document")
    return {"retrieved_context": f"Context from '{filename}':\n\n{context}"}


def rag_response_node(state: ChatState) -> dict:
    messages = state["messages"]
    context = state.get("retrieved_context", "")
    system_msg = SystemMessage(
        content=(
            "You are a helpful assistant. Answer the user's question based on this "
            "context from their uploaded document.\n\n"
            f"{context}\n\n"
            "If the context doesn't contain relevant information, politely say so."
        )
    )
    response = llm.invoke([system_msg] + messages)
    return {"messages": [response]}


def chat_node(state: ChatState) -> dict:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def route_after_classifier(state: ChatState) -> Literal["rag_retrieval", "chat"]:
    if state.get("needs_rag", False):
        return "rag_retrieval"
    return "chat"


# Build and compile the state graph
graph = StateGraph(ChatState)
graph.add_node("classifier", classifier_node)
graph.add_node("rag_retrieval", rag_retrieval_node)
graph.add_node("rag_response", rag_response_node)
graph.add_node("chat", chat_node)
graph.add_edge(START, "classifier")
graph.add_conditional_edges(
    "classifier",
    route_after_classifier,
    {"rag_retrieval": "rag_retrieval", "chat": "chat"},
)
graph.add_edge("rag_retrieval", "rag_response")
graph.add_edge("rag_response", END)
graph.add_edge("chat", END)
chatbot = graph.compile(checkpointer=checkpointer)

# Streaming wrapper used by frontend
def get_streaming_response(user_message: str, thread_id: str) -> tuple[Iterator[str], bool]:
    """
    Returns an iterator yielding tokens for the assistant response and a flag
    indicating whether RAG was applied (True if retrieval was used).
    """
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "thread_id": thread_id,
        "retrieved_context": "",
        "needs_rag": False,
    }

    classifier_result = classifier_node(initial_state)
    needs_rag = classifier_result.get("needs_rag", False)
    context = ""
    if needs_rag:
        initial_state["needs_rag"] = True
        retrieval_result = rag_retrieval_node(initial_state)
        context = retrieval_result.get("retrieved_context", "")

    # Load previous messages from the compiled chatbot state (if any)
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        previous_messages = state.values.get("messages", []) or []
    except Exception:
        previous_messages = []

    all_messages = previous_messages + [HumanMessage(content=user_message)]
    return stream_llm_response(all_messages, context), needs_rag

# ----------------------------------------------------------------------------- #
# UI utilities for threads and titles
# ----------------------------------------------------------------------------- #
def retrieve_all_threads() -> list[str]:
    cur = conn.cursor()
    rows = cur.execute("SELECT thread_id FROM thread_titles ORDER BY rowid").fetchall()
    if rows:
        return [r[0] for r in rows]
    # Fallback to thread_files table if no titles
    rows = cur.execute("SELECT thread_id FROM thread_files ORDER BY rowid").fetchall()
    seen = {}
    for r in rows:
        seen.setdefault(r[0], True)
    return list(seen.keys())


def delete_thread(thread_id: str):
    thread_id = str(thread_id)
    cur = conn.cursor()
    cur.execute("DELETE FROM thread_titles WHERE thread_id = ?", (thread_id,))
    cur.execute("DELETE FROM thread_files WHERE thread_id = ?", (thread_id,))
    conn.commit()
    delete_thread_files(thread_id)


def save_thread_title(thread_id: str, title: str):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO thread_titles (thread_id, title) VALUES (?, ?)",
        (thread_id, title),
    )
    conn.commit()


def get_thread_title(thread_id: str) -> str:
    cur = conn.cursor()
    row = cur.execute("SELECT title FROM thread_titles WHERE thread_id = ?", (thread_id,)).fetchone()
    return row[0] if row else thread_id


def generate_title_from_conversation(messages: list) -> str:
    system_prompt = (
        "You are a title generator. "
        "Given the first user and assistant messages of a chat, "
        "generate a short, clear title (max 6 words). "
        "Avoid punctuation like quotes or periods."
    )
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join([f"{m.type}: {m.content}" for m in messages])},
    ]
    resp = llm.invoke(prompt_messages)
    return resp.content.strip()


def save_message_to_graph(thread_id: str, user_message: str, ai_message: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        current_state = chatbot.get_state(config=config)
        existing_messages = current_state.values.get("messages", []) if current_state.values else []
    except Exception:
        existing_messages = []
    new_messages = list(existing_messages) + [
        HumanMessage(content=user_message),
        AIMessage(content=ai_message),
    ]
    chatbot.update_state(config=config, values={"messages": new_messages}, as_node="chat")
