🚀 PDF-Only RAG Chatbot (LangGraph + Azure OpenAI + Streamlit)

This project is a Retrieval-Augmented Chatbot that lets users upload PDF documents (text-based only, no OCR) and ask questions about their contents.
It uses LangGraph to manage conversation state, Chroma as a vector database, and Azure OpenAI for both embeddings and chat completions.
The UI is built with Streamlit and supports true streaming responses.

Features

📄 PDF-only ingestion (text-based; scanned PDFs not supported).
🔍 Automatic text extraction using PyMuPDF or PyPDF.
✂️ Chunking using RecursiveCharacterTextSplitter.
🧠 RAG (Retrieval-Augmented Generation) with top-k document chunk retrieval.
🧩 LangGraph state machine for intelligent routing:
Classifier → RAG Retrieval → RAG Response Or fallback to Regular Chat (no RAG)

💬 Multi-threaded chats with titles and persistent history (SQLite).
🔁 True streaming responses in the UI.
🗂️ Per-thread vectorstores stored in ./chroma_db/<thread_id>.
🧹 Safe delete with Windows-friendly cleanup for Chroma files.
🖥️ Modern Streamlit UI with rename, delete, thread switching, and PDF indicator.


<img width="1843" height="754" alt="image" src="https://github.com/user-attachments/assets/6c7723dd-708c-4f13-8f68-5fd2cc631ee7" />
