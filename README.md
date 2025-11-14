🚀 PDF-Only RAG Chatbot (LangGraph + Google Gemini + Streamlit)

This project is a Retrieval-Augmented Chatbot that lets users upload PDF documents (text-based only, no OCR) and ask questions about their contents.
It uses LangGraph to manage conversation state, Chroma as a vector database, and Google Gemini for both embeddings and chat completions.
The UI is built with Streamlit and supports true streaming responses.

<h2>✨ Features</h2>

<ul>
  <li>📄 <strong>PDF-only ingestion</strong> (text-based; scanned PDFs not supported)</li>

  <li>🔍 <strong>Automatic text extraction</strong> using <em>PyMuPDF</em> or <em>PyPDF</em></li>

  <li>✂️ <strong>Chunking</strong> powered by <code>RecursiveCharacterTextSplitter</code></li>

  <li>🧠 <strong>RAG (Retrieval-Augmented Generation)</strong> with top-k document chunk retrieval</li>

  <li>🧩 <strong>LangGraph state machine</strong> for intelligent routing:
    <ul>
      <li>Classifier → RAG Retrieval → RAG Response</li>
      <li>Or fallback to <strong>Regular Chat</strong> (no RAG)</li>
    </ul>
  </li>

  <li>💬 <strong>Multi-threaded chats</strong> with titles &amp; persistent history (SQLite)</li>

  <li>🔁 <strong>True streaming responses</strong> in the UI (token-by-token)</li>

  <li>🗂️ <strong>Per-thread vectorstores</strong> stored locally in <code>./chroma_db/&lt;thread_id&gt;</code></li>

  <li>🧹 <strong>Safe delete</strong> with Windows-friendly cleanup for Chroma DB files</li>

  <li>🖥️ <strong>Modern Streamlit UI</strong> with:
    <ul>
      <li>Rename conversation</li>
      <li>Delete conversation</li>
      <li>Switch threads</li>
      <li>PDF upload indicator</li>
    </ul>
  </li>
</ul>



<img width="1843" height="754" alt="image" src="https://github.com/user-attachments/assets/6c7723dd-708c-4f13-8f68-5fd2cc631ee7" />
