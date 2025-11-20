## 🚀 Chatbot (LangGraph + RAG/Stock Tool + Google Gemini + Streamlit)

**Chatbot** built with **LangGraph**, **Google Gemini**, **ChromaDB**, and **Streamlit**, offering intelligent document-based Q&A, Indian stock market analysis, and multi-threaded chat management — all with real-time streaming and persistent state.

### 📄 PDF Document Analysis
- Ask questions directly from the uploaded PDF 
- Text-based PDF ingestion (no OCR)

### 📊 Indian Stock Market Analysis
- Fetches NSE/BSE companies latest financial data: (quarterly results, P&L, balance sheet, ratios, cash flow, shareholding) using Screener.in API  
- LLM-powered stock symbol extraction and matching  
- Automated insights and investment-style analysis  

### 🖥️ Modern Streamlit UI
- Clean, minimal, dark-theme-friendly design  
- Drag-and-drop PDF upload with processing feedback  
- Thread management controls (Create, rename, switch, and delete conversation threads with confirmation)  
- SQLite-based persistent conversation history  
- Auto-generated conversation titles  
- One PDF per thread with isolated vectorstore 
- Document status and feature availability indicators  
- True token-by-token streaming responses  
- Live progress indicators during processing 

## 📐 High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                      │
│                   (Streamlit Application – front.py)        │
└───────────────────────────────────────────┬─────────────────┘
                                            │
                                            │  User Query
                                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       LangGraph Backend                     │
│                             (back.py)                       │
│                                                             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │                   Classifier Node                    │  │
│   │  Determines which workflow to use:                   │  │
│   │    • Document QA (RAG)                               │  │
│   │    • Stock Analysis                                  │  │
│   │    • General Chat                                    │  │
│   └──────────────┬───────────────┬───────────────────────┘  │
│                  │               │                          │
│                  ▼               ▼                          ▼
│        ┌────────────────┐ ┌────────────────┐       ┌────────────────┐
│        │   RAG Workflow │ │  Stock Workflow│       │   Chat Workflow│
│        └────────────────┘ └────────────────┘       └────────────────┘
└─────────────────────────────────────────────────────────────┘
                                            │
                                            │  Executes Tool Calls
                                            ▼
┌─────────────────────────────────────────────────────────────┐
│                          Tools Module                       │
│                           (tools.py)                        │
│                                                             │
│   • RAG utilities (document search & retrieval)             │
│   • Stock tools (symbol lookup, formatting, screening)      │
└─────────────────────────────────────────────────────────────┘
                                            │
                                            │  Accesses External Services
                                            ▼
┌─────────────────────────────────────────────────────────────┐
│                        External Services                    │
│                                                             │
│   ┌────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│   │ Google Gemini  │   │   Screener.in    │   │  Chroma Vector DB  │
│   │     (LLM)      │   │ (Stock Data API) │   │ (Embeddings Store) │
│   └────────────────┘   └──────────────────┘   └────────────────────┘
└─────────────────────────────────────────────────────────────┘


```
## 🔄 Data Flow Diagrams

### Document Q&A Flow
```
User Question
    ↓
Classifier (detects document query)
    ↓
RAG Retrieval Node
    ↓
Chroma Vector DB (semantic search)
    ↓
Retrieve Top 5 Chunks
    ↓
RAG Response Node
    ↓
LLM (with context)
    ↓
Formatted Answer → User

```
### Stock Analysis Flow
```
User Question ("Analyze Titan")
    ↓
Classifier (detects stock query)
    ↓
Symbol Extractor Node
    ↓
LLM (extract company name: "Titan")
    ↓
Screener.in Search API
    ↓
Multiple Results → LLM picks best match
    ↓
Stock Scraper Node (scrap.py)
    ↓
Scrape: Quarterly, P&L, Growth, Balance, Cash, Ratios, Shareholding
    ↓
Stock Analysis Response Node
    ↓
Display Raw Tables + Generate AI Analysis
    ↓
Formatted Report → User

```
### General Chat Flow
```
User Question
    ↓
Classifier (detects general query)
    ↓
Chat Node
    ↓
LLM (direct response)
    ↓
Streaming Answer → User

```
## 🗄️ Storage Architecture

### SQLite Database (`chatbot.db`)
```sql
thread_titles
├── thread_id (TEXT PRIMARY KEY)
└── title (TEXT)

thread_files
├── thread_id (TEXT PRIMARY KEY)
└── filename (TEXT)
```

### Chroma Vector Store (`./chroma_db/{thread_id}/`)
```
Each conversation thread has its own directory:
./chroma_db/
├── thread-uuid-1/
│   └── [embeddings for document 1]
├── thread-uuid-2/
│   └── [embeddings for document 2]
└── ...
```

### LangGraph Checkpointer
```
Stores conversation state in SQLite:
- Full message history
- Graph state (needs_rag, stock_symbol, etc.)
- Enables conversation persistence
```

## 🎯 Decision Logic

### Classifier Logic (3-Way)
```python
IF document exists:
    IF query about stocks (explicit) → STOCK_ANALYSIS
    ELSE IF query likely in document → DOCUMENT_QA
    ELSE → GENERAL_CHAT
ELSE:
    IF query about stocks → STOCK_ANALYSIS
    ELSE → GENERAL_CHAT
```

### Stock Symbol Matching
```python
1. Extract company name via LLM
2. Search Screener.in API
3. IF single result → Use it
4. IF multiple results → LLM picks best match
5. Extract symbol from URL
6. Return symbol or "UNKNOWN"
```

### RAG Retrieval Strategy
```python
1. User query → Embedding (Azure OpenAI)
2. Semantic search in Chroma (k=5)
3. Retrieve top 5 most similar chunks
4. Combine chunks with metadata
5. Pass to LLM as context
6. Generate answer grounded in context
```

<img width="1873" height="761" alt="image" src="https://github.com/user-attachments/assets/ad0a5911-a7d6-4fa4-8966-04e9a0530ec3" />
