<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>PDF-Only RAG Chatbot — README</title>
  <style>
    :root{--bg:#0f1724;--card:#0b1220;--muted:#9aa4b2;--accent:#60a5fa;--glass: rgba(255,255,255,0.04)}
    html,body{height:100%;margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial; background:linear-gradient(180deg,#071027 0%, #071a2b 100%); color:#e6eef6}
    .container{max-width:1000px;margin:36px auto;padding:28px;background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));border-radius:12px;box-shadow:0 6px 30px rgba(2,6,23,0.6);}
    header{display:flex;gap:16px;align-items:center}
    .logo{width:56px;height:56px;border-radius:8px;background:linear-gradient(135deg,#60a5fa,#7c3aed);display:flex;align-items:center;justify-content:center;font-weight:700;color:white}
    h1{margin:0;font-size:20px}
    p.lead{color:var(--muted);margin-top:6px}
    .meta{display:flex;gap:8px;margin-top:12px}
    .badge{background:var(--glass);padding:6px 10px;border-radius:8px;color:var(--muted);font-size:13px}
    section{margin-top:22px}
    h2{font-size:16px;margin-bottom:6px}
    .grid{display:grid;grid-template-columns:1fr 320px;gap:18px}
    .card{background:linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.02));padding:14px;border-radius:10px;border:1px solid rgba(255,255,255,0.03)}
    pre{white-space:pre-wrap;word-break:break-word;background:rgba(0,0,0,0.18);padding:12px;border-radius:8px;color:#dff1ff;overflow:auto}
    code{background:rgba(255,255,255,0.02);padding:2px 6px;border-radius:6px}
    .actions{display:flex;gap:8px;align-items:center}
    button{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.04);color:var(--accent);padding:8px 10px;border-radius:8px;cursor:pointer}
    button.ghost{background:transparent;color:var(--muted);border:1px dashed rgba(255,255,255,0.03)}
    a.repo{color:var(--accent);text-decoration:none}
    .editable{outline:none}
    footer{margin-top:20px;color:var(--muted);font-size:13px}
    .diagram{font-family:monospace;white-space:pre;overflow:auto;padding:8px;background:linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.04));border-radius:8px}
    .copy{font-size:12px;color:var(--muted)}
    .kbd{background:rgba(255,255,255,0.03);padding:2px 6px;border-radius:6px;border:1px solid rgba(255,255,255,0.03)}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(255,255,255,0.03);color:var(--muted);font-size:13px}
    @media (max-width:880px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="container" role="main">
    <header>
      <div class="logo">🏛️</div>
      <div>
        <h1 contenteditable="true" class="editable">PDF-Only RAG Chatbot</h1>
        <p class="lead editable" contenteditable="true">LangGraph + Google Gemini + Chroma + Streamlit — a PDF-grounded RAG assistant with streaming UI.</p>
        <div class="meta">
          <div class="badge">Streams tokens</div>
          <div class="badge">Per-thread Chroma DB</div>
          <div class="badge">SQLite persistence</div>
        </div>
      </div>
    </header>

    <section class="grid">
      <div>
        <div class="card">
          <h2>✨ Features</h2>
          <ul>
            <li contenteditable="true" class="editable">PDF-only ingestion (text-based; scanned PDFs not supported)</li>
            <li contenteditable="true" class="editable">Automatic extraction via PyMuPDF or PyPDF</li>
            <li contenteditable="true" class="editable">Chunking with RecursiveCharacterTextSplitter</li>
            <li contenteditable="true" class="editable">RAG with top-k retrieval (Chroma)</li>
            <li contenteditable="true" class="editable">LangGraph state machine for intelligent routing</li>
            <li contenteditable="true" class="editable">True token streaming in Streamlit UI</li>
            <li contenteditable="true" class="editable">Per-thread vectorstores at <code>./chroma_db/&lt;thread_id&gt;</code></li>
          </ul>

          <h2>Quick Start</h2>
          <pre contenteditable="true" class="editable"># 1. Create .env with keys
GEMINI_API_KEY=your_google_gemini_key
# 2. Install
pip install -r requirements.txt
# 3. Run
streamlit run front.py</pre>

          <h2>Want badges or screenshots?</h2>
          <p class="copy">Tip: edit any text above — this HTML is fully editable in-browser. To persist changes to GitHub, edit the file directly in your repo or copy the content back to README.md.</p>
        </div>

        <div class="card" style="margin-top:14px;">
          <h2>Architecture</h2>
          <div class="diagram" contenteditable="true" spellcheck="false">
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                      │
│                   (Streamlit Application – front.py)        │
└───────────────────────────────────────────┬─────────────────┘
                                            │  User Query
                                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       LangGraph Backend                     │
│                             (back.py)                       │
│   ┌──────────────────────────────────────────────────────┐  │
│   │                   Classifier Node                    │  │
│   │  Selects workflow:                                   │  │
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
          </div>
        </div>

      </div>

      <aside>
        <div class="card">
          <h2>Repository</h2>
          <p class="copy">Edit this file directly on GitHub:</p>
          <p><a class="repo" href="#" id="githubLink">https://github.com/&lt;user&gt;/&lt;repo&gt;/edit/main/README.html</a></p>

          <h2>Files</h2>
          <p class="copy">Main files to look for:</p>
          <ul>
            <li><code>front.py</code> — Streamlit UI</li>
            <li><code>back.py</code> — LangGraph backend</li>
            <li><code>tools.py</code> — helpers (RAG, stock tools)</li>
            <li><code>scrap.py</code> — stock scraping logic</li>
            <li><code>requirements.txt</code> &amp; <code>.env.example</code></li>
          </ul>

          <h2>Storage</h2>
          <p class="pill">SQLite — <code>chatbot.db</code></p>
          <p class="pill">Chroma — <code>./chroma_db/&lt;thread_id&gt;</code></p>
        </div>

        <div class="card" style="margin-top:12px;">
          <h2>Diagrams (Quick)</h2>
          <pre class="copy">Document Q&A → Classifier → RAG Retrieval → Chroma → Gemini → UI</pre>
          <pre class="copy">Stock Flow → Classifier → Screener.in → Scraper → Analysis → UI</pre>
        </div>

        <div class="card" style="margin-top:12px;">
          <h2>Actions</h2>
          <div class="actions">
            <button id="copyMd">Copy README.md</button>
            <button id="download" class="ghost">Download HTML</button>
            <button id="reset" class="ghost">Reset Edits</button>
          </div>
          <p class="copy" style="margin-top:8px">Use <span class="kbd">Ctrl+S</span> to save locally (browser dialog) or edit on GitHub.</p>
        </div>
      </aside>
    </section>

    <footer>
      <div>Made with ❤️ — editable HTML for quick GitHub edits. Replace GitHub link at the top with your repo URL.</div>
    </footer>
  </div>

  <script>
    // Small client-side helpers
    const original = document.documentElement.innerHTML;
    document.getElementById('copyMd').addEventListener('click', async ()=>{
      // Convert the visible content to a simple markdown-ish text
      const title = document.querySelector('h1').innerText.trim();
      const lead = document.querySelector('.lead').innerText.trim();
      const features = Array.from(document.querySelectorAll('.card ul li')).map(li=>'- '+li.innerText.trim()).join('\n');
      const md = `# ${title}\n\n${lead}\n\n## Features\n${features}`;
      try{ await navigator.clipboard.writeText(md); alert('README.md content copied to clipboard') }catch(e){ prompt('Copy the content below', md) }
    });

    document.getElementById('download').addEventListener('click', ()=>{
      const blob = new Blob([document.documentElement.outerHTML], {type:'text/html'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = 'README.html'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    });

    document.getElementById('reset').addEventListener('click', ()=>{
      if(confirm('Reset all edits to the original packaged version?')){
        document.open(); document.write(original); document.close();
      }
    });

    // Helpful UX: keep editables outlined on focus
    document.addEventListener('focusin', e=>{
      if(e.target.classList && e.target.classList.contains('editable')){
        e.target.style.boxShadow='0 0 0 3px rgba(96,165,250,0.12)';
      }
    });
    document.addEventListener('focusout', e=>{
      if(e.target.classList && e.target.classList.contains('editable')){
        e.target.style.boxShadow='none';
      }
    });
  </script>
</body>
</html>
