# 📰 The Berliner Search

**The Berliner Search** is an AI-assisted research tool that transforms _The Berliner_ magazine’s PDF archive into structured, searchable data — enabling journalists, editors, and researchers to explore the publication’s history through intelligent semantic search.

---

## 🎯 MVP Goal

Convert static PDFs into a searchable dataset that combines summarization and semantic retrieval.

### Current Features

- ✅ PDF extraction and parsing (text + metadata)
- ✅ NLP preprocessing: cleaning, chunking, and normalization
- ✅ Summarization using **T5-small**
- ✅ Metadata tagging (issue, page, section)
- ✅ Embedding generation with **MiniLM-multilingual**
- ✅ FAISS-based semantic search
- ✅ Evaluation scripts for quick precision testing

### Project Status (October 2025)

🚀 **Stage 6 — Baseline model integrated**

The current production model is  
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,  
supporting both English and German queries.

---

## 🧩 System Architecture – The Berliner Search MVP

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "lineColor": "#7C3AED",
    "primaryBorderColor": "#7C3AED",
    "primaryTextColor": "#111827",
    "tertiaryColor": "#F5F3FF",
    "fontSize": "14px"
  }
}}%%
graph LR;

  %% === External entities ===
  U["👤 Editors (Browser)"];
  A[(📂 PDF Archive / File Server)];

  %% === Application boundary ===
  subgraph APP["🧩 Application"]
    direction LR
    UI["🖥️ Web UI / Dashboard"];
    API["⚙️ Backend API (FastAPI)"];
    W["🔁 Worker (Extractor / Chunker / Summarizer)"];
    P[(🗄️ Processed Store: JSON / JSONL)];
    V[(🧮 Vector Store: FAISS)];
    L["📈 Logs / Metrics"];
  end

  %% === Data / process flow ===
  U --> UI;
  UI --> API;
  API --> P;
  API --> V;
  W --> A;
  W --> P;
  W --> V;
  W --> L;
  API --> UI;
  UI --> U;

  %% === Styles ===
  classDef user fill:#FDF2F8,stroke:#7C3AED,stroke-width:2px,color:#111827;
  classDef service fill:#F5F3FF,stroke:#7C3AED,stroke-width:2px,rx:12,ry:12,color:#111827;
  classDef store fill:#EEF2FF,stroke:#7C3AED,stroke-width:2px,color:#111827;
  classDef external fill:#FFFFFF,stroke:#7C3AED,stroke-width:2px,stroke-dasharray:5 3,color:#111827;
  classDef log fill:#FFF7ED,stroke:#FB923C,stroke-width:2px,color:#111827;
  classDef boundary stroke:#7C3AED,stroke-width:2px,stroke-dasharray:2 2,fill:#FAF5FF;

  class U user;
  class UI,API,W service;
  class P,V store;
  class A external;
  class L log;
  class APP boundary;

  linkStyle default stroke:#7C3AED,stroke-width:1.5px;

```

- Editors interact with a Web Dashboard, which talks to a FastAPI backend.
- The backend serves content from two stores:

1. A Processed JSON store (for summaries)

2. A Vector store (for semantic search).

- Meanwhile, a background worker reads new PDFs from the archive, processes them (extract, clean, chunk, summarize), and writes the results back to those stores — logging everything along the way.

- The user → API → data → UI loop makes the system interactive;
- the worker → stores pipeline keeps data updated.

# 👩‍💻 Author

Developed by **Vanesa Yepes**
AI Product Management Capstone Project (2025)
