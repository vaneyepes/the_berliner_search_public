# The Berliner Search

Short MVP: parse PDFs → JSON.

## 🧩 System Architecture – The Berliner Search MVP

```mermaidMarkdown Preview Enhanced
flowchart TD

%% --- Editors ---
subgraph Editors
    U[Editors (Browser)]
end

%% --- Archive ---
subgraph Archive
    A[(PDF Archive / File Server)]
end

%% --- Application ---
subgraph Application
    UI[Web UI / Dashboard]
    API[Backend API (FastAPI)]
    W[Worker (Extractor / Chunker / Summarizer)]
    P[(Processed Store: JSON / JSONL)]
    V[(Vector Store: FAISS)]
    L[Logs / Metrics]
end

%% --- Connections ---
U --> UI
UI --> API
API --> P
API --> V
W --> A
W --> P
W --> V
W --> L
API --> UI
UI --> U
```
