# RAG Studio

> **Design, Build, Evaluate and Deploy Production-Grade RAG Systems**

RAG Studio is an advanced developer environment and low-code orchestrator designed to take Retrieval-Augmented Generation (RAG) applications from concept, through side-by-side experimentation, all the way into production with monitoring and explainability built in.

---

## 🎨 Core Architecture

```
                       ┌────────────────────────┐
                       │   Frontend (Next.js)   │
                       └───────────┬────────────┘
                                   │ (REST / Streaming)
                                   ▼
                       ┌────────────────────────┐
                       │   Backend (FastAPI)    │
                       └───────────┬────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Data & RAG     │       │     Queues      │       │     Vectors     │
│  (PostgreSQL)   │       │ (Redis/Celery)  │       │ (Qdrant/Faiss)  │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

---

## 🚀 Key Modules & Features

1. **Authentication & RBAC**: Secure multi-tenant workspaces, organizations, API key lifecycle management, and audit logs.
2. **Workspace & Projects**: Segment environments, indexes, collections, and pipeline versions per business domain.
3. **Ingestion & Data Sources**: Native support for PDF, Docx, CSV, Notion, Confluence, Websites, Google Drive, S3, SQL DBs, and OCR.
4. **Document Processing Pipeline**: Visualizes OCR, cleanups, metadata extraction, deduplication, chunking, and embedding.
5. **Chunking Studio**: Dynamic preview pane comparing Fixed, Recursive, Token, HTML, and Semantic/Agentic chunking side by side.
6. **Embedding Studio**: Provider benchmarks (Google, OpenAI, Voyage, Cohere, HuggingFace) detailing latencies, costs, and dimensions.
7. **Vector Databases**: Connect and inspect namespaces, index types, metadata coverage, memory usage, and statistics.
8. **Hybrid Retrieval Tuning**: Adjust sliders for Dense Search, Sparse (SPLADE/BM25) Search, and Reciprocal Rank Fusion (RRF) dynamically.
9. **Explainable Retrieval**: Step-by-step visualization showing the exact query rewrite, retrieved chunks, reranking scores, and final prompt context.
10. **Evaluation & Playground**: Side-by-side metric comparison (Faithfulness, Recall@K, Latency, Cost) with live chat testing.

---

## 📂 Project Structure

```
RAG-Studio/
├── docker-compose.yml       # Dev stack (FastAPI, Next.js, Postgres, Qdrant, Redis)
├── README.md                # System documentation
├── backend/
│   ├── Dockerfile           # Backend builder
│   ├── requirements.txt     # Python libraries
│   └── app/
│       ├── main.py          # FastAPI application entry
│       ├── core/            # Database configs, environment variables
│       ├── models/          # SQLModel schemas
│       └── services/        # Chunkers, Embedders, RRF, Evaluation
└── frontend/
    ├── package.json         # React UI configuration
    ├── tailwind.config.js   # Tailwinds tokens
    └── src/
        ├── DashboardApp.tsx # Responsive RAG Studio interactive dashboard
        ├── main.tsx         # App mounting
        └── index.css        # Stylesheet (glassmorphism, animations)
```

---

## ⚡ Quick Start

### 1. Requirements
Ensure you have the following installed:
* Docker & Docker Compose
* Python 3.10+ (for local backend development)
* Node.js 18+ (for local frontend development)

### 2. Spinning up the Docker Environment
From the root folder, run:
```bash
docker-compose up --build
```
This spins up:
* **Frontend**: `http://localhost:3000` (Next.js dashboard)
* **Backend**: `http://localhost:8000/docs` (FastAPI Swagger APIs)
* **Vector Store**: `http://localhost:6333` (Qdrant UI dashboard)
* **Database**: `localhost:5432` (PostgreSQL with pgvector)

### 3. Local Development Setup

#### Backend:
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

#### Frontend:
```bash
cd frontend
npm install
npm run dev -- --port 3000
```
