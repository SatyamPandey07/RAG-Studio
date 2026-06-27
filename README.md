# RAG Studio

> **Design, Build, Evaluate, and Deploy Production-Grade RAG Systems**

RAG Studio is an interactive developer environment and no-code/low-code orchestrator designed to help you build, test, compare, and monitor **Retrieval-Augmented Generation (RAG)** systems. It provides full transparency and visual explainability into how AI retrieves documents and answers questions.

---

## Dashboard Mockup

![RAG Studio Dashboard](./dashboard_mockup.png)

---

## What is RAG? (For Layman Users)

Imagine you ask a standard AI model (like ChatGPT) a question about a private company document, a recent news article, or your own notes. The AI won't know the answer because it was not in its training data, so it might make up a fake answer (called **hallucination**).

**RAG (Retrieval-Augmented Generation)** solves this problem in three steps:

```
[ Your Question ]
       │
       ▼
1. RETRIEVE  ───► Search your custom files (PDFs, text) for relevant passages
       │
       ▼
2. AUGMENT   ───► Combine your question with the retrieved passages as context
       │
       ▼
3. GENERATE  ───► Send the combination to the AI (LLM) to write a grounded, fact-checked answer
```

This way, the AI operates like an open-book exam, answering questions based *only* on the pages it looked up.

---

## Key Modules you can Experiment With

1. **Knowledge Sources**: Upload your own files (`.pdf`, `.txt`, `.md`, `.docx`, `.csv`, `.json`, `.html`). The pipeline parses, cleans, and indexes them into the vector database.
2. **Chunking Studio**: Test how the app breaks long files into smaller paragraphs. You can slide controls to change chunk sizes and compare strategies (*Recursive, Fixed, Semantic, Token, Markdown*) side-by-side.
3. **Embedding Studio**: Generate embedding vectors (mathematical representations of meaning) and preview the raw coordinates.
4. **Hybrid Search & Fusion**: Drag sliders to balance **Dense Search** (semantic meaning/concept matches) and **Sparse Search** (exact keyword matching/BM25). The app uses Reciprocal Rank Fusion (RRF) to merge the results.
5. **Re-ranking**: See how secondary cross-encoders re-sort retrieved items to put the absolute best content at the very top.
6. **Chat Playground**: Talk to your documents. Toggle the **Explain Retrieval Steps** accordion to see exactly which document chunks were found, what their similarity scores were, and the total latency/costs.
7. **Evaluation Studio**: Run tests on your pipeline to calculate quantitative scores like **Recall**, **Faithfulness** (no hallucinations), and **Groundedness**.
8. **Monitoring**: Track query volumes, cost breakdowns (LLM vs. Embedding), and user feedback (thumbs up/down) in real-time.

---

## Quick Start

RAG Studio runs in **Demo Mode** by default, meaning **no API keys are required** to start experimenting. It uses SQLite and an in-memory Qdrant client to run instantly with zero database setup.

### 1. Requirements
Ensure you have the following installed:
* [Node.js](https://nodejs.org/) (v18+)
* [Python](https://www.python.org/) (v3.10+)

---

### 2. Startup Guide

#### Terminal 1 — Backend API Server
```bash
cd backend

# 1. Create a local virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install backend dependencies
pip install -r requirements.txt

# 3. Start the FastAPI uvicorn server
uvicorn app.main:app --reload --port 8082
```
* **API Documentation**: [http://localhost:8082/docs](http://localhost:8082/docs)
* **Verify Health Check**: [http://localhost:8082/](http://localhost:8082/)

#### Terminal 2 — Frontend User Interface
```bash
cd frontend

# 1. Install frontend packages
npm install

# 2. Start the Vite React app
npm run dev
```
* **Interactive UI**: Open [http://localhost:3002](http://localhost:3002) in your browser.

---

### 3. Configuring Live Mode (Optional)

To connect RAG Studio to real AI APIs (Google Gemini, OpenAI, Cohere):

1. Inside the `backend/` directory, copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` in a text editor and configure:
   * Change `DEMO_MODE=true` to `DEMO_MODE=false`.
   * Add your `GOOGLE_API_KEY` (available for free at [Google AI Studio](https://aistudio.google.com)). This is used for Gemini 2.0 responses and text embeddings.
   * (Optional) Add `OPENAI_API_KEY` or `COHERE_API_KEY` for other embedders/rerankers.
3. Restart Terminal 1. The application will now generate real vector embeddings and LLM responses!
