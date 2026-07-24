# Flagship Portfolio Project — AI / Generative AI Engineer

**Goal:** Build ONE ambitious, end-to-end project that proves you can engineer a real generative-AI system — not just call an API. This document is your full build plan: the idea, architecture, tech stack, a phased roadmap, what to say in the interview, and how to demo it.

---

## 1. The Project: "Atlas" — An Agentic Research & Answering Assistant

A production-style AI assistant that ingests a body of documents (PDFs, web pages, notes), answers questions about them with citations, and can take actions through tools — all wrapped in a proper API, UI, evaluation suite, and observability.

Think of it as "ChatGPT for a specific knowledge base, but built like an engineer would build it."

**Pick a concrete domain** so it feels real and memorable. Good options:
- Financial filings assistant (answer questions over 10-K / earnings reports)
- Legal/policy assistant (search contracts or regulations)
- Personal research assistant (ingest papers + your notes)
- Developer docs assistant (answer questions over a library's docs)

Recommendation: choose a domain related to the company you're interviewing with. It signals genuine interest.

### Why this project wins interviews
It touches every skill a GenAI employer screens for: RAG, agents/tool-use, MCP, evaluations, guardrails, prompt engineering, vector search, an API, and deployment. Most candidates show a toy chatbot. You'll show a *system*.

---

## 2. Capabilities (what it actually does)

1. **RAG (Retrieval-Augmented Generation)** — ingest documents, chunk + embed them, store in a vector DB, retrieve relevant context, answer with inline citations.
2. **Agentic tool use** — the model decides when to call tools (search the web, run a calculation, query a database) instead of only answering from text.
3. **Custom MCP server** — expose your tools through the Model Context Protocol so they also work inside Claude Desktop / any MCP client. This is your differentiator.
4. **Evaluation harness** — a test set of question/answer pairs; automatically measure answer accuracy, citation correctness, hallucination rate, latency, and cost.
5. **Guardrails** — input validation, PII detection/redaction, and output checks (refuse when context is missing rather than hallucinate).
6. **Observability** — trace every request: which chunks were retrieved, which tools fired, tokens used, cost, latency. Log it.
7. **API + UI** — a FastAPI backend and a lightweight chat UI (Streamlit or Next.js).
8. **Deployment** — Dockerized, deployed to a free/cheap cloud host with a public link.

---

## 3. Architecture

```
                ┌──────────────┐
   User ──────► │   Chat UI    │  (Streamlit / Next.js)
                └──────┬───────┘
                       │ HTTP
                ┌──────▼───────┐
                │  FastAPI API │  ── auth, rate limit, request tracing
                └──────┬───────┘
                       │
             ┌─────────▼──────────┐
             │   Agent Orchestrator│  ── plans, decides tool calls
             └───┬────────┬───────┘
                 │        │
     ┌───────────▼─┐   ┌──▼─────────────┐
     │  RAG Engine │   │  Tool Layer     │
     │  retrieve + │   │  (also exposed  │
     │  rerank     │   │  via MCP server)│
     └──────┬──────┘   └──┬──────────────┘
            │             │
     ┌──────▼──────┐   ┌──▼───────┐
     │  Vector DB  │   │ Web / DB │
     │ (embeddings)│   │  tools   │
     └─────────────┘   └──────────┘

  Cross-cutting: Guardrails • Evaluation harness • Observability/logging
```

**Ingestion pipeline (offline):** load docs → clean → chunk → embed → upsert into vector DB, with metadata (source, page, title) for citations.

**Query pipeline (online):** user question → guardrail check → embed query → retrieve top-k chunks → optional rerank → build prompt with context → agent may call tools → LLM generates answer with citations → output guardrail check → return + log trace.

---

## 4. Tech Stack

Keep it standard so interviewers recognize it.

| Layer | Choice | Notes |
|---|---|---|
| Language | Python | Default for AI roles |
| LLM | Anthropic Claude or OpenAI API | Pick one; abstract behind an interface so you can swap |
| Embeddings | OpenAI `text-embedding-3-small` or open-source (BGE) | |
| Vector DB | Chroma (local) or Qdrant / pgvector | Chroma is easiest to start |
| Framework | Plain Python + FastMCP for the MCP server | Avoid over-relying on LangChain — showing you can wire RAG yourself impresses more |
| Reranker | Cohere rerank or a cross-encoder | Optional, boosts quality |
| API | FastAPI | |
| UI | Streamlit (fast) or Next.js (polished) | |
| Evals | Custom harness + optional Ragas | |
| Observability | Structured logging + LangSmith / Phoenix (optional) | |
| Deploy | Docker + Render / Railway / Fly.io | Free tiers exist |
| Version control | GitHub, clean commits, good README | This IS part of the deliverable |

---

## 5. Phased Roadmap

Scale the pace to your available time. Each phase produces something demoable.

### Phase 0 — Setup (Day 1)
- Repo, virtual env, `.env` for keys, project structure, README skeleton.
- "Hello LLM" script that calls the model and prints a response.

### Phase 1 — Basic RAG (Days 2–4)
- Ingestion script: load PDFs, chunk (~500 tokens, overlap), embed, store in Chroma.
- Retrieval + answer function with citations.
- CLI you can ask questions in. **Milestone: it answers questions over your docs with sources.**

### Phase 2 — API + UI (Days 5–6)
- Wrap the pipeline in FastAPI (`/chat` endpoint).
- Streamlit chat UI hitting the API. **Milestone: a usable web app.**

### Phase 3 — Agent + Tools + MCP (Days 7–9)
- Add tool-calling: web search tool, calculator, a "query the vector store" tool.
- Build an MCP server (FastMCP) exposing the same tools. Connect it to Claude Desktop and screenshot it working. **Milestone: the agent decides when to use tools; tools also run in an MCP client.**

### Phase 4 — Evaluation (Days 10–11)
- Write 20–40 gold Q/A pairs for your corpus.
- Eval script scoring accuracy, citation match, hallucination (answers with no support), latency, cost. Output a report/table. **Milestone: you can quantify quality — huge signal to interviewers.**

### Phase 5 — Guardrails + Observability (Days 12–13)
- Input guard (block empty/malicious prompts), PII redaction, "no-context → refuse" behavior.
- Trace logging: retrieved chunks, tool calls, tokens, cost, latency per request. **Milestone: you can debug and explain any answer.**

### Phase 6 — Polish + Deploy (Day 14)
- Dockerize, deploy, public URL.
- README with architecture diagram, screenshots, eval results, and a 60-sec demo GIF/video. **Milestone: shareable link + repo.**

### Stretch goals (if time / to go further)
- Streaming responses (SSE).
- Conversation memory across turns.
- Hybrid search (keyword + vector).
- A fine-tuned or prompt-optimized component with before/after eval numbers.
- Multi-agent setup (planner + researcher + writer).
- Caching layer to cut cost/latency.

---

## 6. What to Say in the Interview

Have crisp answers ready — the project is a springboard for these:

- **"Walk me through the architecture."** Use the diagram. Explain the ingestion vs. query paths.
- **"How do you know it's good?"** Point to your eval harness and numbers. This separates you from 90% of candidates.
- **"How do you prevent hallucinations?"** Retrieval grounding + "refuse when no context" guardrail + citation checking in evals.
- **"How would you scale it?"** Managed vector DB, async workers for ingestion, caching, batching embeddings, a bigger reranker, rate limiting.
- **"What's chunking and why does it matter?"** Explain size/overlap trade-offs and how they affect retrieval.
- **"Why MCP?"** Standard protocol so your tools work across clients; decouples tool logic from the model.
- **"What did you learn / what's hard?"** Have 2–3 real war stories (e.g. retrieval returned irrelevant chunks until you added a reranker; cost blew up until you added caching).

Prepare metrics you can quote: corpus size, accuracy %, avg latency, cost per query.

---

## 7. How to Present It

- **GitHub README** is the front door: one-paragraph pitch, architecture diagram, demo GIF, eval results table, "run it locally" steps, tech stack. Recruiters skim this in 30 seconds.
- **Live demo link** so they can try it.
- **A short Loom/video (60–90s)** walking through a real question end to end.
- **A one-page write-up** of a design decision and a tradeoff you made — engineers love this.

---

## 8. Common Mistakes to Avoid

- Don't just glue together a LangChain quickstart — build the core yourself so you can explain every line.
- Don't skip evals; "it seems to work" is not an engineering answer.
- Don't hardcode one LLM provider — abstract it.
- Don't leave API keys in the repo (use `.env`, add to `.gitignore`).
- Don't over-scope the UI; a clean Streamlit app beats a half-finished Next.js one.
- Keep commits clean and frequent — your git history is part of the story.

---

## 9. Immediate Next Steps

1. Pick your domain + gather ~20–50 documents.
2. Confirm language (Python) and LLM provider.
3. Do Phase 0 + Phase 1 to get a working RAG loop fast — momentum matters.

Tell me your domain and provider and I'll scaffold the repo structure and write the Phase 1 code with you.
