# PRD — EquityCrew: Multi-Agent Equity Research Analyst

| | |
|---|---|
| **Author** | Krishna |
| **Date** | July 2026 |
| **Status** | Approved — v1 in development |
| **Version** | 1.0 |

---

## 1. Overview

EquityCrew is a multi-agent AI system that produces a grounded, citation-backed investment research memo for any public company. A crew of specialized agents (financials, news/sentiment, risk, writer, critic) collaborates via a LangGraph state graph, pulling **real-time market data** through tools and validating its own output through a critic-revision loop. Tools are also exposed over **MCP** so any MCP client (e.g. Claude Desktop) can use them.

**Primary purpose:** portfolio-grade demonstration of production AI engineering — agent orchestration, tool use, RAG, evaluation, observability, and deployment.

## 2. Problem statement

Manual equity research is slow: an analyst reads filings, gathers prices and news, and drafts a memo over hours. LLMs alone can't do this reliably because they hallucinate numbers and have no access to live data. The solution must combine live tools (facts), retrieval (grounding in filings), and self-review (quality control) — which is exactly a multi-agent architecture.

## 3. Goals & non-goals

**Goals**

1. Given a ticker, produce a structured research memo in under 2 minutes.
2. Zero invented numbers — every figure traceable to a tool call or retrieved document.
3. Self-review loop: a critic agent approves or forces one revision.
4. Measurable quality: eval harness reporting numeric accuracy and critic catch-rate.
5. Usable three ways: CLI, REST API (FastAPI), and MCP tools.
6. Deployable: Dockerized, cloud-hosted, public demo URL.

**Non-goals**

- Not a trading system; no buy/sell execution. Output is educational, explicitly "not investment advice."
- No user accounts, billing, or multi-tenancy (v1).
- No coverage of private companies or non-equity assets (v1).
- Voice interface (Vapi) is a later, optional layer — not in v1 scope.

## 4. Users

| User | Need |
|---|---|
| Interviewer / hiring manager | See a working agentic system and probe design decisions |
| Krishna (builder) | Learn and demonstrate LangGraph, RAG, MCP, evals, deployment |
| Retail investor (illustrative persona) | Fast, grounded first-pass research on a stock |

## 5. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| F1 | Accept a stock ticker via CLI, REST API, and MCP tool | P0 |
| F2 | Financials agent fetches live price + fundamentals via tools | P0 |
| F3 | News agent fetches recent headlines and summarizes sentiment | P0 |
| F4 | Risk agent identifies key risks; v1.1 grounds this in the 10-K via RAG with citations | P0 (RAG P1) |
| F5 | Writer agent produces memo: Thesis, Fundamentals, News & Sentiment, Key Risks, Bull vs Bear, Rating | P0 |
| F6 | Critic agent reviews draft (structured output), approves or returns feedback; max 1 revision loop | P0 |
| F7 | All numeric claims sourced from tool outputs; agent prompts forbid invented figures | P0 |
| F8 | MCP server exposes price, fundamentals, news, full_analysis | P0 |
| F9 | Eval harness: numeric accuracy vs. ground truth; critic catch-rate on injected errors; latency & cost per memo | P1 |
| F10 | FastAPI endpoint (`POST /research {ticker}`) with streaming progress | P1 |
| F11 | Tracing/observability of each agent step (LangSmith) | P1 |
| F12 | Graceful degradation: if a data source fails, memo states data unavailable rather than guessing | P0 |
| F13 | Invalid ticker → clear error, no memo | P0 |

## 6. Non-functional requirements

- **Latency:** full memo < 120 s (target < 60 s); single tool call < 3 s.
- **Cost:** < $0.15 per memo at default model settings; tracked per run.
- **Reliability:** critic loop hard-capped (no infinite loops); every external call wrapped with error handling.
- **Security:** all keys in `.env` (never committed); no PII processed.
- **Portability:** runs locally with only an Anthropic key; Docker image for deploy.
- **Compliance/safety:** every memo ends with "This is not investment advice."

## 7. Architecture (v1)

```
CLI / FastAPI / MCP client
        │
        ▼
  LangGraph state graph
  START → financials → news → risk → writer → critic ─(approve/cap)→ finalize → END
                                          └──(revise ×1)──→ writer
        │
        ▼ tools (only source of numbers)
  get_price · get_fundamentals · get_news   (yfinance / Finnhub)
  [v1.1] vector store (Chroma/FAISS) ← 10-K ingestion
```

**Stack:** Python 3.11+, LangGraph, LangChain, Claude (langchain-anthropic), yfinance, Pydantic, FastMCP, FastAPI, Chroma/FAISS, Docker, LangSmith.

## 8. Success metrics

| Metric | Target |
|---|---|
| Numeric accuracy (memo figures match tool data) | ≥ 95% |
| Critic catch-rate (injected-error detection) | ≥ 80% |
| Memo completeness (all 6 sections present) | 100% |
| Latency (end-to-end) | < 120 s |
| Cost per memo | < $0.15 |
| Demo reliability (10 consecutive runs succeed) | 10/10 |

## 9. Milestones

| Milestone | Scope | Status |
|---|---|---|
| M1 — Core crew | Repo, tools, LangGraph graph, critic loop, CLI | ✅ Done |
| M2 — MCP | MCP server exposing tools | ✅ Done |
| M3 — RAG | 10-K ingestion (SEC EDGAR → FAISS) + Risk agent citations | ✅ Done |
| M3.5 — Verifier | Self-correcting agent: re-checks all numbers vs fresh tool calls | ✅ Done |
| M4 — Evals + tracing | Harness, metrics report, LangSmith | ⚠️ Partial — harness + reliability done; **LangSmith tracing not implemented** |
| M5 — API + deploy | FastAPI, Docker, cloud URL, README polish, demo video | ⚠️ Partial — API, Docker, live URL and README done; **demo video outstanding** |
| M6 — Voice (optional) | Vapi layer: talk to the analyst from the laptop | Not started (stretch) |

### Measured against §8 success metrics

| Metric | Target | Actual |
|---|---|---|
| Numeric accuracy | ≥ 95% | 94% (NVDA) / 96% (AAPL) — **borderline** |
| Critic catch-rate | ≥ 80% | 100% |
| Memo completeness | 100% | 100% (10/10 runs, all six sections) |
| Latency | < 120 s | mean 86.1s; one run hit 128.5s under CPU contention |
| Cost per memo | < $0.15 | $0.101 mean |
| Demo reliability | 10/10 | 10/10 |

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| yfinance rate limits / breakage | Cache responses in dev; Finnhub as fallback source |
| LLM output drift / non-determinism | Structured (Pydantic) outputs; low temperature; evals to catch regressions |
| Critic loop cost/latency | Hard cap at 1 revision; measure in evals |
| Hallucinated numbers slip through | Prompts forbid invention + critic checks + numeric-accuracy eval |
| Free-tier data quality varies | Memo states "data unavailable" honestly (F12) |

## 11. Open questions

1. Chroma vs. FAISS for the vector store (leaning FAISS — named in target JDs).
2. Run research agents in parallel (latency win) vs. sequential (simplicity) — revisit at M4 with latency data.
3. Which 10-K source: SEC EDGAR full-text (free) vs. pre-downloaded PDFs — decide at M3.
