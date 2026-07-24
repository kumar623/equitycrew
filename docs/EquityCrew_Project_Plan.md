# EquityCrew — Multi-Agent Equity Research Analyst

**A LangGraph-orchestrated crew of AI agents that produces an investment research memo on any public company, using real-time market data and live news.**

Built with: **Python · LangChain · LangGraph · Claude (Anthropic) · real-time financial APIs**

This is your flagship portfolio project for AI / Generative-AI Engineer interviews. It demonstrates multi-agent orchestration, RAG over filings, real-time tool use, structured output, evaluation, and observability.

---

## 1. What it does

Input: a company ticker (e.g. `NVDA`).

Output: a structured research memo containing an investment thesis, the fundamentals, recent news & sentiment, key risks, a bull/bear case, and a rating — assembled by a crew of specialized agents that plan, delegate, critique, and revise.

The system pulls **live data** (current price, latest financials, breaking news) so every run is fresh — a strong "this is real, not a demo" signal.

---

## 2. The agent crew

Each agent is a node in a LangGraph state graph. They share a typed state object and hand off work.

| Agent | Role | Tools / data |
|---|---|---|
| **Orchestrator / Planner** | Parses the request, builds a plan, routes to the right agents, assembles the final memo | LangGraph routing |
| **Financials agent** | Fetches and summarizes fundamentals (revenue, margins, growth, valuation) | Real-time financial data API |
| **News & Sentiment agent** | Gathers recent headlines, scores sentiment, flags catalysts | Real-time news API |
| **Risk agent** | Extracts risk factors and red flags from the 10-K | RAG over filing (vector store) |
| **Writer agent** | Synthesizes everything into a clean memo with thesis + bull/bear + rating | Claude |
| **Critic agent** | Reviews the draft for unsupported claims / missing data; triggers ONE revision loop | Claude |

The **critic → writer revision cycle** is the standout feature — it turns a linear pipeline into a real agentic graph with a loop, which is what senior interviewers look for.

---

## 3. Architecture (LangGraph state graph)

```
                         ┌──────────────┐
        ticker  ───────► │  Orchestrator │
                         │   / Planner   │
                         └──────┬────────┘
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
     ┌────────────────┐ ┌───────────────┐ ┌──────────────┐
     │  Financials    │ │ News &        │ │  Risk        │
     │  agent         │ │ Sentiment     │ │  agent       │
     │ (live data)    │ │ agent (live)  │ │ (RAG on 10-K)│
     └───────┬────────┘ └───────┬───────┘ └──────┬───────┘
             └───────────────┬──┴────────────────┘
                             ▼
                     ┌───────────────┐
                     │  Writer agent │◄───────┐
                     └───────┬───────┘        │  revise (max 1x)
                             ▼                │
                     ┌───────────────┐        │
                     │  Critic agent │────────┘
                     └───────┬───────┘
                             ▼  (approved)
                     ┌───────────────┐
                     │  Final Memo   │
                     └───────────────┘

  State (shared): ticker, financials, news, risks, draft, critique, revision_count
  Cross-cutting: streaming updates • tracing (LangSmith) • evals • guardrails
```

**Why LangGraph over a linear chain:** you need branching (run three research agents), a conditional loop (critic decides revise vs. approve), and shared state. LangGraph models exactly this; a plain LangChain chain can't cleanly express the loop.

---

## 4. Real-time components ("the real-time things")

1. **Live market data** — current price, key financials, valuation ratios, fetched at request time (e.g. `yfinance`, or Finnhub / Alpha Vantage free tier).
2. **Live news feed** — latest headlines for the ticker at request time (Finnhub news, NewsAPI, or RSS).
3. **Streaming output** — stream tokens and per-agent progress to the UI as the crew works, so the user watches "Financials agent → News agent → Writer…" live. LangGraph supports streaming state updates natively.
4. (Stretch) **Live tracing dashboard** — LangSmith run view showing each agent call in real time.

---

## 5. Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| LLM | Claude (Anthropic API) via `langchain-anthropic` |
| Orchestration | **LangGraph** (state graph, cycles, streaming) |
| LLM tooling | **LangChain** (tools, prompt templates, output parsers) |
| Market data | `yfinance` (free) or Finnhub / Alpha Vantage |
| News | Finnhub / NewsAPI / RSS |
| RAG (10-K) | LangChain loaders + Chroma vector store + Claude |
| Structured output | Pydantic models + LangChain structured output |
| Observability | LangSmith tracing |
| API | FastAPI (streaming endpoint) |
| UI | Streamlit (live agent progress + memo render) |
| Evals | Custom harness (numbers accuracy, critic catch-rate) |
| Deploy | Docker + Render / Railway / Fly.io |

---

## 6. Phased roadmap

Each phase is demoable on its own.

### Phase 0 — Setup
Repo, venv, `.env` (Anthropic key + data API keys), project structure, README skeleton. Verify a basic Claude call via `langchain-anthropic`.

### Phase 1 — Tools + real-time data
Build and test standalone LangChain tools: `get_financials(ticker)`, `get_news(ticker)`, `get_price(ticker)`. Confirm they return live data. **Milestone: live data flowing.**

### Phase 2 — Single agents
Implement Financials, News/Sentiment, and Risk agents as functions that call Claude + their tools and return structured (Pydantic) results. Test each in isolation. **Milestone: each agent works alone.**

### Phase 3 — LangGraph orchestration
Define the shared state, wire the three research agents to run, then Writer, then Critic with the conditional revision loop. **Milestone: full crew produces a memo end to end.**

### Phase 4 — RAG for the Risk agent
Ingest the company's 10-K into Chroma; Risk agent retrieves + cites real risk factors. **Milestone: grounded risk section with citations.**

### Phase 5 — Streaming API + UI
FastAPI streaming endpoint; Streamlit UI that shows agents working live and renders the final memo. **Milestone: watchable live demo.**

### Phase 6 — Evals + observability
LangSmith tracing on. Eval set: check reported numbers vs. ground truth; inject errors into a draft and measure whether the critic catches them. Produce a results table. **Milestone: you can quantify quality.**

### Phase 7 — Polish + deploy
Dockerize, deploy, public URL. README with architecture diagram, demo GIF, eval results, sample memo.

### Stretch
- Comparison mode (two tickers side by side).
- Human-in-the-loop approval node before the memo finalizes.
- Portfolio-level analysis across several tickers.
- Expose the tools via an MCP server too (works in Claude Desktop).

---

## 7. Interview talking points

- **"Why multi-agent / why LangGraph?"** Branching research + a critic-revise cycle needs a graph with conditional edges and shared state — not a linear chain.
- **"How do you handle non-determinism / eval a multi-agent system?"** Scoped task, structured (Pydantic) outputs, a critic node, and an eval harness that (a) checks numeric facts against ground truth and (b) measures the critic's error-catch rate on injected mistakes.
- **"How do agents communicate?"** Through the shared typed state object in LangGraph; each node reads what it needs and writes its slice.
- **"How do you prevent hallucinated numbers?"** Numbers come from real-time tools, not the model; the critic flags any claim not backed by fetched data.
- **"What was hard?"** Real war stories: controlling the revision loop so it terminates, keeping agents from duplicating work, parsing messy live news, latency of running agents sequentially vs. in parallel.
- **"How would you scale / productionize?"** Run research agents in parallel, cache data calls, add rate limiting, stream results, add retries + fallbacks.

Have metrics ready: number of agents, avg memo latency, critic catch-rate %, cost per memo.

---

## 8. Pitfalls to avoid

- Don't let the critic loop run forever — cap revisions (e.g. 1) with a counter in state.
- Don't let the model invent numbers — always source them from tools and cite.
- Don't run everything sequentially if you can parallelize the three research agents (good talking point either way).
- Keep the Streamlit UI simple; the memo + live agent progress is the wow factor.
- Never commit API keys — use `.env` + `.gitignore`.
- Free data APIs have rate limits — cache during development.

---

## 9. Next step

Start with **Phase 0 + Phase 1**: repo scaffold and the three live-data tools, so you see real market data flowing before any agent logic. Say the word and I'll scaffold the repo structure and write the Phase 1 tools.
