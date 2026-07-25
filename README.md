---
title: EquityCrew
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
short_description: Multi-agent equity research analyst with live data and self-verification
---

# EquityCrew — Multi-Agent Equity Research Analyst

**Live demo: https://srv1855275.hstgr.cloud** — enter a ticker and watch seven
agents research it in real time.

A crew of AI agents that produces an investment research memo on any public company,
using **real-time market data** and a **LangGraph** state graph with a critic→writer
revision loop. Tools are also exposed over **MCP**.

**Stack:** Python · LangGraph · LangChain · Claude (Anthropic) · yfinance · MCP

---

## What it does

Give it a ticker (`NVDA`) and a crew of specialized agents collaborates:

```
START → financials → news → risk → writer → critic ─(approve)→ finalize → END
                                        └─(revise, capped)→ writer
```

- **Financials agent** — pulls live price + fundamentals, summarizes them
- **News & Sentiment agent** — pulls recent headlines, reads sentiment
- **Risk agent** — flags key risks (RAG over the SEC EDGAR 10-K when indexed)
- **Writer agent** — drafts the memo (thesis, bull/bear, rating)
- **Critic agent** — approves or sends back for one revision (structured output)
- **Verifier agent** — re-fetches live data and fact-checks every number in the
  approved memo before it ships (self-correcting; guarded so it can fix the memo
  but never destroy it)

All numbers come from tools — agents never invent figures.

The web UI shows the crew as a live ring: each agent lights up as it runs,
arrows animate the handoffs (including the amber critic→writer revision loop),
and a center gauge fills to 100% as the memo comes together.

---

## Evals (measured, not vibes)

`python -m src.evals.run_evals NVDA` runs the pipeline end to end and then
attacks it: plants numeric errors for the verifier, feeds sabotaged memos to the
critic. Latest report (`data/evals/`, claude-sonnet-5, list pricing):

| metric | NVDA | AAPL | how it's measured |
|---|---|---|---|
| Numeric accuracy | **94%** | **96%** | every financial figure regex-extracted from the memo and matched against tool data |
| Verifier catch-rate | **100%** (4/4) | **100%** (4/4) | numeric errors planted in the memo; counts how many the verifier fixes |
| Critic catch-rate | **100%** | **100%** | sabotaged memos (missing rating, unsupported claim, truncated) must be rejected |
| Latency | **53s** | **54s** | full 7-node run, end to end |
| Cost | **$0.074** | **$0.090** | token usage tracked per LLM call via a LangChain callback |

### Cost engineering

Per-agent cost attribution (the UI shows it live, per agent) found the spend was
not where intuition said. The three "trivial" summarising agents were 15% of the
bill combined, while the verifier alone was 35% — because it echoed the entire
memo back just to fix a couple of numbers.

Two changes, measured rather than assumed:

- **Verifier returns patches, not prose.** It now emits `find`/`replace` pairs
  that Python applies, cutting ~1,400 output tokens per run. This also removed a
  failure mode: a malformed response used to be able to truncate the memo (it
  once did), so the code carried a length guard. Now the worst case is no edit.
- **Model routing per agent.** Financials and news run on Haiku 4.5; the writer,
  critic, verifier and risk agents stay on Sonnet 5, since those are the
  judgement calls and the last line of defence against a wrong number shipping.

Together: **$0.102 → $0.074 per memo (−27%) and 70s → 53s (−24%)**, with the
verifier and critic catch-rates unchanged at 100%.

Honest caveat, also in the report JSON: the critic is strict enough that it
sometimes rejects the clean memo too. In the pipeline that is harmless — it
triggers the capped revision loop — but it means the critic optimizes for recall
over precision, and the harness records that rather than hiding it.

### Reliability

`python -m src.evals.reliability --runs 10` runs the crew end to end repeatedly
and fails a run that raises, returns no memo, or drops any required section.

**10/10 runs succeeded.** Mean latency 86.1s (min 56.1s, max 128.5s), mean cost
$0.101, $1.01 for the batch. Every run triggered exactly one critic revision.

The spread is worth noting: the slowest runs happened while a Docker build was
competing for CPU on the same laptop, and one run lost ~50s to Hugging Face Hub
timeouts while loading the embedding model. The container now sets
`HF_HUB_OFFLINE=1`, since the model is baked into the image and the Hub call
only ever cost time.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add your ANTHROPIC_API_KEY

# test the live-data tools (no API key needed):
python -m src.tools.market_data AAPL

# run the full crew (CLI):
python main.py NVDA

# or run the web app (recommended demo):
uvicorn api:app --reload
# open http://127.0.0.1:8000 — watch agents light up live, memo renders at the end
```

## MCP server

Exposes `price`, `fundamentals`, `news`, and `full_analysis` to any MCP client,
so the same tools that power the graph also work over the protocol.

```bash
python mcp_server.py
```

**Requires Python 3.10+** — the MCP SDK does not publish wheels for 3.9, so on an
older interpreter `pip install -r requirements.txt` will fail on this dependency.
The Docker image runs 3.12, so the container is the reliable way to serve it:

```bash
docker run -i --rm --env-file .env equitycrew python mcp_server.py
```

That command is also what goes in a Claude Desktop MCP config entry (`command:
"docker"`, with the remaining words as `args`).

---

## Project layout

```
equitycrew/
├── main.py                 # CLI entry point
├── mcp_server.py           # MCP server (Claude Desktop, etc.)
├── requirements.txt
├── .env.example
└── src/
    ├── config.py           # env + shared LLM client
    ├── state.py            # shared graph state (TypedDict)
    ├── agents.py           # the agent crew (graph nodes)
    ├── graph.py            # LangGraph wiring
    ├── evals/              # eval harness (accuracy, catch-rates, cost)
    ├── rag/                # 10-K ingestion + FAISS retrieval
    └── tools/
        └── market_data.py  # real-time price / fundamentals / news
```

## Roadmap

- [x] Live-data tools
- [x] LangGraph multi-agent graph with critic loop
- [x] MCP server
- [x] RAG over 10-K for the Risk agent (FAISS + SEC EDGAR, local embeddings)
- [x] Verifier agent — re-checks every number against fresh tool calls (self-correcting)
- [x] Eval harness — numeric accuracy, verifier/critic catch-rates, latency, cost
- [x] FastAPI + live circular pipeline UI (SSE streaming)
- [x] Docker image (CPU-only torch, model + indexes baked in) + Fly/Render configs
- [ ] Live public URL + demo video
- [ ] (Optional) Vapi voice layer — talk to the analyst live

## Deploy

The image is self-contained: CPU-only torch, the MiniLM embedding model, and any
built 10-K indexes are baked in, so a cold container serves RAG-grounded memos
with no runtime downloads. `.env` is never copied into the image — the API key is
injected as a platform secret.

```bash
docker build -t equitycrew .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... equitycrew
```

**Hugging Face Spaces** (what the live demo runs on — free, 16GB RAM, no card).
The Space config lives in this README's frontmatter (`sdk: docker`,
`app_port: 8000`); set `ANTHROPIC_API_KEY` under the Space's *Settings →
Variables and secrets*, then push this repo to the Space remote.

**Fly.io** (deploys straight from this folder, no GitHub needed — needs a card):

```bash
fly launch --no-deploy --copy-config
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly deploy
```

**Render** — push to GitHub, then New → Blueprint (`render.yaml`), and set
`ANTHROPIC_API_KEY` in the dashboard.

Measured on a full run inside the container: **457MB image**, **~400MB
resident**, running as UID 1000. Size the box at **1GB RAM** — a 512MB instance
leaves no headroom once torch and the embedding model load. On a smaller box the
app still runs; the Risk agent just falls back to reasoning over live
fundamentals instead of citing the 10-K.

Two details that matter more than they look: installing CPU-only torch before
`requirements.txt` keeps ~2GB of unusable CUDA libraries out of the image, and
copying with `--chown` instead of a trailing `chown -R` avoids duplicating the
whole app tree into an extra layer (that one alone was 84MB).

---

## RAG: ground the Risk agent in the real 10-K

```bash
# fetch latest 10-K from SEC EDGAR, chunk, embed locally, save FAISS index:
python -m src.rag.ingest --ticker NVDA
# or use a filing you downloaded yourself:
python -m src.rag.ingest --ticker NVDA --file path/to/10k.htm
```

Once an index exists in `data/indexes/<TICKER>/`, the Risk agent automatically
retrieves and cites real risk-factor passages (falls back gracefully otherwise).
NVDA's index is built from a 364K-character filing; the agent cites passages as
`[chunk 112]`, e.g. NVIDIA's dependence on third-party foundries and its export-
control exposure — language taken from the 10-K, not from the model.

> Educational project. Not investment advice.
