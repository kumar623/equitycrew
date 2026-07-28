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
**Source: https://github.com/kumar623/equitycrew**

A crew of AI agents that produces an investment research memo on any public company,
using **real-time market data** and a **LangGraph** state graph with a critic→writer
revision loop. Tools are also exposed over **MCP**.

**Stack:** Python · LangGraph · LangChain · Claude (Anthropic) · yfinance · MCP

---

## What it does

Give it a ticker (`NVDA`) and a crew of specialized agents collaborates:

```
START ─┬→ financials ─┐
       ├→ news       ─┼→ writer → critic ─(approve)→ verifier → finalize → END
       └→ risk       ─┘            └─(revise, capped)→ writer
       (parallel: no shared data)
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
| Latency | **37s** | **54s** | full 7-node run, end to end |
| Cost | **$0.068** | **$0.090** | token usage tracked per LLM call via a LangChain callback |

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

**Parallel research.** The financials, news and risk agents share no data, so
they now fan out from START and the writer waits for all three. Verified from
per-node timings: the three finish within 2.9s of each other, so the research
phase costs the slowest agent rather than the sum. What remains is writer →
critic → verifier, which is inherently sequential.

### The critic, and a bug worth documenting

The revision loop did not work for most of this project's life. `route_after_critic`
compared `revision_count >= MAX_REVISIONS`, but the critic increments that counter
for the very rejection being routed on — so with a budget of one revision, the
first rejection already looked like the budget was spent and the graph skipped
straight to the verifier. **The writer never saw the feedback.** The symptom was
visible in the metrics all along: every run was exactly 6 LLM calls, one per agent.
The comparison is now `>`, and an integration test with stub nodes asserts the
writer actually runs twice.

Fixing that exposed a second problem. The critic rejected almost every draft,
demanding citations for figures that came from live tool calls — because it only
ever received the draft, never the research, so it could not tell a grounded
number from an invented one. It now receives both, and is told that a verifier
re-checks every figure downstream.

Section completeness moved out of the prompt entirely. Whether a memo contains a
rating is a fact, not a judgement, and the LLM proved unreliable at it — it read a
memo with the rating stripped out and approved it. `structural_defects()` checks
that in code, so those defects are impossible to miss and cost no API call. The
LLM critic now judges only what needs judgement: unsupported claims and
contradictions.

Result: catch-rate went from 67% to 100% while the clean memo went from rejected
to approved — fewer false positives *and* fewer false negatives.

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

The classic pipeline — **chunk → embed → top-k → LLM** — with every stage visible
in the UI while it runs.

```bash
# fetch latest 10-K from SEC EDGAR, chunk, embed locally, index it:
python -m src.rag.ingest --ticker NVDA

# same corpus in Chroma instead of FAISS — one env var, no code change:
VECTOR_STORE=chroma python -m src.rag.ingest --ticker NVDA

# or use a filing you downloaded yourself:
python -m src.rag.ingest --ticker NVDA --file path/to/10k.htm
```

| Stage | What happens |
|---|---|
| **Chunk** | `RecursiveCharacterTextSplitter`, 1200 chars with 200 overlap, so a risk factor split across a boundary still lands whole in one chunk |
| **Embed** | `all-MiniLM-L6-v2` locally — no embedding API cost, model cached process-wide |
| **Store** | FAISS (default) or Chroma, selected by `VECTOR_STORE`. Same interface, different tradeoffs: FAISS is an in-process index over a fixed corpus; Chroma is a persistent collection with metadata filters and live upserts |
| **Top-k** | k=4 nearest chunks. Lower starves the model of context, higher pays tokens for noise |
| **LLM** | Answers from those chunks only, citing them as `[chunk 112]` |

Once an index exists the Risk agent uses it automatically and falls back to live
fundamentals when no filing is indexed — it never fails the run. NVDA's index is
built from a 364K-character filing; the agent cites passages like NVIDIA's
dependence on third-party foundries and its export-control exposure — language
taken from the 10-K, not from the model.

**Watch it happen.** Click the Risk agent while a run is in flight and the
inspector shows which store answered, the retrieval latency, and the actual
passages handed to the model with their similarity scores — the retrieval step
is shown, not asserted.

> Educational project. Not investment advice.
