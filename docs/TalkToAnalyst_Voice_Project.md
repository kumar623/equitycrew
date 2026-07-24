# TalkToAnalyst — Real-Time Voice AI Equity Analyst

**A voice agent you can talk to live from your laptop.** You speak, it fetches real-time market data through custom tools, runs a multi-agent analysis, and talks the answer back — while your screen shows the live transcript and data being pulled.

Built with: **Vapi (voice) · Claude · FastAPI · LangGraph · real-time financial APIs**

This is your demo-first flagship: a genuinely impressive, real-time, agentic project that also proves the whole JD skill set (voice AI, function calling, agents, real-time tools, APIs, deployment).

---

## 1. Why this is a great interview project

- **It's live.** You open your laptop, click "Talk", and have a real spoken conversation in front of the interviewer. Almost no candidate does this.
- **It's agentic.** The voice layer calls real tools mid-conversation and can trigger a multi-agent backend — not a scripted bot.
- **It's real-time.** Live price + news, streamed responses, on-screen data updating as you speak.
- **It covers the JDs:** voice AI + function calling + agents (LangGraph) + RAG + FastAPI + real-time data + deployment.

---

## 2. How Vapi works (the mental model)

Vapi runs a real-time **listen → think → speak** loop and handles telephony/websockets/orchestration for you. You plug in swappable components:

- **Listen (STT):** Deepgram (recommended), AssemblyAI, or OpenAI.
- **Think (LLM):** your model — we use **Claude**. This is where **function/tool calling** happens: the model decides to call one of *your* tools.
- **Speak (TTS):** ElevenLabs / PlayHT / Deepgram voices.

**Custom tools** are the key feature: you register tool schemas in the Vapi assistant; when the model calls one, Vapi sends a webhook to **your backend**, your backend returns data, and the model speaks it. That backend is where your real work lives.

**Two ways to talk to it:**
1. **Web SDK (recommended for demo):** talk through the laptop mic in the browser — no phone number, minimal cost.
2. **Phone number:** real inbound/outbound calls (costs more; great "wow" but optional).

**Cost note:** new accounts get ~$10 free credit. Pricing is ~\$0.05/min orchestration **plus** STT + TTS + LLM usage. For demos this is a few dollars. Use the Web SDK to avoid phone-number charges. *(Verify current pricing before you rely on it.)*

---

## 3. Architecture

```
   ┌──────────────────────────────┐
   │   Laptop browser (demo)      │
   │   "Talk" button · Vapi Web   │
   │   SDK · live transcript +    │
   │   data panel                 │
   └───────────────┬──────────────┘
                   │ mic audio / events
                   ▼
        ┌────────────────────┐        listen → think → speak
        │       VAPI         │  STT (Deepgram) → LLM (Claude) → TTS (ElevenLabs)
        │  (voice orchestr.) │
        └─────────┬──────────┘
                  │ tool call (webhook) when user asks about a stock
                  ▼
        ┌────────────────────────────┐
        │   Your FastAPI backend     │
        │   /tools/get_price         │──► yfinance / Finnhub (LIVE)
        │   /tools/get_news          │──► news API (LIVE)
        │   /tools/get_fundamentals  │──► financial data (LIVE)
        │   /tools/get_analysis      │──► LangGraph EquityCrew (multi-agent)
        └────────────────────────────┘
```

The FastAPI backend is shared with the EquityCrew project — the voice layer is a new front-end on top of the same agentic engine.

---

## 4. Tech stack

| Layer | Choice |
|---|---|
| Voice orchestration | **Vapi** |
| STT | Deepgram |
| LLM | **Claude (Anthropic)** |
| TTS | ElevenLabs or PlayHT |
| Backend | **FastAPI** (tool webhooks) |
| Tunnel (dev) | ngrok (expose localhost to Vapi) |
| Real-time data | yfinance (free) / Finnhub |
| Multi-agent engine | **LangGraph + LangChain** (EquityCrew) |
| Demo frontend | HTML/JS + **Vapi Web SDK** |
| Deploy | Docker + Render / Railway / a cloud VM |

---

## 5. Build roadmap

### Phase 1 — Vapi hello-world (get talking fast)
- Create a Vapi account; create an assistant (Deepgram + Claude + a TTS voice).
- Write the system prompt: *"You are a concise, friendly equity research analyst. Keep answers short and speakable. Always use tools for live numbers; never guess prices."*
- Talk to it from the Vapi dashboard / Web SDK. **Milestone: you can have a spoken conversation.**

### Phase 2 — Backend tools (real-time data)
- FastAPI with `/tools/get_price`, `/tools/get_news`, `/tools/get_fundamentals`.
- Each returns clean JSON from live sources (yfinance/Finnhub).
- Expose via ngrok so Vapi can reach it. **Milestone: endpoints return live data over the internet.**

### Phase 3 — Wire tools into Vapi
- Register the tool schemas in the assistant (name, description, parameters like `ticker`), pointing at your ngrok URL.
- Test: ask "What's Apple trading at?" → the model calls `get_price` → speaks the live number. **Milestone: voice agent uses live tools.**

### Phase 4 — Laptop demo UI
- Web page with a **Talk** button (Vapi Web SDK) that starts a mic session.
- Show the **live transcript** and a **data panel** that displays which tool fired and what it returned. **Milestone: a watchable, screen-shareable demo.**

### Phase 5 — Multi-agent depth
- Add `/tools/get_analysis` that runs the **LangGraph EquityCrew** (financials + news + risk + writer + critic) and returns a short spoken-friendly thesis.
- Now "Give me your take on Tesla" triggers a real multi-agent workflow. **Milestone: voice + agents combined.**

### Phase 6 — Polish + deploy
- Dockerize the backend, deploy to a cloud host (stable public URL instead of ngrok), point Vapi at it.
- README: architecture diagram, demo GIF/video, setup steps. Record a 60–90s demo.
- (Optional) enable a phone number for a real "call it live" moment.

### Stretch
- **Barge-in / interruptions** handling (Vapi supports it) — feels natural, good talking point.
- **Conversation memory** across turns.
- **Guardrails:** disclaimers ("not financial advice"), refuse if data unavailable.
- **Outbound call demo:** the agent calls *your* phone and briefs you on a stock.
- **Latency tuning:** measure and reduce time-to-first-word (great engineering story).

---

## 6. What to say in the interview

- **"Walk me through it."** Use the diagram: Vapi handles the voice loop; the LLM calls my FastAPI tools; tools return live data; one tool runs a LangGraph multi-agent crew.
- **"How does function calling work here?"** The model decides to call a tool, Vapi webhooks my backend with the args, I return JSON, the model speaks it. I never let it invent numbers.
- **"How do you keep latency low?"** Streaming, fast STT/TTS, keeping tool responses small, caching data calls. I measured time-to-first-word.
- **"How do you handle errors / no data?"** Tools return a clear "unavailable" and the agent says so rather than hallucinating.
- **"Why Vapi?"** It exposes every layer (STT/LLM/TTS) as swappable and handles real-time orchestration, so I focus on the agent logic and tools.
- **"How would you productionize?"** Stable hosted backend, rate limiting, retries/fallbacks, monitoring call quality and latency, cost tracking per call.

Have metrics ready: avg response latency, cost per call, which tools fire most.

---

## 7. Costs & accounts you'll need

- **Vapi** account (~$10 free credit to start).
- **Anthropic** API key (Claude).
- **Deepgram** (STT) and **ElevenLabs/PlayHT** (TTS) keys — often small free tiers.
- **Finnhub** (free tier) or just **yfinance** (no key) for market data.
- **ngrok** (free) for local dev.

Keep every key in `.env`, never in the repo.

---

## 8. Immediate next steps

1. Create the Vapi account and confirm you can talk to a basic assistant.
2. I'll scaffold the FastAPI backend with the three live-data tools + the ngrok setup.
3. Wire the tools into Vapi and do the first "ask about a stock, hear a live number" test.

Tell me when your Vapi account is ready (or if you want me to scaffold the backend first so it's waiting), and we'll start Phase 1–2.
