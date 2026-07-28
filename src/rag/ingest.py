"""Ingest a company's 10-K into a FAISS vector store.

Two ways to get the filing:
  1. Auto-fetch latest 10-K from SEC EDGAR (free, no key):
       python -m src.rag.ingest --ticker NVDA
  2. Use a local file (txt/html) you downloaded yourself:
       python -m src.rag.ingest --ticker NVDA --file path/to/10k.htm

Vector store is swappable — FAISS (default) or Chroma:
       VECTOR_STORE=chroma python -m src.rag.ingest --ticker NVDA

Embeddings are local (sentence-transformers) — free, no API key.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import store
from .store import EMBED_MODEL, INDEX_DIR, get_embeddings  # re-exported

# SEC requires a descriptive User-Agent with contact info.
SEC_HEADERS = {"User-Agent": "EquityCrew research project krishnakumar623@gmail.com"}

# Chunking: big enough that a risk factor survives intact, with overlap so a
# point split across a boundary still lands whole in one of the two chunks.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# ---------- EDGAR fetch ----------

def _cik_for_ticker(ticker: str) -> str:
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    for row in r.json().values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker} not found on EDGAR")


def fetch_latest_10k(ticker: str) -> str:
    """Download the latest 10-K primary document text from SEC EDGAR."""
    cik = _cik_for_ticker(ticker)
    subs = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                        headers=SEC_HEADERS, timeout=30).json()
    recent = subs["filings"]["recent"]
    for form, acc, doc in zip(recent["form"], recent["accessionNumber"],
                              recent["primaryDocument"]):
        if form == "10-K":
            acc_nodash = acc.replace("-", "")
            url = (f"https://www.sec.gov/Archives/edgar/data/"
                   f"{int(cik)}/{acc_nodash}/{doc}")
            html = requests.get(url, headers=SEC_HEADERS, timeout=60).text
            return _html_to_text(html)
    raise ValueError(f"No 10-K found for {ticker}")


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text)


# ---------- build index ----------

def build_index(ticker: str, text: str) -> str:
    """Chunk -> embed -> persist. Returns where it landed."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)
    metas = [{"ticker": ticker.upper(), "chunk": i} for i in range(len(chunks))]

    location = store.build(ticker, chunks, metas)

    # Sidecar manifest so the app can report corpus size without loading it.
    out = INDEX_DIR / ticker.upper()
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({
        "ticker": ticker.upper(), "chunks": len(chunks),
        "store": store.backend(), "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
    }, indent=2))
    return location


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--file", help="Optional local 10-K file (txt/htm/html)")
    args = p.parse_args()

    if args.file:
        raw = Path(args.file).read_text(errors="ignore")
        text = _html_to_text(raw) if args.file.endswith(("htm", "html")) else raw
    else:
        print(f"Fetching latest 10-K for {args.ticker} from SEC EDGAR…")
        text = fetch_latest_10k(args.ticker)

    print(f"Filing length: {len(text):,} chars")
    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) + embedding "
          f"locally into {store.backend()}…")
    out = build_index(args.ticker, text)
    print(f"Index saved to {out}")


if __name__ == "__main__":
    main()
