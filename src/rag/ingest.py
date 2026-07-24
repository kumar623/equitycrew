"""Ingest a company's 10-K into a FAISS vector store.

Two ways to get the filing:
  1. Auto-fetch latest 10-K from SEC EDGAR (free, no key):
       python -m src.rag.ingest --ticker NVDA
  2. Use a local file (txt/html) you downloaded yourself:
       python -m src.rag.ingest --ticker NVDA --file path/to/10k.htm

Output: FAISS index at  data/indexes/<TICKER>/
Embeddings are local (sentence-transformers) — free, no API key.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# SEC requires a descriptive User-Agent with contact info.
SEC_HEADERS = {"User-Agent": "EquityCrew research project krishnakumar623@gmail.com"}
INDEX_DIR = Path("data/indexes")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


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

def build_index(ticker: str, text: str) -> Path:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)
    docs_meta = [{"ticker": ticker.upper(), "chunk": i} for i in range(len(chunks))]
    store = FAISS.from_texts(chunks, get_embeddings(), metadatas=docs_meta)

    out = INDEX_DIR / ticker.upper()
    out.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out))
    (out / "meta.json").write_text(json.dumps({"ticker": ticker.upper(),
                                               "chunks": len(chunks)}))
    return out


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

    print(f"Filing length: {len(text):,} chars. Chunking + embedding (local model)…")
    out = build_index(args.ticker, text)
    print(f"Index saved to {out}")


if __name__ == "__main__":
    main()
