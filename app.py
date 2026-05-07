"""
Movie Metadata RAG Assistant
============================

Flask web app that answers movie-related questions by:
  1. Retrieving the top-k matching rows from movies_metadata.csv via
     TF-IDF + cosine similarity (RAG retrieval stage).
  2. Building a compact grounded context block from those rows.
  3. Sending the question + context to a locally hosted Ollama LLM
     (model: minimax-m2.1:cloud) at http://localhost:11434/api/generate.
  4. Rendering the retrieved rows AND the grounded answer in the UI.

Run:
    python app.py
Then open: http://127.0.0.1:5005
"""

from __future__ import annotations

import io
import os
import logging
from typing import List, Dict, Any, Tuple

import pandas as pd
import requests
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_URL = "https://hiperc.buffalostate.edu/courses/movies_metadata.csv"
LOCAL_CACHE = "movies_metadata.csv"  # cache so we don't re-download every run

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "minimax-m2.1:cloud"
OLLAMA_TIMEOUT = 120  # seconds

USEFUL_COLUMNS = ["title", "overview", "genres", "release_date", "vote_average"]
TOP_K_DEFAULT = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("movie_rag")


# ---------------------------------------------------------------------------
# Data loading & retrieval
# ---------------------------------------------------------------------------

def load_movies() -> pd.DataFrame:
    """Load and clean the movies metadata dataset.

    Caches the CSV locally on first run, then keeps only the useful columns,
    drops rows with missing title/overview, and normalizes the genres column
    (which in the source CSV is a stringified list of dicts).
    """
    if os.path.exists(LOCAL_CACHE):
        log.info("Loading movies from local cache: %s", LOCAL_CACHE)
        df = pd.read_csv(LOCAL_CACHE, low_memory=False)
    else:
        log.info("Downloading movies dataset from %s", DATASET_URL)
        resp = requests.get(DATASET_URL, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        try:
            df.to_csv(LOCAL_CACHE, index=False)
        except OSError as exc:
            log.warning("Could not cache CSV locally: %s", exc)

    keep = [c for c in USEFUL_COLUMNS if c in df.columns]
    df = df[keep].copy()

    df = df.dropna(subset=["title", "overview"])
    df["title"] = df["title"].astype(str).str.strip()
    df["overview"] = df["overview"].astype(str).str.strip()
    df = df[(df["title"] != "") & (df["overview"] != "")]

    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(_normalize_genres)
    if "release_date" in df.columns:
        df["release_date"] = df["release_date"].astype(str).str.strip()
    if "vote_average" in df.columns:
        df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")

    df = df.reset_index(drop=True)
    log.info("Loaded %d movies after cleaning.", len(df))
    return df


def _normalize_genres(value: Any) -> str:
    """Convert the raw genres field (e.g. "[{'id':18,'name':'Drama'}]") to
    a comma-separated string like "Drama, Romance"."""
    if pd.isna(value):
        return ""
    s = str(value)
    try:
        import ast
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            names = [
                item.get("name", "")
                for item in parsed
                if isinstance(item, dict) and item.get("name")
            ]
            return ", ".join(names)
    except (ValueError, SyntaxError):
        pass
    return s


def _build_search_corpus(df: pd.DataFrame) -> List[str]:
    """Concatenate the searchable text fields per row for TF-IDF indexing."""
    parts: List[pd.Series] = [
        df["title"].fillna("").astype(str),
        df["overview"].fillna("").astype(str),
    ]
    if "genres" in df.columns:
        parts.append(df["genres"].fillna("").astype(str))

    columns = [p.tolist() for p in parts]
    return [" ".join(values).strip() for values in zip(*columns)]


# Build the index ONCE at startup so each request is fast.
log.info("Initializing movie data + TF-IDF index...")
MOVIES_DF: pd.DataFrame = load_movies()
_CORPUS: List[str] = _build_search_corpus(MOVIES_DF)
_VECTORIZER = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50_000,
    min_df=2,
)
_DOC_MATRIX = _VECTORIZER.fit_transform(_CORPUS)
log.info("TF-IDF index ready: %d docs x %d features",
         _DOC_MATRIX.shape[0], _DOC_MATRIX.shape[1])


def retrieve_movies(question: str, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    """Return the top-k movies most similar to `question` along with scores."""
    q = (question or "").strip()
    if not q:
        return []

    q_vec = _VECTORIZER.transform([q])
    sims = cosine_similarity(q_vec, _DOC_MATRIX).ravel()

    if sims.size == 0:
        return []

    top_idx = sims.argsort()[::-1][:top_k]
    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(top_idx, start=1):
        score = float(sims[idx])
        if score <= 0:
            continue
        row = MOVIES_DF.iloc[int(idx)]
        results.append({
            "rank": rank,
            "score": round(score, 4),
            "title": row.get("title", ""),
            "overview": row.get("overview", ""),
            "genres": row.get("genres", "") if "genres" in MOVIES_DF.columns else "",
            "release_date": row.get("release_date", "") if "release_date" in MOVIES_DF.columns else "",
            "vote_average": (
                None if pd.isna(row.get("vote_average"))
                else float(row.get("vote_average"))
            ) if "vote_average" in MOVIES_DF.columns else None,
        })
    return results


# ---------------------------------------------------------------------------
# RAG context + LLM call
# ---------------------------------------------------------------------------

def build_context(rows: List[Dict[str, Any]]) -> str:
    """Compact, structured context block sent to the LLM."""
    if not rows:
        return "NO_CONTEXT"

    lines: List[str] = []
    for r in rows:
        rating = r["vote_average"]
        rating_s = f"{rating:.1f}" if isinstance(rating, (int, float)) and rating is not None else "N/A"
        overview = (r.get("overview") or "").replace("\n", " ").strip()
        if len(overview) > 500:
            overview = overview[:500].rstrip() + "..."
        lines.append(
            f"[{r['rank']}] Title: {r.get('title','')}\n"
            f"    Genres: {r.get('genres','') or 'N/A'}\n"
            f"    Release: {r.get('release_date','') or 'N/A'} | Rating: {rating_s}\n"
            f"    Overview: {overview}"
        )
    return "\n\n".join(lines)


def ask_ollama(question: str, context: str) -> Tuple[str, str]:
    """POST the grounded prompt to Ollama. Returns (answer, error)."""
    prompt = (
        "You are a helpful movie assistant. Answer the user's question USING ONLY "
        "the information in the CONTEXT below. If the context does not contain "
        "enough information to answer, reply exactly: "
        "\"I don't have enough information in the retrieved movies to answer that.\" "
        "Do not invent titles, ratings, or facts. Cite movie titles you reference.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return "", (
            "Could not reach Ollama at http://localhost:11434. "
            "Make sure the Ollama service is running (`ollama serve`) and the "
            f"model '{OLLAMA_MODEL}' is pulled."
        )
    except requests.exceptions.Timeout:
        return "", f"Ollama request timed out after {OLLAMA_TIMEOUT}s."
    except requests.exceptions.RequestException as exc:
        return "", f"Network error talking to Ollama: {exc}"

    if resp.status_code != 200:
        return "", f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}"

    try:
        data = resp.json()
    except ValueError:
        return "", f"Ollama returned non-JSON response: {resp.text[:300]}"

    answer = (data.get("response") or "").strip()
    if not answer:
        return "", f"Ollama returned an empty response. Raw: {str(data)[:300]}"
    return answer, ""


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    rows: List[Dict[str, Any]] = []
    answer = ""
    error = ""
    context_preview = ""

    if request.method == "POST":
        question = (request.form.get("question") or "").strip()
        if not question:
            error = "Please enter a question before submitting."
        else:
            try:
                rows = retrieve_movies(question, top_k=TOP_K_DEFAULT)
            except Exception as exc:
                log.exception("Retrieval failed")
                error = f"Retrieval failed: {exc}"
                rows = []

            if not rows and not error:
                error = "No matching movies were found in the dataset for that question."

            if rows:
                context_preview = build_context(rows)
                answer, llm_err = ask_ollama(question, context_preview)
                if llm_err:
                    error = llm_err

    return render_template(
        "index.html",
        question=question,
        rows=rows,
        answer=answer,
        error=error,
        context_preview=context_preview,
        model=OLLAMA_MODEL,
        dataset_size=len(MOVIES_DF),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005, debug=False)
