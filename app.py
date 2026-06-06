"""
Podcast Q&A Bot — Flask Backend
================================
- Fetches YouTube transcripts via youtube-transcript-api
- Two-level chunking: semantic chunks (300-600 words) + fine timestamp refinement
- TF-IDF retrieval (scikit-learn) — no heavy embeddings needed
- Groq llama-3.1-8b-instant for answer generation
- Caches transcript + chunks to cache/<video_id>.json
"""

import os
import json
import re
import math
import logging
import threading
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    IpBlocked,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from groq import Groq

# ── Setup ────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback_secret_key")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_file(video_id: str) -> Path:
    return CACHE_DIR / f"{video_id}.json"

# ── Global active tasks (in-memory, for tracking loading progress per video_id) ──
active_tasks = {}
active_tasks_lock = threading.Lock()

# ── YouTube helpers (from test.py patterns) ──────────────────────────────────
def get_youtube_client():
    """Create a YouTubeTranscriptApi instance, optionally with loaded cookies if available."""
    cookies_paths = [
        Path("www.youtube.com_cookies.txt"),
        Path("cookies.txt"),
    ]
    for cp in cookies_paths:
        if cp.exists():
            try:
                import requests
                from http.cookiejar import MozillaCookieJar
                session = requests.Session()
                cj = MozillaCookieJar(str(cp))
                cj.load(ignore_discard=True, ignore_expires=True)
                session.cookies = cj
                logger.info(f"Loaded YouTube authentication cookies from {cp}")
                return YouTubeTranscriptApi(http_client=session)
            except Exception as e:
                logger.warning(f"Failed to load cookies from {cp}: {e}")
                
    logger.info("No YouTube authentication cookies found. Using default client.")
    return YouTubeTranscriptApi()


def extract_video_id(url_or_id: str) -> str:
    """Accept full YouTube URL or bare video ID."""
    url_or_id = url_or_id.strip()
    if "v=" in url_or_id:
        return url_or_id.split("v=")[1].split("&")[0]
    if "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    # Handle youtube.com/shorts/ID
    if "shorts/" in url_or_id:
        return url_or_id.split("shorts/")[1].split("?")[0]
    return url_or_id


def fetch_transcript(video_id: str) -> list[dict]:
    """
    Fetch English transcript. Returns list of:
      {'text': '...', 'start': 0.0, 'duration': 4.5}
    """
    ytt_api = get_youtube_client()
    transcript = ytt_api.fetch(video_id, languages=["en"])
    return [
        {"text": snippet.text, "start": snippet.start, "duration": snippet.duration}
        for snippet in transcript
    ]


# ── Two-level chunking ────────────────────────────────────────────────────────
TARGET_WORDS = 450      # ~400-500 words per chunk (1-3 min)
OVERLAP_WORDS = 75      # ~75-word overlap between chunks


def count_words(text: str) -> int:
    return len(text.split())


def build_chunks(snippets: list[dict]) -> list[dict]:
    """
    Level 1: Group raw snippets into semantic chunks of ~TARGET_WORDS words.
    Each chunk carries the list of its constituent snippets for Level 2 search.
    
    Returns: [{chunk_id, start, end, text, word_count, snippets}, ...]
    """
    chunks = []
    chunk_id = 0
    i = 0

    while i < len(snippets):
        chunk_snippets = []
        word_count = 0
        chunk_start = snippets[i]["start"]

        # Collect snippets until we hit the word target
        while i < len(snippets) and word_count < TARGET_WORDS:
            chunk_snippets.append(snippets[i])
            word_count += count_words(snippets[i]["text"])
            i += 1

        if not chunk_snippets:
            break

        chunk_end = chunk_snippets[-1]["start"] + chunk_snippets[-1]["duration"]
        chunk_text = " ".join(s["text"] for s in chunk_snippets)
        # Clean up transcript artifacts (newlines, extra spaces)
        chunk_text = re.sub(r"\s+", " ", chunk_text).strip()

        chunks.append({
            "chunk_id": chunk_id,
            "start": chunk_start,
            "end": chunk_end,
            "text": chunk_text,
            "word_count": word_count,
            "snippets": chunk_snippets,
        })
        chunk_id += 1

        # Overlap: step back by OVERLAP_WORDS worth of snippets
        # IMPORTANT: only overlap if we haven't exhausted the array —
        # otherwise stepping back causes an infinite loop on the tail.
        if i < len(snippets):
            overlap_words = 0
            while i > 0 and overlap_words < OVERLAP_WORDS:
                i -= 1
                overlap_words += count_words(snippets[i]["text"])

    logger.info(f"Built {len(chunks)} chunks from {len(snippets)} snippets")
    return chunks


# ── TF-IDF index ─────────────────────────────────────────────────────────────
def build_tfidf_index(chunks: list[dict]):
    """Fit TF-IDF vectorizer on chunk texts. Returns (vectorizer, matrix)."""
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,    # log-scale TF
    )
    matrix = vectorizer.fit_transform(texts)
    logger.info(f"TF-IDF index built: {matrix.shape[0]} docs × {matrix.shape[1]} terms")
    return vectorizer, matrix


# ── Level 2: Fine-grained timestamp within a chunk ──────────────────────────
def find_fine_timestamp(query: str, chunk: dict) -> tuple[float, str]:
    """
    Search individual snippets inside the best chunk.
    Returns (start_seconds, hms_string) of the most relevant snippet.
    Falls back to chunk start if no good match.
    """
    snippets = chunk.get("snippets", [])
    if not snippets:
        return chunk["start"], seconds_to_hms(chunk["start"])

    # Score individual snippets with simple word overlap (lightweight)
    query_words = set(re.sub(r"[^\w\s]", "", query.lower()).split())
    
    best_score = -1
    best_start = chunk["start"]

    for snippet in snippets:
        snippet_words = set(re.sub(r"[^\w\s]", "", snippet["text"].lower()).split())
        overlap = len(query_words & snippet_words)
        # Weight by position (prefer earlier mentions slightly)
        score = overlap
        if score > best_score:
            best_score = score
            best_start = snippet["start"]

    return best_start, seconds_to_hms(best_start)


def seconds_to_hms(seconds: float) -> str:
    """Convert float seconds → HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Groq LLM ─────────────────────────────────────────────────────────────────
def ask_groq(question: str, context_chunks: list[dict]) -> str:
    """
    Send question + retrieved context to Groq llama-3.1-8b-instant.
    Returns the answer string.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        return (
            "[Groq API key not configured. Add GROQ_API_KEY to your .env file. "
            "Get a free key at https://console.groq.com]"
        )

    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        start_hms = seconds_to_hms(chunk["start"])
        end_hms = seconds_to_hms(chunk["end"])
        context_parts.append(
            f"[Segment {i} | {start_hms} – {end_hms}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are an expert podcast analyst. "
        "Answer the user's question based ONLY on the provided transcript segments. "
        "Be concise and insightful (2–4 sentences). "
        "If the answer isn't in the transcript, say so honestly. "
        "Do NOT make up information."
    )

    user_prompt = (
        f"Transcript segments:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Groq error: {e}")
        return f"[LLM error: {str(e)}]"


# ── Caching ───────────────────────────────────────────────────────────────────
def save_cache(video_id: str, snippets: list[dict], chunks: list[dict]):
    """Save transcript + chunks to JSON for fast reload."""
    # Don't cache the snippet list inside chunks (redundant + large)
    slim_chunks = [
        {k: v for k, v in c.items() if k != "snippets"}
        for c in chunks
    ]
    cache = {
        "video_id": video_id,
        "snippets": snippets,
        "chunks": slim_chunks,
    }
    cache_file = get_cache_file(video_id)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.info(f"Cache saved → {cache_file}")


def load_cache(video_id: str) -> tuple[list, list] | tuple[None, None]:
    """Load cached transcript + chunks if they exist for this video_id."""
    cache_file = get_cache_file(video_id)
    if not cache_file.exists():
        return None, None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("video_id") != video_id:
            return None, None
        # Re-attach snippets to chunks by time range
        snippets = cache["snippets"]
        chunks = cache["chunks"]
        for chunk in chunks:
            chunk["snippets"] = [
                s for s in snippets
                if chunk["start"] <= s["start"] < chunk["end"]
            ]
        logger.info(f"Cache loaded for {video_id}: {len(chunks)} chunks")
        return snippets, chunks
    except Exception as e:
        logger.warning(f"Cache load failed for {video_id}: {e}")
        return None, None


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    video_id = request.args.get("video_id", "").strip()
    if not video_id:
        return jsonify({"error": "No video_id provided"}), 400

    with active_tasks_lock:
        task = active_tasks.get(video_id)

    if not task:
        # Check if cache file exists for this video
        snippets, chunks = load_cache(video_id)
        if snippets is not None and chunks is not None:
            return jsonify({
                "status": "ready",
                "stage": f"Loaded {len(chunks)} chunks from cache ⚡",
                "video_id": video_id,
                "chunks": len(chunks),
                "snippet_count": len(snippets),
                "duration_hms": seconds_to_hms(
                    snippets[-1]["start"] + snippets[-1]["duration"] if snippets else 0
                ),
                "error": None
            })
        return jsonify({"status": "idle", "stage": "", "video_id": video_id, "error": None})

    snippets = task.get("raw_snippets", [])
    chunks = task.get("chunks", [])

    return jsonify({
        "status":   task["status"],
        "stage":    task["stage"],
        "video_id": task["video_id"],
        "chunks":   len(chunks),
        "snippet_count": len(snippets),
        "duration_hms": seconds_to_hms(
            snippets[-1]["start"] + snippets[-1]["duration"]
            if snippets else 0
        ),
        "error":    task["error"],
    })


@app.route("/api/init", methods=["POST"])
def api_init():
    """
    POST { "url": "<youtube url or video id>" }
    Immediately returns {status: loading, video_id: ...} and processes in a background thread.
    Poll /api/status?video_id=<video_id> to track progress.
    """
    data = request.get_json(force=True)
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    video_id = extract_video_id(url)
    logger.info(f"Initializing for video_id: {video_id}")

    with active_tasks_lock:
        # Prevent double-init if already loading
        if video_id in active_tasks and active_tasks[video_id]["status"] == "loading":
            return jsonify({"status": "loading", "stage": active_tasks[video_id]["stage"], "video_id": video_id}), 202

        # Initialize/reset active task state
        active_tasks[video_id] = {
            "status": "loading",
            "stage": "Starting...",
            "error": None,
            "video_id": video_id,
            "raw_snippets": [],
            "chunks": [],
        }

    def _worker(vid):
        try:
            # Stage 1 — cache check
            with active_tasks_lock:
                active_tasks[vid]["stage"] = "Checking cache..."
            snippets, chunks = load_cache(vid)

            if snippets is None:
                # Stage 2 — connect to YouTube
                with active_tasks_lock:
                    active_tasks[vid]["stage"] = "Connecting to YouTube..."
                logger.info(f"No cache — fetching from YouTube for {vid}...")

                # Stage 3 — download transcript
                with active_tasks_lock:
                    active_tasks[vid]["stage"] = "Downloading transcript (this may take 10–30s)..."
                snippets = fetch_transcript(vid)

                # Stage 4 — chunking
                with active_tasks_lock:
                    active_tasks[vid]["stage"] = f"Chunking {len(snippets)} transcript snippets..."
                chunks = build_chunks(snippets)

                # Stage 5 — save cache
                with active_tasks_lock:
                    active_tasks[vid]["stage"] = "Saving to cache..."
                save_cache(vid, snippets, chunks)
            else:
                with active_tasks_lock:
                    active_tasks[vid]["stage"] = f"Loaded {len(chunks)} chunks from cache ⚡"

            # Done — update state
            with active_tasks_lock:
                active_tasks[vid]["raw_snippets"]  = snippets
                active_tasks[vid]["chunks"]        = chunks
                active_tasks[vid]["status"]        = "ready"
                active_tasks[vid]["stage"]         = f"Ready — {len(chunks)} chunks indexed"
            logger.info(f"Init complete for {vid}.")

        except TranscriptsDisabled:
            err = "Transcripts are disabled for this video."
        except NoTranscriptFound:
            err = "No English transcript found for this video."
        except VideoUnavailable:
            err = "Video is unavailable."
        except (IpBlocked, RequestBlocked):
            err = "YouTube is blocking requests from this IP. Try again in a few minutes."
        except Exception as e:
            err = f"Unexpected error: {str(e)}"
        else:
            return  # success path exits here

        # Error path
        with active_tasks_lock:
            active_tasks[vid]["status"] = "error"
            active_tasks[vid]["stage"]  = err
            active_tasks[vid]["error"]  = err
        logger.error(f"Init failed for {vid}: {err}")

    threading.Thread(target=_worker, args=(video_id,), daemon=True).start()
    return jsonify({"status": "loading", "stage": "Starting...", "video_id": video_id}), 202


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    POST { "question": "...", "video_id": "..." }
    Returns:
      {
        answer, timestamp_s, timestamp_hms,
        chunk_start, chunk_end, chunk_start_hms, chunk_end_hms,
        chunk_text, video_id, top_chunks
      }
    """
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    video_id = data.get("video_id", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not video_id:
        return jsonify({"error": "No video_id provided"}), 400

    logger.info(f"Question for {video_id}: {question}")

    # Load transcript snippets and chunks
    snippets, chunks = None, None
    with active_tasks_lock:
        task = active_tasks.get(video_id)
        if task and task["status"] == "ready":
            snippets = task["raw_snippets"]
            chunks = task["chunks"]

    # Fallback to file cache
    if snippets is None or chunks is None:
        snippets, chunks = load_cache(video_id)

    if snippets is None or chunks is None:
        return jsonify({
            "error": "Transcript not loaded. Please initialize with a video URL first."
        }), 400

    # Level 1: TF-IDF retrieval (dynamic fitting, very fast)
    vectorizer, tfidf_matrix = build_tfidf_index(chunks)
    
    # Retrieve top chunks
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(sims)[::-1][:3]

    top_chunks = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["score"] = float(sims[idx])
        top_chunks.append(chunk)

    if not top_chunks:
        return jsonify({"error": "No relevant content found."}), 404

    best_chunk = top_chunks[0]

    # Level 2: Fine timestamp within best chunk
    fine_ts, fine_hms = find_fine_timestamp(question, best_chunk)

    # Generate answer with Groq
    answer = ask_groq(question, top_chunks)

    # Build response
    result = {
        "answer": answer,
        "timestamp_s": fine_ts,
        "timestamp_hms": fine_hms,
        "chunk_start": best_chunk["start"],
        "chunk_end": best_chunk["end"],
        "chunk_start_hms": seconds_to_hms(best_chunk["start"]),
        "chunk_end_hms": seconds_to_hms(best_chunk["end"]),
        "chunk_text": best_chunk["text"][:500] + ("..." if len(best_chunk["text"]) > 500 else ""),
        "video_id": video_id,
        "relevance_score": round(best_chunk.get("score", 0), 3),
        "top_chunks": [
            {
                "start": c["start"],
                "end": c["end"],
                "start_hms": seconds_to_hms(c["start"]),
                "end_hms": seconds_to_hms(c["end"]),
                "score": round(c.get("score", 0), 3),
                "preview": c["text"][:120] + "...",
            }
            for c in top_chunks
        ],
    }

    logger.info(f"Answer generated for {video_id}. Timestamp: {fine_hms} (score: {best_chunk.get('score', 0):.3f})")
    return jsonify(result)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
