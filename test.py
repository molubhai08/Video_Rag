"""
YouTube Transcript API - Python Examples
=========================================
Requires: pip install --upgrade youtube-transcript-api
Tested on: youtube-transcript-api >= 1.2.4, Python 3.8+

Key changes from 0.6.x → 1.x:
  - No more static methods (YouTubeTranscriptApi.get_transcript etc.)
  - Always instantiate:  ytt_api = YouTubeTranscriptApi()
  - proxy_config= parameter (not proxies=)
  - .fetch() returns a FetchedTranscript object (iterable, not a plain list)
"""

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    IpBlocked,
)
from youtube_transcript_api._transcripts import FetchedTranscript


# ── Shared API instance (no proxy) ─────────────────────────────────────────
ytt_api = YouTubeTranscriptApi()


# ── Helper: extract video ID from URL or bare ID ───────────────────────────
def extract_video_id(url_or_id: str) -> str:
    """Accepts a full YouTube URL or a bare video ID."""
    if "v=" in url_or_id:
        return url_or_id.split("v=")[1].split("&")[0]
    if "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    return url_or_id


# ── 1. Basic transcript fetch ───────────────────────────────────────────────
def get_transcript(video_url_or_id: str) -> FetchedTranscript:
    """
    Returns a FetchedTranscript (iterable of snippet dicts):
      [{'text': '...', 'start': 0.0, 'duration': 4.5}, ...]
    Default language: English.
    """
    video_id = extract_video_id(video_url_or_id)
    return ytt_api.fetch(video_id)


# ── 2. Get transcript as plain text ────────────────────────────────────────
def get_plain_text(video_url_or_id: str) -> str:
    transcript = get_transcript(video_url_or_id)
    return " ".join(snippet.text for snippet in transcript)


# ── 3. Get transcript in a specific language ────────────────────────────────
def get_transcript_in_language(
    video_url_or_id: str, language_code: str = "en"
) -> FetchedTranscript:
    """
    language_code examples: 'en', 'hi', 'fr', 'de', 'es', 'ja'
    Falls back to English if the requested language is unavailable.
    """
    video_id = extract_video_id(video_url_or_id)
    return ytt_api.fetch(video_id, languages=[language_code, "en"])


# ── 4. List all available transcripts for a video ──────────────────────────
def list_available_transcripts(video_url_or_id: str):
    video_id = extract_video_id(video_url_or_id)
    transcript_list = ytt_api.list(video_id)

    print(f"\nAvailable transcripts for: {video_id}")
    for t in transcript_list:
        kind = "auto-generated" if t.is_generated else "manual"
        print(f"  [{kind}] {t.language} ({t.language_code})")

    return transcript_list


# ── 5. Translate a transcript ───────────────────────────────────────────────
def get_translated_transcript(
    video_url_or_id: str, target_language: str = "en"
) -> FetchedTranscript:
    """
    Finds the first available transcript and translates it.
    target_language: BCP-47 code, e.g. 'en', 'hi', 'fr'
    """
    video_id = extract_video_id(video_url_or_id)
    transcript_list = ytt_api.list(video_id)
    transcript = transcript_list.find_transcript(["en", "hi"])
    return transcript.translate(target_language).fetch()


# ── 6. Fetch transcripts for multiple videos ───────────────────────────────
def get_batch_transcripts(video_urls_or_ids: list[str]) -> dict:
    """Returns {video_id: FetchedTranscript | error_string} for each video."""
    results = {}
    for item in video_urls_or_ids:
        vid = extract_video_id(item)
        try:
            results[vid] = ytt_api.fetch(vid)
        except Exception as e:
            results[vid] = f"Error: {e}"
    return results


# ── 7. Error-safe wrapper ───────────────────────────────────────────────────
def safe_get_transcript(
    video_url_or_id: str, language: str = "en"
) -> str | None:
    """
    Returns the full transcript as plain text, or None on any error.
    Prints a descriptive message for each known error type.
    """
    video_id = extract_video_id(video_url_or_id)
    try:
        transcript = ytt_api.fetch(video_id, languages=[language, "en"])
        return " ".join(snippet.text for snippet in transcript)

    except TranscriptsDisabled:
        print(f"[!] Transcripts are disabled for video: {video_id}")
    except NoTranscriptFound:
        print(f"[!] No '{language}' transcript found for video: {video_id}")
    except VideoUnavailable:
        print(f"[!] Video unavailable: {video_id}")
    except IpBlocked:
        print(
            "[!] Your IP is blocked by YouTube.\n"
            "    → Upgrade and use proxy_config= (see get_transcript_with_proxy)"
        )
    except RequestBlocked:
        print(
            "[!] Request blocked by YouTube.\n"
            "    → On cloud servers use residential proxies.\n"
            "    → On local machine, wait a few minutes and retry."
        )
    except Exception as e:
        print(f"[!] Unexpected error: {e}")

    return None


# ── 8. Using proxies (for cloud / IP-blocked environments) ─────────────────
def get_transcript_with_proxy(video_url_or_id: str) -> FetchedTranscript:
    """
    Use when running on AWS / GCP / Azure or when IP is blocked.
    Requires: pip install --upgrade youtube-transcript-api  (>= 1.0.0)

    Option A: Any generic HTTP/SOCKS proxy
    Option B: Webshare rotating residential proxy (recommended by library author)
    """
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig

    video_id = extract_video_id(video_url_or_id)

    # -- Option A: Generic proxy --
    # proxied_api = YouTubeTranscriptApi(
    #     proxy_config=GenericProxyConfig(
    #         http_url="http://user:pass@proxy-host:port",
    #         https_url="https://user:pass@proxy-host:port",
    #     )
    # )

    # -- Option B: Webshare rotating residential proxies (recommended) --
    proxied_api = YouTubeTranscriptApi(
        proxy_config=WebshareProxyConfig(
            proxy_username="YOUR_WEBSHARE_USERNAME",
            proxy_password="YOUR_WEBSHARE_PASSWORD",
        )
    )

    return proxied_api.fetch(video_id)


# ── Timestamped transcript ──────────────────────────────────────────────────

def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_timestamped_transcript(video_url_or_id: str) -> list[dict]:
    """
    Returns list of:
      {'timestamp': '00:00:18', 'start': 18.6, 'duration': 4.5, 'text': '...'}
    """
    video_id = extract_video_id(video_url_or_id)
    transcript = ytt_api.fetch(video_id)

    result = []
    for snippet in transcript:
        result.append({
            "timestamp": seconds_to_hms(snippet.start),
            "start":     snippet.start,
            "duration":  snippet.duration,
            "text":      snippet.text,
        })
    return result


def print_timestamped_transcript(video_url_or_id: str):
    """Prints transcript with timestamps, like YouTube's transcript panel."""
    for entry in get_timestamped_transcript(video_url_or_id):
        print(f"[{entry['timestamp']}]  {entry['text']}")


def get_transcript_at_time(video_url_or_id: str, at_seconds: float) -> str | None:
    """Returns the caption text active at a specific second in the video."""
    for entry in get_timestamped_transcript(video_url_or_id):
        end = entry["start"] + entry["duration"]
        if entry["start"] <= at_seconds < end:
            return entry["text"]
    return None


def export_transcript_as_srt(video_url_or_id: str, output_file: str = "transcript.srt"):
    """Exports the transcript as a proper .srt subtitle file."""
    entries = get_timestamped_transcript(video_url_or_id)

    def to_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(output_file, "w", encoding="utf-8") as f:
        for i, entry in enumerate(entries, start=1):
            start = to_srt_time(entry["start"])
            end   = to_srt_time(entry["start"] + entry["duration"])
            f.write(f"{i}\n{start} --> {end}\n{entry['text']}\n\n")

    print(f"Saved: {output_file}")


# ── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    VIDEO = "https://www.youtube.com/watch?v=Rni7Fz7208c"

    # Print with timestamps
    print_timestamped_transcript(VIDEO)

    # Get caption at a specific moment
    caption = get_transcript_at_time(VIDEO, at_seconds=18.6)
    print(f"\nCaption at 0:18 → {caption}")

    # Export as .srt file
    export_transcript_as_srt(VIDEO, "rickroll.srt")