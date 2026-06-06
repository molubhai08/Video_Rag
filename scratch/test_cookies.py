import requests
from http.cookiejar import MozillaCookieJar
from youtube_transcript_api import YouTubeTranscriptApi

try:
    session = requests.Session()
    cj = MozillaCookieJar("www.youtube.com_cookies.txt")
    cj.load(ignore_discard=True, ignore_expires=True)
    session.cookies = cj
    print("Cookies loaded successfully!")
    
    # Try fetching a known video transcript
    # Using a popular video: e.g. "dQw4w9WgXcQ"
    api = YouTubeTranscriptApi(http_client=session)
    transcript = api.fetch("dQw4w9WgXcQ", languages=["en"])
    print(f"Successfully fetched transcript: {len(transcript)} snippets!")
except Exception as e:
    print(f"Error: {e}")
