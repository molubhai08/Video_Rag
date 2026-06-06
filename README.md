# PodcastBot — Video RAG Q&A App

A premium, YouTube-style web application that performs Retrieval-Augmented Generation (RAG) on YouTube video transcripts. Paste a YouTube URL, index the transcript, and ask questions. The app answers using Groq's Llama 3.1 model and automatically seeks the YouTube video player to the exact relevant timestamp.

---

### **Submission Deliverables**
- **Solution Writeup (PDF)**: [`solution_writeup.pdf` in root directory](solution_writeup.pdf)
- **Demo & Explainer Video**: [Google Drive Video Link](https://drive.google.com/file/d/1xJhp6rtrXSHlRKektrkxna3-oFBTugUL/view?usp=sharing)
- **GitHub Repository**: [https://github.com/molubhai08/Video_Rag](https://github.com/molubhai08/Video_Rag)

---

## Features
- **YouTube Transcript Processing**: Automatically downloads and processes English transcripts.
- **Two-Level Retrieval**: Semantic chunking of 450 words + 100-word overlap (Level 1) combined with token-level keyword overlap alignment (Level 2) to pinpoint exact timestamps.
- **Interactive YouTube Player**: Synchronized with Q&A timestamps (click a timestamp to jump in the video).
- **Premium YouTube-style UI**: Glassmorphic dark mode layout featuring a main video player area and live-chat style Q&A sidebar.
- **Local JSON Caching**: Caches transcript and chunk indices to avoid redundant external network requests and bypass rate limits.

---

## Local Setup

### 1. Prerequisites
- Python 3.10+
- A Groq API Key (Get a free key at [Groq Console](https://console.groq.com/))

### 2. Installation
Clone/download the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
FLASK_SECRET_KEY=your_custom_secret_key_here
```

### 4. Running the App

#### Development Mode (Flask Dev Server)
```bash
python app.py
```
Go to `http://127.0.0.1:5000` in your web browser.

---

## How It Works

1. **Extraction**: The user provides a YouTube URL. The system extracts the 11-character `video_id`.
2. **Transcript Processing & Chunking**:
   - The app attempts to load the cached transcript. If not cached, it fetches the transcript using the custom YouTube client wrapper `youtube-transcript-api`.
   - **Level 1 Chunking**: Groups transcript snippets into semantic chunks of target ~450 words with a 100-word sliding overlap to preserve contextual boundaries.
3. **Retrieval**:
   - We fit a `TfidfVectorizer` (unigrams + bigrams) on the fly over the extracted chunks.
   - We vectorize the user query and calculate cosine similarity scores. The top 3 chunks are selected.
4. **Level 2 Refinement (Timestamp Pinpointing)**:
   - Within the top-scoring chunk, the app calculates token-level word overlaps between the query and each individual subtitle snippet.
   - The snippet with the highest overlap is selected as the exact seeking timestamp.
5. **Answer Generation**:
   - The top 3 chunks are formatted as context text alongside their timestamps.
   - The context and user query are passed to the Groq API.
6. **Seeking Interface**:
   - Clicking a timestamp seeks the embedded YouTube player to the exact moment in the video.
