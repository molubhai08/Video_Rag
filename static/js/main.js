/**
 * PodcastBot — main.js
 * Handles: YouTube IFrame API, chat UI, timestamp seeking, embed fallback
 */

// ── State ─────────────────────────────────────────────────────────────────
const appState = {
  player: null,
  playerReady: false,
  embedFailed: false,
  videoId: null,
  isLoaded: false,
  isAsking: false,
};

// ── DOM refs ──────────────────────────────────────────────────────────────
const videoUrlInput    = document.getElementById('video-url-input');
const loadBtn          = document.getElementById('load-btn');
const loadStatus       = document.getElementById('load-status');
const playerPlaceholder= document.getElementById('player-placeholder');
const ytPlayerContainer= document.getElementById('yt-player-container');
const videoInfoDiv     = document.getElementById('video-info');
const infoChunks       = document.getElementById('info-chunks');
const infoSnippets     = document.getElementById('info-snippets');
const infoDuration     = document.getElementById('info-duration');
const infoStatus       = document.getElementById('info-status');
const videoTitleDisplay= document.getElementById('video-title-display');
const chatMessages     = document.getElementById('chat-messages');
const chatStatusBadge  = document.getElementById('chat-status-badge');
const questionInput    = document.getElementById('question-input');
const sendBtn          = document.getElementById('send-btn');
const jumpToast        = document.getElementById('jump-toast');
const jumpToastText    = document.getElementById('jump-toast-text');
const exampleChips     = document.querySelectorAll('.example-chip');

// ── YouTube IFrame API ────────────────────────────────────────────────────
// Called by YouTube script when API is ready
window.onYouTubeIframeAPIReady = function () {
  // Player will be created when user loads a video
};

function createYouTubePlayer(videoId) {
  appState.videoId = videoId;
  appState.player = null;
  appState.playerReady = false;
  appState.embedFailed = false;

  // Show container, hide placeholder immediately
  playerPlaceholder.hidden = true;
  ytPlayerContainer.hidden = false;

  // Use a plain <iframe> embed — always works, always shows video+controls.
  // enablejsapi=1 lets us send postMessage commands for seeking.
  const embedUrl = `https://www.youtube.com/embed/${videoId}` +
    `?controls=1&rel=0&modestbranding=1&fs=1&playsinline=1&enablejsapi=1` +
    `&origin=${encodeURIComponent(window.location.origin)}`;

  const iframe = document.createElement('iframe');
  iframe.id = 'yt-iframe';
  iframe.src = embedUrl;
  iframe.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen';
  iframe.allowFullscreen = true;
  iframe.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;border:none;display:block;';

  ytPlayerContainer.innerHTML = '';
  ytPlayerContainer.appendChild(iframe);

  // Mark player as ready — postMessage doesn't need an onReady callback
  iframe.addEventListener('load', () => {
    appState.playerReady = true;
    appState.embedFailed = false;
  });

  // Store iframe reference as the "player"
  appState.player = iframe;
}

function setEmbedFallback(videoId) {
  // Show a message that embedding is disabled, timestamps open in new tab
  playerPlaceholder.hidden = false;
  ytPlayerContainer.hidden = true;
  playerPlaceholder.innerHTML = `
    <div style="text-align:center; padding: 16px; color: #9898b8;">
      <div style="font-size:32px; margin-bottom:10px;">🔒</div>
      <p style="font-size:13px; margin-bottom:8px;">Embedding disabled for this video</p>
      <p style="font-size:12px; color:#5a5a7a;">Timestamp links will open YouTube in a new tab</p>
      <a href="https://youtube.com/watch?v=${videoId}" target="_blank" rel="noopener"
         style="display:inline-block; margin-top:12px; padding:6px 16px; border:1px solid rgba(120,90,220,0.4);
                border-radius:999px; color:#a78bfa; font-size:12px; text-decoration:none;">
        ▶ Watch on YouTube
      </a>
    </div>
  `;
}

function seekToTimestamp(seconds) {
  const iframe = document.getElementById('yt-iframe');
  if (iframe && iframe.contentWindow) {
    // seekTo via postMessage (works with enablejsapi=1)
    iframe.contentWindow.postMessage(
      JSON.stringify({ event: 'command', func: 'seekTo', args: [seconds, true] }),
      '*'
    );
    iframe.contentWindow.postMessage(
      JSON.stringify({ event: 'command', func: 'playVideo', args: [] }),
      '*'
    );
    showJumpToast(seconds);
  } else if (appState.videoId) {
    // Fallback: open YouTube at timestamp
    const url = `https://www.youtube.com/watch?v=${appState.videoId}&t=${Math.floor(seconds)}s`;
    window.open(url, '_blank', 'noopener');
    showJumpToast(seconds);
  }
}

function showJumpToast(seconds) {
  const hms = secondsToHms(seconds);
  jumpToastText.textContent = `Jumping to ${hms}`;
  jumpToast.hidden = false;
  jumpToast.classList.add('visible');
  setTimeout(() => {
    jumpToast.classList.remove('visible');
    setTimeout(() => { jumpToast.hidden = true; }, 300);
  }, 2000);
}

// ── Load Video ────────────────────────────────────────────────────────────
loadBtn.addEventListener('click', loadVideo);
videoUrlInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') loadVideo();
});

async function loadVideo() {
  const url = videoUrlInput.value.trim();
  if (!url) {
    setLoadStatus('Please enter a YouTube URL', 'error');
    return;
  }

  setLoadStatus('Starting...', 'loading');
  setLoadBtnState(true);
  setChatStatusBadge('loading', 'Loading...');

  try {
    // Fire init — returns immediately (202) while backend works in background thread
    const res = await fetch('/api/init', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    const initData = await res.json();

    if (initData.error) throw new Error(initData.error);

    // Poll /api/status until ready or error
    await pollUntilReady();

  } catch (err) {
    setLoadStatus(`✗ ${err.message}`, 'error');
    setChatStatusBadge('error', 'Error');
    appendSystemMessage(`❌ Failed to load: ${err.message}`);
    setLoadBtnState(false);
  }
}

async function pollUntilReady() {
  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        const res  = await fetch('/api/status');
        const data = await res.json();

        // Update live stage text
        if (data.stage) setLoadStatus(`⏳ ${data.stage}`, 'loading');

        if (data.status === 'ready') {
          clearInterval(interval);

          // Success
          appState.isLoaded = true;
          appState.videoId  = data.video_id;

          infoChunks.textContent   = data.chunks;
          infoSnippets.textContent = data.snippet_count;
          infoDuration.textContent = data.duration_hms;
          infoStatus.textContent   = 'Ready';
          infoStatus.className     = 'info-value status-ready';
          videoInfoDiv.hidden      = false;

          // Update video title display with the URL
          if (videoTitleDisplay) {
            const urlVal = videoUrlInput.value.trim();
            videoTitleDisplay.textContent = `youtube.com/watch?v=${data.video_id}`;
          }

          setLoadStatus(`✓ Loaded! ${data.chunks} chunks, ${data.snippet_count} snippets`, 'success');
          setChatStatusBadge('ready', 'Ready');
          sendBtn.disabled = false;
          setLoadBtnState(false);

          createYouTubePlayer(data.video_id);
          appendSystemMessage(
            `✅ Video loaded! <strong>${data.chunks} semantic chunks</strong> indexed from <strong>${data.snippet_count} transcript snippets</strong> (${data.duration_hms} total). Ask me anything!`
          );
          resolve();

        } else if (data.status === 'error') {
          clearInterval(interval);
          const err = data.error || 'Unknown error';
          setLoadStatus(`✗ ${err}`, 'error');
          setChatStatusBadge('error', 'Error');
          appendSystemMessage(`❌ Failed to load: ${err}`);
          setLoadBtnState(false);
          reject(new Error(err));
        }
        // If still 'loading', keep polling

      } catch (e) {
        clearInterval(interval);
        setLoadStatus(`✗ Polling error: ${e.message}`, 'error');
        setLoadBtnState(false);
        reject(e);
      }
    }, 800); // poll every 800ms
  });
}

function setLoadBtnState(loading) {
  const btnText    = loadBtn.querySelector('.btn-text');
  const btnSpinner = loadBtn.querySelector('.btn-spinner');
  loadBtn.disabled = loading;
  btnText.hidden   = loading;
  btnSpinner.hidden= !loading;
}

function setLoadStatus(msg, type) {
  loadStatus.textContent  = msg;
  loadStatus.className    = `load-status ${type}`;
}

function setChatStatusBadge(type, label) {
  chatStatusBadge.textContent = label;
  chatStatusBadge.className   = `chat-status-badge status-${type}`;
}

// ── Send Question ─────────────────────────────────────────────────────────
sendBtn.addEventListener('click', sendQuestion);
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});

// Auto-resize textarea
questionInput.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
});

// Example chips
exampleChips.forEach(chip => {
  chip.addEventListener('click', () => {
    if (!appState.isLoaded) {
      setLoadStatus('Load a video first!', 'error');
      return;
    }
    questionInput.value = chip.dataset.query;
    questionInput.dispatchEvent(new Event('input'));
    sendQuestion();
  });
});

async function sendQuestion() {
  if (appState.isAsking) return;
  const question = questionInput.value.trim();
  if (!question) return;
  if (!appState.isLoaded) {
    appendSystemMessage('⚠️ Please load a video first before asking questions.');
    return;
  }

  appState.isAsking = true;
  sendBtn.disabled  = true;
  questionInput.value = '';
  questionInput.style.height = 'auto';

  // Append user message
  appendUserMessage(question);

  // Append thinking indicator
  const thinkingId = appendThinking();

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();

    removeThinking(thinkingId);

    if (!res.ok || data.error) {
      appendBotError(data.error || 'Something went wrong');
      return;
    }

    appendBotAnswer(data);

  } catch (err) {
    removeThinking(thinkingId);
    appendBotError(`Network error: ${err.message}`);
  } finally {
    appState.isAsking = false;
    sendBtn.disabled  = false;
    questionInput.focus();
  }
}

// ── Message rendering ─────────────────────────────────────────────────────
function appendUserMessage(text) {
  const now = new Date();
  const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const div = document.createElement('div');
  div.className = 'message user-message';
  div.innerHTML = `
    <div class="message-avatar user-avatar">👤</div>
    <div class="message-content">
      <div class="message-bubble">${escapeHtml(text)}</div>
      <div class="message-time">${timeStr}</div>
    </div>
  `;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function appendThinking() {
  const id = `thinking-${Date.now()}`;
  const div = document.createElement('div');
  div.className = 'message bot-message';
  div.id = id;
  div.innerHTML = `
    <div class="message-avatar bot-avatar">🤖</div>
    <div class="message-content">
      <div class="thinking-bubble">
        <div class="thinking-dots">
          <span></span><span></span><span></span>
        </div>
        Searching transcript &amp; crafting answer…
      </div>
    </div>
  `;
  chatMessages.appendChild(div);
  scrollToBottom();
  return id;
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendBotAnswer(data) {
  const {
    answer, timestamp_s, timestamp_hms,
    chunk_start_hms, chunk_end_hms,
    relevance_score, video_id, top_chunks
  } = data;

  // Build related segments pills (top 2-3)
  const relatedHtml = top_chunks.length > 1
    ? `<div class="related-segments">
        <span class="related-label">Other relevant segments</span>
        <div class="related-chips">
          ${top_chunks.slice(1).map(c => `
            <button class="related-chip"
              data-seconds="${c.start}"
              title="${escapeHtml(c.preview)}">
              ⏱ ${c.start_hms}–${c.end_hms}
              <span style="opacity:0.5; font-size:10px">${Math.round(c.score * 100)}%</span>
            </button>
          `).join('')}
        </div>
      </div>`
    : '';

  // Determine if we can seek or need to open in new tab
  const jumpLabel = (appState.playerReady && !appState.embedFailed)
    ? '▶ Jump in player'
    : '↗ Open on YouTube';

  const ytUrl = `https://www.youtube.com/watch?v=${video_id}&t=${Math.floor(timestamp_s)}s`;

  const div = document.createElement('div');
  div.className = 'message bot-message';
  div.innerHTML = `
    <div class="message-avatar bot-avatar">🤖</div>
    <div class="message-content" style="max-width:85%;">
      <div class="message-bubble">
        <p>${escapeHtml(answer)}</p>
      </div>
      <div class="timestamp-card">
        <div class="timestamp-main">
          <span class="ts-badge">⏱ ${timestamp_hms}</span>
          <button class="jump-btn" data-seconds="${timestamp_s}">
            ${jumpLabel}
          </button>
          <a class="yt-link-btn" href="${ytUrl}" target="_blank" rel="noopener">
            🔗 YouTube
          </a>
        </div>
        <div class="segment-range">
          Retrieved from segment ${chunk_start_hms} – ${chunk_end_hms}
          &nbsp;·&nbsp;
          <span class="relevance-pill">${Math.round(relevance_score * 100)}% match</span>
        </div>
        ${relatedHtml}
      </div>
    </div>
  `;

  // Jump button click
  div.querySelectorAll('.jump-btn').forEach(btn => {
    btn.addEventListener('click', () => seekToTimestamp(parseFloat(btn.dataset.seconds)));
  });

  // Related chip clicks
  div.querySelectorAll('.related-chip').forEach(chip => {
    chip.addEventListener('click', () => seekToTimestamp(parseFloat(chip.dataset.seconds)));
  });

  chatMessages.appendChild(div);
  scrollToBottom();
}

function appendBotError(msg) {
  const div = document.createElement('div');
  div.className = 'message bot-message';
  div.innerHTML = `
    <div class="message-avatar bot-avatar">🤖</div>
    <div class="message-content">
      <div class="message-bubble" style="border-color: rgba(239,68,68,0.3);">
        <p style="color:#fca5a5;">⚠️ ${escapeHtml(msg)}</p>
      </div>
    </div>
  `;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function appendSystemMessage(html) {
  const div = document.createElement('div');
  div.className = 'message bot-message';
  div.innerHTML = `
    <div class="message-avatar bot-avatar" style="background:linear-gradient(135deg,#0f766e,#065f46);">ℹ️</div>
    <div class="message-content">
      <div class="message-bubble" style="border-color:rgba(16,185,129,0.2); background:rgba(16,185,129,0.04);">
        <p style="color:#6ee7b7;">${html}</p>
      </div>
    </div>
  `;
  chatMessages.appendChild(div);
  scrollToBottom();
}

// ── Utilities ─────────────────────────────────────────────────────────────
function scrollToBottom() {
  setTimeout(() => {
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
  }, 50);
}

function secondsToHms(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return [
    String(h).padStart(2, '0'),
    String(m).padStart(2, '0'),
    String(s).padStart(2, '0'),
  ].join(':');
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}
