from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>SOP_GPT 한국어 챗봇</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f7f8;
      color: #1a1a1a;
      height: 100vh;
      display: flex;
      justify-content: center;
    }
    .app { width: 100%; max-width: 1600px; display: flex; flex-direction: column; height: 100vh; }

    /* ── 로그인 화면 ── */
    #login-view {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 20px;
      padding: 32px 24px;
      text-align: center;
    }
    #login-view h1 { font-size: 26px; font-weight: 700; }
    #login-view > p { color: #666; font-size: 14px; }

    .login-card {
      background: #fff;
      border: 1.5px solid #e0e0e0;
      border-radius: 16px;
      padding: 36px 40px;
      width: 360px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    }
    .login-card label {
      font-size: 12px;
      font-weight: 600;
      color: #555;
      text-align: left;
      margin-bottom: -6px;
    }
    .login-card input {
      padding: 11px 16px;
      border: 1.5px solid #d0d0d0;
      border-radius: 10px;
      font-size: 14px;
      outline: none;
      font-family: inherit;
      transition: border-color 0.15s;
    }
    .login-card input:focus { border-color: #2563eb; }
    .login-btn {
      margin-top: 4px;
      padding: 12px;
      border: none;
      border-radius: 10px;
      background: #2563eb;
      color: #fff;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      font-family: inherit;
      transition: background 0.15s;
    }
    .login-btn:hover { background: #1d4ed8; }
    .login-btn:disabled { background: #93b4f5; cursor: default; }
    .login-hint {
      font-size: 12px;
      color: #aaa;
      text-align: center;
    }
    .login-error {
      font-size: 12px;
      color: #ef4444;
      text-align: center;
      display: none;
    }

    /* ── 채팅 화면 ── */
    #chat-view { flex: 1; display: none; flex-direction: column; height: 100vh; overflow: hidden; }
    .chat-header {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      border-bottom: 1px solid #e5e5e5;
      background: #fff;
      flex-shrink: 0;
    }
    .back-btn {
      border: none;
      background: none;
      font-size: 13px;
      color: #555;
      cursor: pointer;
      padding: 6px 10px;
      border-radius: 6px;
    }
    .back-btn:hover { background: #f0f0f0; }
    #chat-title { font-size: 15px; font-weight: 600; flex: 1; }
    #chat-user-id { font-size: 11px; color: #aaa; font-family: monospace; }

    /* ── 분할 화면 ── */
    .split-container { flex: 1; display: flex; overflow: hidden; }
    .split-panel {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      min-width: 0;
    }
    .split-panel:not(:last-child) { border-right: 1px solid #e5e5e5; }
    .split-label {
      padding: 8px 16px;
      font-size: 12px;
      font-weight: 600;
      background: #f9f9f9;
      border-bottom: 1px solid #eee;
      flex-shrink: 0;
    }
    .split-label.sop { color: #2563eb; }
    .split-label.claude { color: #7c3aed; }
    .split-label.qwen { color: #059669; }
    .split-label.qwen-q { color: #d97706; }

    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .msg {
      max-width: 85%;
      padding: 9px 13px;
      border-radius: 16px;
      line-height: 1.55;
      font-size: 13px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .msg.user      { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
    .msg.assistant { align-self: flex-start; background: #fff; border: 1px solid #e5e5e5; border-bottom-left-radius: 4px; }
    .msg.assistant.claude-msg { border-color: #ddd5fe; }
    .msg.assistant.qwen-msg   { border-color: #a7f3d0; }
    .msg.assistant.qwen-q-msg { border-color: #fde68a; }
    .msg.loading   { color: #aaa; font-style: italic; }
    .msg.history   { opacity: 0.75; }
    .route-badge {
      align-self: flex-end;
      font-size: 10px;
      color: #2563eb;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 20px;
      padding: 2px 8px;
      margin-top: -6px;
      margin-bottom: 2px;
    }
    .rag-label {
      align-self: flex-start;
      font-size: 10px;
      color: #bbb;
      margin-top: -4px;
      padding: 0 4px;
      max-width: 85%;
    }

    .input-area {
      display: flex;
      gap: 8px;
      padding: 12px 16px;
      border-top: 1px solid #e5e5e5;
      background: #fff;
      flex-shrink: 0;
    }
    .input-area input {
      flex: 1;
      padding: 10px 14px;
      border: 1px solid #d0d0d0;
      border-radius: 20px;
      font-size: 14px;
      outline: none;
      font-family: inherit;
    }
    .input-area input:focus { border-color: #2563eb; }
    .input-area button {
      padding: 10px 20px;
      border: none;
      border-radius: 20px;
      background: #2563eb;
      color: #fff;
      font-size: 14px;
      cursor: pointer;
      font-family: inherit;
    }
    .input-area button:hover { background: #1d4ed8; }
    .input-area button:disabled { background: #93b4f5; cursor: default; }
    #send-btn.stopping { background: #ef4444; }
    #send-btn.stopping:hover { background: #dc2626; }
  </style>
</head>
<body>
  <div class="app">

    <!-- 로그인 화면 -->
    <div id="login-view">
      <div>
        <h1>SOP_GPT 한국어 챗봇</h1>
        <p style="color:#666; font-size:13px; margin-top:8px;">자동 라우팅 모드 — 질문에 맞는 모드를 자동으로 선택합니다.</p>
      </div>

      <div class="login-card">
        <label for="user-id-input">사용자 ID</label>
        <input id="user-id-input" placeholder="영문·숫자로 입력" autocomplete="off" autocorrect="off" spellcheck="false">
        <label for="password-input">비밀번호</label>
        <input id="password-input" type="password" placeholder="비밀번호 입력">
        <button class="login-btn" id="login-btn" onclick="enterChat()">입장하기</button>
        <span class="login-error" id="login-error">비밀번호가 일치하지 않습니다.</span>
        <span class="login-hint">처음 사용하면 자동으로 계정이 생성됩니다.</span>
      </div>
    </div>

    <!-- 채팅 화면 (분할) -->
    <div id="chat-view">
      <div class="chat-header">
        <button class="back-btn" onclick="goBack()">← 나가기</button>
        <span id="chat-title">SOP GPT</span>
        <span id="chat-user-id"></span>
      </div>
      <div class="split-container">
        <div class="split-panel">
          <div class="split-label sop">🤖 SOP_GPT</div>
          <div class="messages" id="messages-sop"></div>
        </div>
        <div class="split-panel">
          <div class="split-label claude">✨ Claude (Haiku)</div>
          <div class="messages" id="messages-claude"></div>
        </div>
        <div class="split-panel">
          <div class="split-label qwen">🟢 Qwen3-1.7B (BF16)</div>
          <div class="messages" id="messages-qwen"></div>
        </div>
        <div class="split-panel">
          <div class="split-label qwen-q">🟡 Qwen3-1.7B (Q4)</div>
          <div class="messages" id="messages-qwen-q"></div>
        </div>
      </div>
      <div class="input-area">
        <input id="msg-input" placeholder="질문을 입력하세요" autocomplete="off">
        <button id="send-btn" onclick="sendOrStop()">전송</button>
      </div>
    </div>

  </div>

  <script>
    let currentUserId = '';
    let pendingCount = 0;
    let abortController = null;

    // 단기 메모리: userId → {sop, claude, qwen, qwenQ}
    const sessionCache = new Map();

    async function enterChat() {
      const idInput  = document.getElementById('user-id-input');
      const pwInput  = document.getElementById('password-input');
      const errSpan  = document.getElementById('login-error');
      const btn      = document.getElementById('login-btn');

      const userId   = idInput.value.trim().toLowerCase();
      const password = pwInput.value;

      if (!userId || !password) {
        errSpan.textContent = 'ID와 비밀번호를 모두 입력해주세요.';
        errSpan.style.display = 'block';
        return;
      }

      btn.disabled = true;
      btn.textContent = '확인 중...';
      errSpan.style.display = 'none';

      try {
        const res  = await fetch('/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, password }),
        });
        const data = await res.json();

        if (!data.ok) {
          errSpan.textContent = '비밀번호가 일치하지 않습니다.';
          errSpan.style.display = 'block';
          btn.disabled = false;
          btn.textContent = '입장하기';
          return;
        }

        currentUserId = userId;
        document.getElementById('chat-user-id').textContent = 'ID: ' + userId;
        document.getElementById('login-view').style.display = 'none';
        document.getElementById('chat-view').style.display = 'flex';
        document.getElementById('chat-view').style.flexDirection = 'column';

        const cached = sessionCache.get(userId);
        if (cached) {
          document.getElementById('messages-sop').innerHTML    = cached.sop;
          document.getElementById('messages-claude').innerHTML = cached.claude;
          document.getElementById('messages-qwen').innerHTML   = cached.qwen;
          document.getElementById('messages-qwen-q').innerHTML = cached.qwenQ;
        } else {
          document.getElementById('messages-sop').innerHTML    = '';
          document.getElementById('messages-claude').innerHTML = '';
          document.getElementById('messages-qwen').innerHTML   = '';
          document.getElementById('messages-qwen-q').innerHTML = '';
          await loadHistory(userId);
        }

        document.getElementById('msg-input').focus();
      } catch (_) {
        errSpan.textContent = '서버 연결에 실패했습니다.';
        errSpan.style.display = 'block';
      } finally {
        btn.disabled = false;
        btn.textContent = '입장하기';
      }
    }

    function goBack() {
      if (currentUserId) {
        sessionCache.set(currentUserId, {
          sop:    document.getElementById('messages-sop').innerHTML,
          claude: document.getElementById('messages-claude').innerHTML,
          qwen:   document.getElementById('messages-qwen').innerHTML,
          qwenQ:  document.getElementById('messages-qwen-q').innerHTML,
        });
      }
      document.getElementById('login-view').style.display = 'flex';
      document.getElementById('chat-view').style.display  = 'none';
    }

    async function loadHistory(userId) {
      try {
        const [sopData, claudeData, qwenData, qwenQData] = await Promise.all([
          fetch(`/chat/auto/history?thread_id=${encodeURIComponent(userId)}`).then(r => r.json()),
          fetch(`/chat/claude/auto/history?thread_id=${encodeURIComponent(userId)}`).then(r => r.json()),
          fetch(`/chat/auto/history?thread_id=${encodeURIComponent(userId + ':bf16')}`).then(r => r.json()),
          fetch(`/chat/auto/history?thread_id=${encodeURIComponent(userId + ':q4')}`).then(r => r.json()),
        ]);
        for (const msg of (sopData.messages || [])) {
          addMsg('messages-sop', msg.role === 'user' ? 'user' : 'assistant history', msg.content);
        }
        for (const msg of (claudeData.messages || [])) {
          addMsg('messages-claude', msg.role === 'user' ? 'user' : 'assistant claude-msg history', msg.content);
        }
        for (const msg of (qwenData.messages || [])) {
          addMsg('messages-qwen', msg.role === 'user' ? 'user' : 'assistant qwen-msg history', msg.content);
        }
        for (const msg of (qwenQData.messages || [])) {
          addMsg('messages-qwen-q', msg.role === 'user' ? 'user' : 'assistant qwen-q-msg history', msg.content);
        }
      } catch (_) {}
    }

    function addMsg(containerId, cls, text) {
      const box = document.getElementById(containerId);
      const div = document.createElement('div');
      div.className = 'msg ' + cls;
      div.textContent = text;
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
      return div;
    }

    function addRouteBadge(containerId, label) {
      const box = document.getElementById(containerId);
      const div = document.createElement('div');
      div.className = 'route-badge';
      div.textContent = '→ ' + label;
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
    }

    function addLabel(containerId, text) {
      const box = document.getElementById(containerId);
      const div = document.createElement('div');
      div.className = 'rag-label';
      div.textContent = text;
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
    }

    function sendOrStop() {
      if (pendingCount > 0) {
        if (abortController) abortController.abort();
      } else {
        send();
      }
    }

    async function send() {
      const input    = document.getElementById('msg-input');
      const btn      = document.getElementById('send-btn');
      const question = input.value.trim();
      if (!question || !currentUserId) return;

      addMsg('messages-sop',    'user', question);
      addMsg('messages-claude', 'user', question);
      addMsg('messages-qwen',   'user', question);
      addMsg('messages-qwen-q', 'user', question);
      input.value = '';
      btn.textContent = '정지';
      btn.classList.add('stopping');
      pendingCount = 4;

      abortController = new AbortController();
      const signal = abortController.signal;

      streamPanelWith('/chat/auto/stream',               question, 'messages-sop',    'assistant',            signal);
      streamPanelWith('/chat/claude/auto/stream',        question, 'messages-claude', 'assistant claude-msg', signal);
      streamPanelWith('/chat/qwen/langgraph/stream',     question, 'messages-qwen',   'assistant qwen-msg',   signal);
      streamPanelWith('/chat/qwen-q/langgraph/stream',   question, 'messages-qwen-q', 'assistant qwen-q-msg', signal);
    }

    async function streamPanelWith(endpoint, question, containerId, msgClass, signal) {
      const box = document.getElementById(containerId);
      const statusDiv = addMsg(containerId, 'assistant loading', '분류 중…');
      let answerDiv = null;
      let routeLabel = null;

      try {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, thread_id: currentUserId || null }),
          signal,
        });

        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const parts = buf.split('\\n\\n');
          buf = parts.pop();
          for (const part of parts) {
            if (!part.startsWith('data: ')) continue;
            const evt = JSON.parse(part.slice(6));

            if (evt.type === 'mode') {
              routeLabel = evt.label;
              statusDiv.textContent = '[' + evt.label + '] 생성 중…';
              box.scrollTop = box.scrollHeight;
            } else if (evt.type === 'status') {
              statusDiv.textContent = evt.text;
              box.scrollTop = box.scrollHeight;
            } else if (evt.type === 'text') {
              statusDiv.textContent = evt.text || '(빈 응답)';
              statusDiv.className = 'msg ' + msgClass;
              box.scrollTop = box.scrollHeight;
            } else if (evt.type === 'text_fallback') {
              if (answerDiv === null) {
                answerDiv = addMsg(containerId, msgClass, '');
              }
              answerDiv.textContent = evt.text || '(빈 응답)';
              box.scrollTop = box.scrollHeight;
            } else if (evt.type === 'rag_context') {
              addLabel(containerId, '참고: ' + evt.text);
            }
          }
        }
      } catch (e) {
        if (e.name === 'AbortError') {
          statusDiv.textContent = '중단됨';
        } else {
          statusDiv.textContent = '오류가 발생했습니다.';
          statusDiv.className = 'msg ' + msgClass;
        }
      } finally {
        if (routeLabel) addRouteBadge(containerId, routeLabel);
        pendingCount--;
        if (pendingCount === 0) {
          const btn = document.getElementById('send-btn');
          btn.textContent = '전송';
          btn.classList.remove('stopping');
          document.getElementById('msg-input').focus();
        }
      }
    }

    document.getElementById('msg-input').addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.isComposing && e.keyCode !== 229) send();
    });
    document.getElementById('password-input').addEventListener('keydown', function(e) {
      if (e.key === 'Enter') enterChat();
    });
    document.getElementById('user-id-input').addEventListener('keydown', function(e) {
      if (e.key === 'Enter') document.getElementById('password-input').focus();
    });
  </script>
</body>
</html>"""


@router.get("/", response_class=HTMLResponse)
def index():
    return _HTML
