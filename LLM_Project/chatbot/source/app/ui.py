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
    .app { width: 100%; max-width: 1100px; display: flex; flex-direction: column; height: 100vh; }

    /* ── 모드 선택 화면 ── */
    #select-view {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 32px;
      padding: 32px 24px;
      text-align: center;
    }
    #select-view h1 { font-size: 26px; font-weight: 700; }
    #select-view p { color: #666; font-size: 14px; }
    .mode-cards { display: flex; gap: 16px; flex-wrap: wrap; justify-content: center; }
    .mode-card {
      width: 210px;
      padding: 28px 20px;
      border: 1.5px solid #e0e0e0;
      border-radius: 14px;
      background: #fff;
      cursor: pointer;
      transition: box-shadow 0.15s, transform 0.15s, border-color 0.15s;
      text-align: left;
    }
    .mode-card:hover {
      box-shadow: 0 6px 18px rgba(0,0,0,0.09);
      transform: translateY(-3px);
      border-color: #2563eb;
    }
    .mode-card .icon { font-size: 28px; margin-bottom: 12px; }
    .mode-card h3 { font-size: 15px; font-weight: 600; margin-bottom: 6px; }
    .mode-card p { font-size: 12px; color: #888; line-height: 1.5; }

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
    #chat-title { font-size: 15px; font-weight: 600; }

    /* ── 분할 화면 ── */
    .split-container { flex: 1; display: flex; overflow: hidden; }
    .split-panel {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      min-width: 0;
    }
    .split-panel:first-child { border-right: 1px solid #e5e5e5; }
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
    .msg.loading   { color: #aaa; font-style: italic; }
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
  </style>
</head>
<body>
  <div class="app">

    <!-- 모드 선택 화면 -->
    <div id="select-view">
      <div>
        <h1>SOP_GPT 한국어 챗봇</h1>
        <p style="margin-top:8px">검색 방식을 선택하세요 (왼쪽 SOP_GPT / 오른쪽 Claude)</p>
      </div>
      <div class="mode-cards">
        <div class="mode-card" onclick="startChat('basic', '기본 모델')">
          <div class="icon">💬</div>
          <h3>기본 모델</h3>
          <p>검색 없이 직접 답변합니다.</p>
        </div>
        <div class="mode-card" onclick="startChat('rag', 'RAG 기반 검색')">
          <div class="icon">🔍</div>
          <h3>RAG 기반 검색</h3>
          <p>TF-IDF로 관련 문서를 검색한 뒤 답변합니다.</p>
        </div>
        <div class="mode-card" onclick="startChat('langchain', 'LangChain 기반 검색')">
          <div class="icon">⚡</div>
          <h3>LangChain 기반 검색</h3>
          <p>BM25 + FAISS 하이브리드 검색 후 답변합니다.</p>
        </div>
      </div>
    </div>

    <!-- 채팅 화면 (분할) -->
    <div id="chat-view">
      <div class="chat-header">
        <button class="back-btn" onclick="goBack()">← 뒤로</button>
        <span id="chat-title">채팅</span>
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
      </div>
      <div class="input-area">
        <input id="msg-input" placeholder="질문을 입력하세요" autocomplete="off">
        <button id="send-btn" onclick="send()">전송</button>
      </div>
    </div>

  </div>

  <script>
    let currentMode = '';
    let pendingCount = 0;

    const CLAUDE_MAP = { basic: 'claude/basic', rag: 'claude/rag', langchain: 'claude/langchain' };

    function startChat(mode, title) {
      currentMode = mode;
      document.getElementById('chat-title').textContent = title;
      document.getElementById('messages-sop').innerHTML = '';
      document.getElementById('messages-claude').innerHTML = '';
      document.getElementById('select-view').style.display = 'none';
      document.getElementById('chat-view').style.display = 'flex';
      document.getElementById('chat-view').style.flexDirection = 'column';
      document.getElementById('msg-input').focus();
    }

    function goBack() {
      document.getElementById('select-view').style.display = 'flex';
      document.getElementById('chat-view').style.display = 'none';
      currentMode = '';
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

    function addLabel(containerId, text) {
      const box = document.getElementById(containerId);
      const div = document.createElement('div');
      div.className = 'rag-label';
      div.textContent = text;
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
    }

    async function send() {
      const input = document.getElementById('msg-input');
      const btn   = document.getElementById('send-btn');
      const question = input.value.trim();
      if (!question || !currentMode || pendingCount > 0) return;

      addMsg('messages-sop',    'user', question);
      addMsg('messages-claude', 'user', question);
      input.value = '';
      btn.disabled = true;
      pendingCount = 2;

      const sopUrl    = '/chat/' + currentMode + '/stream';
      const claudeUrl = '/chat/' + CLAUDE_MAP[currentMode] + '/stream';

      streamPanelWith(sopUrl,    question, 'messages-sop',    'assistant');
      streamPanelWith(claudeUrl, question, 'messages-claude', 'assistant claude-msg');
    }

    async function streamPanelWith(endpoint, question, containerId, msgClass) {
      const box = document.getElementById(containerId);
      const div = addMsg(containerId, 'assistant loading', '생성 중…');

      try {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question }),
        });

        const reader = res.body.getReader();
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
            if (evt.type === 'text') {
              div.textContent = evt.text || '(빈 응답)';
              div.className = 'msg ' + msgClass;
              box.scrollTop = box.scrollHeight;
            } else if (evt.type === 'rag_context') {
              addLabel(containerId, '참고: ' + evt.text);
            }
          }
        }
      } catch (e) {
        div.textContent = '오류가 발생했습니다.';
        div.className = 'msg ' + msgClass;
      } finally {
        pendingCount--;
        if (pendingCount === 0) {
          document.getElementById('send-btn').disabled = false;
          document.getElementById('msg-input').focus();
        }
      }
    }

    document.getElementById('msg-input').addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.isComposing && e.keyCode !== 229) send();
    });
  </script>
</body>
</html>"""


@router.get("/", response_class=HTMLResponse)
def index():
    return _HTML
