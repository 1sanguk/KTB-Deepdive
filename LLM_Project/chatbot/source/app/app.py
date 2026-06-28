import sys
from pathlib import Path
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / "api_keys")

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import device, SOP_GPT, SOP_GPT_Span
from bpe import build_vocab, base_alphabet, load_bpe
from lc.llm import SOP_GPT_LLM, make_span_extractor
from lc.chain import build_basic_chain, build_rag_chain
from lc.retriever import build_hybrid_retriever
from rag.rag import build_tfidf_retriever

BPE_PATH = MODEL_DIR / "bpe_vocab.json"
GEN_CKPT = MODEL_DIR / "SOP_GPT.pt"
QA_CKPT = MODEL_DIR / "SOP_GPT_qa.pt"
SPAN_CKPT = MODEL_DIR / "SOP_GPT_span.pt"

RAG_SIM_THRESHOLD = 0.515   # LangChain 하이브리드 held-out 검증 최적값 (정확도 82.7%)
TFIDF_SIM_THRESHOLD = 0.25  # TF-IDF 단독 임계값 (Phase 7 기준)

# ── 모델 & 토크나이저 로드 ──────────────────────────────────────────────────────
vocab, merges = load_bpe(BPE_PATH)
stoi, itos = build_vocab(vocab)
vocab_size = len(vocab)
base_set = base_alphabet(vocab)

gen_model = SOP_GPT(vocab_size).to(device)
gen_model.load_state_dict(torch.load(GEN_CKPT, map_location=device))
gen_model.eval()

qa_model = SOP_GPT(vocab_size).to(device)
qa_model.load_state_dict(torch.load(QA_CKPT, map_location=device))
qa_model.eval()

span_model = SOP_GPT_Span(vocab_size).to(device)
span_model.load_state_dict(torch.load(SPAN_CKPT, map_location=device))
span_model.eval()

# ── 검색기 ─────────────────────────────────────────────────────────────────────
tfidf_retriever = build_tfidf_retriever()
lc_retriever = build_hybrid_retriever()

# ── LangChain 컴포넌트 ─────────────────────────────────────────────────────────
gen_llm = SOP_GPT_LLM(
    torch_model=gen_model, stoi=stoi, itos=itos, merges=merges, base_set=base_set,
    stop_on="sentence", temperature=0.7, top_k=None, top_p=0.9,
    repetition_penalty=1.3, max_new_tokens=200,
)
qa_llm = SOP_GPT_LLM(
    torch_model=qa_model, stoi=stoi, itos=itos, merges=merges, base_set=base_set,
    stop_on="line", temperature=0.8, top_k=40,
    repetition_penalty=1.3, max_new_tokens=60,
)
span_extractor_fn = make_span_extractor(span_model, stoi, merges, base_set)

# ── LCEL 체인 ──────────────────────────────────────────────────────────────────
basic_chain = build_basic_chain(qa_llm)
tfidf_rag_chain = build_rag_chain(tfidf_retriever, qa_llm, span_extractor_fn, TFIDF_SIM_THRESHOLD)
lc_rag_chain = build_rag_chain(lc_retriever, qa_llm, span_extractor_fn, RAG_SIM_THRESHOLD)

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="SOP_GPT 한국어 챗봇")


class GenerateRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 200


class GenerateResponse(BaseModel):
    text: str


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    retrieved_context: str
    used_rag: bool


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Stage 1: gen_llm 으로 입력 문장을 이어써서 완전한 문장으로 끝낸다."""
    text = gen_llm.invoke(req.prompt)
    return GenerateResponse(text=text)


@app.post("/chat/basic", response_model=ChatResponse)
def chat_basic(req: ChatRequest):
    """basic_chain: 검색 없이 QA LLM 으로 직접 답변 (Stage 2)."""
    answer = basic_chain.invoke(req.question)
    return ChatResponse(answer=answer, retrieved_context="", used_rag=False)


@app.post("/chat/rag", response_model=ChatResponse)
def chat_rag(req: ChatRequest):
    """tfidf_rag_chain: TF-IDF 검색 + 유사도 라우팅 → Span 추출 or QA 폴백."""
    result = tfidf_rag_chain.invoke(req.question)
    return ChatResponse(**result)


@app.post("/chat/langchain", response_model=ChatResponse)
def chat_langchain(req: ChatRequest):
    """lc_rag_chain: LangChain BM25+FAISS 하이브리드 검색 + 유사도 라우팅."""
    result = lc_rag_chain.invoke(req.question)
    return ChatResponse(**result)


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
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
    .app { width: 100%; max-width: 760px; display: flex; flex-direction: column; height: 100vh; }

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
    #chat-view { flex: 1; display: none; flex-direction: column; height: 100vh; }
    .chat-header {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      border-bottom: 1px solid #e5e5e5;
      background: #fff;
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
    .mode-badge {
      margin-left: auto;
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 20px;
      background: #eff6ff;
      color: #2563eb;
      font-weight: 500;
    }

    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 20px 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .msg {
      max-width: 78%;
      padding: 10px 14px;
      border-radius: 16px;
      line-height: 1.55;
      font-size: 14px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .msg.user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
    .msg.assistant { align-self: flex-start; background: #fff; border: 1px solid #e5e5e5; border-bottom-left-radius: 4px; }
    .msg.loading { color: #aaa; font-style: italic; }
    .rag-label {
      align-self: flex-start;
      font-size: 11px;
      color: #999;
      margin-top: -6px;
      padding: 0 6px;
      max-width: 78%;
    }

    .input-area {
      display: flex;
      gap: 8px;
      padding: 12px 16px;
      border-top: 1px solid #e5e5e5;
      background: #fff;
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
        <p style="margin-top:8px">검색 방식을 선택하세요</p>
      </div>
      <div class="mode-cards">
        <div class="mode-card" onclick="startChat('basic', '기본 모델', 'basic')">
          <div class="icon">💬</div>
          <h3>기본 모델</h3>
          <p>검색 없이 QA LLM이 직접 답변합니다. (Stage 2)</p>
        </div>
        <div class="mode-card" onclick="startChat('rag', 'RAG 기반 검색', 'rag')">
          <div class="icon">🔍</div>
          <h3>RAG 기반 검색</h3>
          <p>TF-IDF로 관련 문서를 검색한 뒤 추출형 QA로 답변합니다.</p>
        </div>
        <div class="mode-card" onclick="startChat('langchain', 'LangChain 기반 검색', 'langchain')">
          <div class="icon">⚡</div>
          <h3>LangChain 기반 검색</h3>
          <p>BM25 + FAISS 하이브리드 검색 후 추출형 QA로 답변합니다.</p>
        </div>
      </div>
    </div>

    <!-- 채팅 화면 (공유) -->
    <div id="chat-view">
      <div class="chat-header">
        <button class="back-btn" onclick="goBack()">← 뒤로</button>
        <span id="chat-title">채팅</span>
        <span class="mode-badge" id="mode-badge"></span>
      </div>
      <div class="messages" id="messages"></div>
      <div class="input-area">
        <input id="msg-input" placeholder="질문을 입력하세요" autocomplete="off">
        <button id="send-btn" onclick="send()">전송</button>
      </div>
    </div>

  </div>

  <script>
    let currentEndpoint = '';

    function startChat(mode, title, badge) {
      currentEndpoint = '/chat/' + mode;
      document.getElementById('chat-title').textContent = title;
      document.getElementById('mode-badge').textContent = badge;
      document.getElementById('messages').innerHTML = '';
      document.getElementById('select-view').style.display = 'none';
      document.getElementById('chat-view').style.display = 'flex';
      document.getElementById('chat-view').style.flexDirection = 'column';
      document.getElementById('msg-input').focus();
    }

    function goBack() {
      document.getElementById('select-view').style.display = 'flex';
      document.getElementById('chat-view').style.display = 'none';
      currentEndpoint = '';
    }

    function addMsg(cls, text) {
      const box = document.getElementById('messages');
      const div = document.createElement('div');
      div.className = 'msg ' + cls;
      div.textContent = text;
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
      return div;
    }

    function addLabel(text) {
      const box = document.getElementById('messages');
      const div = document.createElement('div');
      div.className = 'rag-label';
      div.textContent = text;
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
    }

    async function send() {
      const input = document.getElementById('msg-input');
      const btn = document.getElementById('send-btn');
      const question = input.value.trim();
      if (!question || !currentEndpoint) return;

      addMsg('user', question);
      input.value = '';
      btn.disabled = true;
      const loading = addMsg('assistant loading', '생성 중…');

      try {
        const res = await fetch(currentEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question }),
        });
        const data = await res.json();
        loading.textContent = data.answer || '(빈 응답)';
        loading.className = 'msg assistant';
        if (data.used_rag && data.retrieved_context) {
          addLabel('참고: ' + data.retrieved_context);
        }
      } catch (e) {
        loading.textContent = '오류가 발생했습니다.';
        loading.className = 'msg assistant';
      } finally {
        btn.disabled = false;
        input.focus();
      }
    }

    document.getElementById('msg-input').addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.isComposing && e.keyCode !== 229) send();
    });
  </script>
</body>
</html>
"""
