import sys
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# model/ 디렉토리(아키텍처, BPE, 체크포인트)를 모듈 검색 경로에 추가한다.
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_DIR))

from model import device, SOP_GPT, SOP_GPT_Span
from bpe import build_vocab, base_alphabet, tokenize, decode, load_bpe
from chat import extract_answer
import rag

BPE_PATH = MODEL_DIR / "bpe_vocab.json"
GEN_CKPT = MODEL_DIR / "SOP_GPT.pt"        # Stage 1: 이어쓰기
QA_CKPT = MODEL_DIR / "SOP_GPT_qa.pt"      # Stage 2: 잡담형 Q&A (RAG 폴백용)
SPAN_CKPT = MODEL_DIR / "SOP_GPT_span.pt"  # Stage 4: 추출형 RAG QA

RAG_SIM_THRESHOLD = 0.515  # TF-IDF+임베딩 하이브리드 점수 기준 (held-out 검증 최적값, 정확도 82.7%)

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

rag_tfidf_vectorizer, rag_tfidf_matrix, rag_embed_matrix, rag_passages, rag_norm_bounds = rag.build_index()

# "." / "?" / "!"로 끝나는 토큰 -> 한 문장이 끝나면 멈춤 (이어쓰기 모드)
SENTENCE_STOP_TOKENS = {i for t, i in stoi.items() if t and t[-1] in ".?!"}
# "\n"으로 끝나는 토큰 -> "답변: ..." 한 줄이 끝나면 멈춤 (Q&A 모드)
LINE_STOP_TOKENS = {i for t, i in stoi.items() if t.endswith("\n")}

START_ID = 0  # 입력 토큰이 없을 때 사용할 시작 토큰


def encode(text):
    return [stoi[t] for t in tokenize(text, merges, base_set) if t in stoi]


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
    """Stage 1: 입력 문장을 이어써서 완전한 문장으로 끝낸다."""
    ids = encode(req.prompt) or [START_ID]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = gen_model.generate(
        idx, req.max_new_tokens,
        stop_tokens=SENTENCE_STOP_TOKENS, temperature=0.7, top_p=0.9, repetition_penalty=1.3,
    )[0].tolist()
    return GenerateResponse(text=decode(out[len(ids):], itos))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """검색 유사도가 충분히 높으면(RAG_SIM_THRESHOLD 이상) Stage4 추출형 QA로 KorQuAD 문서에서
    정답 구간을 직접 뽑아 답하고, 관련 문서가 없으면 Stage2 잡담형 모델로 자연스럽게 답한다."""
    context, score = rag.best_match(
        req.question, rag_tfidf_vectorizer, rag_tfidf_matrix, rag_embed_matrix, rag_norm_bounds, rag_passages
    )

    if score >= RAG_SIM_THRESHOLD:
        answer = extract_answer(span_model, stoi, merges, base_set, req.question, context)
        return ChatResponse(answer=answer, retrieved_context=context, used_rag=True)

    prompt = f"질문: {req.question}\n답변: "
    ids = encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = qa_model.generate(
        idx, 60,
        stop_tokens=LINE_STOP_TOKENS, temperature=0.8, top_k=40, repetition_penalty=1.3,
    )[0].tolist()
    return ChatResponse(answer=decode(out[len(ids):], itos).strip(), retrieved_context="", used_rag=False)


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>SOP_GPT 한국어 챗봇</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: #f7f7f8;
      color: #1a1a1a;
      height: 100vh;
      display: flex;
      justify-content: center;
    }
    .app { width: 100%; max-width: 720px; display: flex; flex-direction: column; height: 100vh; }

    #select-view {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 24px;
      padding: 24px;
      text-align: center;
    }
    #select-view h1 { margin: 0; font-size: 24px; }
    #select-view p { margin: 0; color: #666; }
    .mode-buttons { display: flex; gap: 16px; flex-wrap: wrap; justify-content: center; }
    .mode-card {
      width: 200px;
      padding: 24px;
      border: 1px solid #e0e0e0;
      border-radius: 12px;
      background: #fff;
      cursor: pointer;
      transition: box-shadow 0.15s, transform 0.15s;
      text-align: left;
    }
    .mode-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); transform: translateY(-2px); }
    .mode-card h3 { margin: 0 0 8px; font-size: 16px; }
    .mode-card p { margin: 0; font-size: 13px; color: #888; }

    .chat-view { flex: 1; display: none; flex-direction: column; height: 100vh; }
    .chat-header {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      border-bottom: 1px solid #e5e5e5;
      background: #fff;
    }
    .back-btn { border: none; background: none; font-size: 14px; color: #555; cursor: pointer; padding: 6px 10px; border-radius: 6px; }
    .back-btn:hover { background: #f0f0f0; }
    .chat-title { font-size: 15px; font-weight: 600; }

    .messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
    .msg { max-width: 80%; padding: 10px 14px; border-radius: 16px; line-height: 1.5; font-size: 14px; white-space: pre-wrap; }
    .msg.user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
    .msg.assistant { align-self: flex-start; background: #fff; border: 1px solid #e5e5e5; border-bottom-left-radius: 4px; }
    .msg.loading { color: #999; font-style: italic; }
    .rag-context { align-self: flex-start; max-width: 80%; font-size: 11px; color: #999; margin-top: -6px; padding: 0 4px; }

    .input-area { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid #e5e5e5; background: #fff; }
    .input-area input {
      flex: 1;
      padding: 10px 14px;
      border: 1px solid #d0d0d0;
      border-radius: 20px;
      font-size: 14px;
      outline: none;
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
    }
    .input-area button:hover { background: #1d4ed8; }
    .input-area button:disabled { background: #aac; cursor: default; }
  </style>
</head>
<body>
  <div class="app">
    <div id="select-view">
      <h1>SOP_GPT 한국어 챗봇</h1>
      <p>모드를 선택하세요</p>
      <div class="mode-buttons">
        <div class="mode-card" onclick="showView('generate')">
          <h3>이어쓰기</h3>
          <p>문장을 입력하면 이어서 완성된 문장을 생성합니다.</p>
        </div>
        <div class="mode-card" onclick="showView('chat')">
          <h3>Q&amp;A</h3>
          <p>질문을 입력하면 답변을 생성합니다.</p>
        </div>
      </div>
    </div>

    <div id="generate-view" class="chat-view">
      <div class="chat-header">
        <button class="back-btn" onclick="showView('select')">&larr; 뒤로</button>
        <div class="chat-title">이어쓰기</div>
      </div>
      <div class="messages" id="gen-messages"></div>
      <div class="input-area">
        <input id="gen-input" placeholder="예: 오늘 날씨는" autocomplete="off">
        <button id="gen-btn" onclick="sendGenerate()">입력</button>
      </div>
    </div>

    <div id="chat-view" class="chat-view">
      <div class="chat-header">
        <button class="back-btn" onclick="showView('select')">&larr; 뒤로</button>
        <div class="chat-title">Q&amp;A</div>
      </div>
      <div class="messages" id="chat-messages"></div>
      <div class="input-area">
        <input id="chat-input" placeholder="예: 오늘 기분 어때?" autocomplete="off">
        <button id="chat-btn" onclick="sendChat()">입력</button>
      </div>
    </div>
  </div>

  <script>
    function showView(name) {
      document.getElementById('select-view').style.display = name === 'select' ? 'flex' : 'none';
      document.getElementById('generate-view').style.display = name === 'generate' ? 'flex' : 'none';
      document.getElementById('chat-view').style.display = name === 'chat' ? 'flex' : 'none';
    }

    function addMessage(containerId, className, text) {
      const container = document.getElementById(containerId);
      const div = document.createElement('div');
      div.className = 'msg ' + className;
      div.textContent = text;
      container.appendChild(div);
      container.scrollTop = container.scrollHeight;
      return div;
    }

    async function sendGenerate() {
      const input = document.getElementById('gen-input');
      const btn = document.getElementById('gen-btn');
      const prompt = input.value.trim();
      if (!prompt) return;

      addMessage('gen-messages', 'user', prompt);
      input.value = '';
      btn.disabled = true;
      const loading = addMessage('gen-messages', 'assistant loading', '생성 중...');

      try {
        const res = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt }),
        });
        const data = await res.json();
        loading.textContent = prompt + data.text;
        loading.className = 'msg assistant';
      } catch (e) {
        loading.textContent = '오류가 발생했습니다.';
        loading.className = 'msg assistant';
      } finally {
        btn.disabled = false;
        input.focus();
      }
    }

    async function sendChat() {
      const input = document.getElementById('chat-input');
      const btn = document.getElementById('chat-btn');
      const question = input.value.trim();
      if (!question) return;

      addMessage('chat-messages', 'user', question);
      input.value = '';
      btn.disabled = true;
      const loading = addMessage('chat-messages', 'assistant loading', '생성 중...');

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question }),
        });
        const data = await res.json();
        loading.textContent = data.answer;
        loading.className = 'msg assistant';
        if (data.used_rag) {
          const ctx = document.createElement('div');
          ctx.className = 'rag-context';
          ctx.textContent = '참고: ' + data.retrieved_context;
          loading.after(ctx);
        }
      } catch (e) {
        loading.textContent = '오류가 발생했습니다.';
        loading.className = 'msg assistant';
      } finally {
        btn.disabled = false;
        input.focus();
      }
    }

    document.getElementById('gen-input').addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.isComposing && e.keyCode !== 229) sendGenerate();
    });
    document.getElementById('chat-input').addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.isComposing && e.keyCode !== 229) sendChat();
    });
  </script>
</body>
</html>
"""
