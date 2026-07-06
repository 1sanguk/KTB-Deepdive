"""Claude Haiku로 질문을 분류해 적절한 체인을 선택하는 라우터."""

import os
import anthropic

_SYSTEM = (
    "질문을 아래 세 가지 중 하나로만 분류하세요. 분류명만 단답으로 답하세요.\n\n"
    "chit_chat: 인사, 감정 표현, 일상 잡담 (예: 안녕, 심심해, 배고파, 좋아, 사랑해)\n"
    "factual: 역사·과학·인물·지리 등 사실 확인이 필요한 질문 (예: 세종대왕은 언제 태어났어?, 파리는 어느 나라 수도야?)\n"
    "general: 추천·의견·개념 설명을 요구하는 일반 질문 (예: 파이썬이 뭐야?, 영화 추천해줘, 취업 준비 어떻게 해?)"
)

# 분류 결과 → 체인 이름
CHAIN_FOR: dict[str, str] = {
    "chit_chat": "basic",
    "factual":   "langgraph",
    "general":   "langchain",
}

# 체인 이름 → UI 표시 레이블
MODE_LABEL: dict[str, str] = {
    "basic":     "기본 모델",
    "langchain": "LangChain 검색",
    "langgraph": "LangGraph 검색",
}


def classify_question(question: str) -> str:
    """질문을 분류해 'chit_chat' | 'factual' | 'general' 을 반환. 오류 시 'general' 폴백."""
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            temperature=0,
            system=_SYSTEM,
            messages=[{"role": "user", "content": question}],
        )
        label = msg.content[0].text.strip().lower()
        if label in CHAIN_FOR:
            return label
        return "general"
    except Exception:
        return "general"
