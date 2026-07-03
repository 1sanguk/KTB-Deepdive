from typing import TypedDict


class GraphState(TypedDict):
    """LangGraph 파이프라인 전체 노드가 공유하는 상태 스키마."""

    query: str           # 사용자 입력 질문
    documents: list[str] # retrieve 노드가 채우는 검색 청크 목록
    score: float         # 하이브리드 유사도 점수 — grade 노드의 임계값 판단 기준
    answer: str          # 최종 생성 답변
    used_rag: bool       # grade 노드가 설정하는 RAG 경로 사용 여부
    retry_count: int     # increment_retry 노드가 올리는 현재 재시도 횟수


class AgentState(TypedDict):
    query: str
    messages: list
    stop_reason: str
    answer: str
    used_rag: bool
    retrieved_context: str

    