"""LCEL(LangChain Expression Language) 기반 RAG 파이프라인 체인.

build_basic_chain  : 검색 없이 QA LLM 으로 직접 답변
build_rag_chain    : 검색기 + 유사도 임계값 기반 라우팅 (Span 추출 or LLM 폴백)
"""

from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

QA_PROMPT = PromptTemplate.from_template("질문: {question}\n답변: ")


def build_basic_chain(llm):
    """Stage 2: 검색 없이 QA 모델이 직접 답변하는 LCEL 체인.

    Input : str  — 질문
    Output: str  — 답변
    """
    return (
        {"question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )


def build_rag_chain(retriever, llm, span_extractor_fn: Callable, threshold: float):
    """RAG 기반 LCEL 체인.

    retriever.best_match(question) → (context, score)
    score >= threshold : span_extractor_fn 으로 정답 구간 추출  (Stage 4)
    score <  threshold : QA LLM 으로 폴백                       (Stage 2)

    Input : str  — 질문
    Output: dict — {answer: str, retrieved_context: str, used_rag: bool}
    """
    def retrieve_and_answer(question: str) -> dict:
        context, score = retriever.best_match(question)
        if score >= threshold:
            answer = span_extractor_fn({"question": question, "context": context})
            return {"answer": answer, "retrieved_context": context, "used_rag": True}
        answer = llm.invoke(f"질문: {question}\n답변: ")
        return {"answer": answer, "retrieved_context": "", "used_rag": False}

    return RunnableLambda(retrieve_and_answer)
