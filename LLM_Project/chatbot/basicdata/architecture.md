# LLM 챗봇 아키텍처

이 문서는 `LLM_Project/chatbot`의 실제 코드 구성을 기준으로 웹 요청, RAG 파이프라인,
모델 추론, 데이터 저장소 사이의 관계를 나타낸다.

## 전체 시스템 구조

```mermaid
flowchart TB
    user["사용자"]

    subgraph client["클라이언트"]
        ui["웹 UI<br/>static/index.html"]
    end

    subgraph serving["FastAPI 서빙 계층 · source/app"]
        app["app.py<br/>애플리케이션 진입점"]
        chatRouter["chat.py<br/>일반 응답 API"]
        streamRouter["stream.py<br/>SSE 스트리밍 API"]
        streaming["streaming.py<br/>모드 라우팅·병렬 실행"]
        auth["auth.py<br/>인증"]
        sessions["sessions.py<br/>세션 관리"]
        history["history.py<br/>대화 기록"]
        state["state.py<br/>모델·검색기·그래프 초기화"]
    end

    subgraph orchestration["오케스트레이션 계층"]
        autoRouter["질문 자동 분류<br/>Claude Haiku"]
        lcel["LangChain LCEL<br/>Basic·TF-IDF·Hybrid RAG"]
        langgraph["LangGraph<br/>검색·판정·재시도·생성"]
        compare["Compare 실행기<br/>4개 그래프 병렬 실행"]
        judge["Gemini Flash<br/>답변·세션 평가"]
    end

    subgraph retrieval["검색 계층"]
        tfidf["TF-IDF Retriever"]
        hybrid["Hybrid Retriever"]
        bm25["BM25<br/>키워드 검색"]
        faiss["FAISS + Ko-SRoBERTa<br/>의미 검색"]
        calibrate["점수 보정·결합<br/>0.5 sparse + 0.5 dense"]
    end

    subgraph models["LLM·추론 계층"]
        sopQA["SOP_GPT QA<br/>직접 생성"]
        sopSpan["SOP_GPT Span<br/>정답 구간 추출"]
        claude["Claude Haiku API"]
        vllmOllama["vLLM / Ollama<br/>OpenAI-compatible API 서버"]
        qwenBF16["Qwen3-1.7B BF16<br/>Transformers (로컬 폴백)"]
        qwenQ4["Qwen3-1.7B Q4_K_M<br/>llama.cpp (로컬 폴백)"]
    end

    subgraph data["데이터·영속화"]
        knowledge["RAG 지식 베이스<br/>KorQuAD + ragdata Markdown"]
        modelFiles["모델 파일<br/>HF Hub 캐시(SOP GPT) + GGUF(Qwen)"]
        jsonData["JSON 데이터<br/>사용자·세션·대화 기록"]
        cache["검색 인덱스 캐시<br/>source/.cache"]
        memory["LangGraph MemorySaver<br/>스레드별 단기 메모리"]
    end

    subgraph observability["관측 (비활성화)"]
        langsmith["LangSmith<br/>트레이싱·평가"]
    end

    user --> ui
    ui -->|"HTTP / SSE"| app
    app --> chatRouter
    app --> streamRouter
    chatRouter --> state
    streamRouter --> streaming
    streaming --> state
    app --> auth
    app --> sessions
    streaming --> history

    streaming --> autoRouter
    streaming --> lcel
    streaming --> langgraph
    streaming --> compare
    compare --> langgraph
    compare --> judge

    lcel --> tfidf
    lcel --> hybrid
    langgraph --> hybrid
    autoRouter --> claude
    hybrid --> bm25
    hybrid --> faiss
    bm25 --> calibrate
    faiss --> calibrate

    tfidf --> knowledge
    bm25 --> knowledge
    faiss --> knowledge
    faiss --> cache

    lcel --> sopQA
    lcel --> sopSpan
    langgraph --> sopQA
    langgraph --> sopSpan
    langgraph --> claude
    langgraph --> vllmOllama
    langgraph -.->|"폴백"| qwenBF16
    langgraph -.->|"폴백"| qwenQ4
    vllmOllama -.->|"폴백"| qwenBF16
    vllmOllama -.->|"폴백"| qwenQ4
    sopQA --> modelFiles
    sopSpan --> modelFiles
    qwenBF16 --> modelFiles
    qwenQ4 --> modelFiles

    auth --> jsonData
    sessions --> jsonData
    history --> jsonData
    langgraph --> memory
    lcel -. 추적 .-> langsmith
    langgraph -. 추적 .-> langsmith

    classDef clientNode fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef apiNode fill:#dcfce7,stroke:#16a34a,color:#052e16
    classDef flowNode fill:#fef3c7,stroke:#d97706,color:#451a03
    classDef modelNode fill:#f3e8ff,stroke:#9333ea,color:#3b0764
    classDef dataNode fill:#f1f5f9,stroke:#64748b,color:#0f172a
    class ui clientNode
    class app,chatRouter,streamRouter,streaming,auth,sessions,history,state apiNode
    class autoRouter,lcel,langgraph,compare,judge,tfidf,hybrid,bm25,faiss,calibrate flowNode
    class sopQA,sopSpan,claude,vllmOllama,qwenBF16,qwenQ4 modelNode
    class knowledge,modelFiles,jsonData,cache,memory,langsmith dataNode
```

## LangGraph RAG 실행 흐름

SOP_GPT, Claude, Qwen 그래프는 같은 제어 흐름을 공유하며 마지막 생성 노드에서 사용할
모델만 다르다.

```mermaid
flowchart LR
    start(("요청")) --> init["상태 초기화"]
    init --> retrieve["BM25 + FAISS 검색"]
    retrieve --> grade{"관련성 기준 충족?"}
    grade -->|"예"| context["검색 문맥 기반 응답"]
    grade -->|"아니요, 재시도 가능"| retry["retry_count 증가<br/>임계값 완화"]
    retry --> retrieve
    grade -->|"아니요, 최대 2회 소진"| direct["모델 직접 응답"]
    context --> finish(("SSE 응답"))
    direct --> finish
```

## Compare 모드 실행 흐름

```mermaid
sequenceDiagram
    actor User as 사용자
    participant UI as 웹 UI
    participant API as FastAPI SSE
    participant Runner as Compare 실행기
    participant SOP as SOP_GPT Graph
    participant Claude as Claude Graph
    participant Qwen as Qwen Graph (BF16·Q4)
    participant vLLM as vLLM / Ollama API
    participant Judge as Gemini Judge
    participant Store as JSON History

    User->>UI: 질문 입력
    UI->>API: POST /chat/compare/stream
    API->>Runner: compare_stream
    par 모델별 LangGraph 병렬 실행
        Runner->>SOP: 질문 + thread_id (Semaphore 순차)
        Runner->>Claude: 질문 + thread_id:c (병렬)
        Runner->>Qwen: 질문 + thread_id:bf16·q4 (Semaphore 순차)
    end
    Qwen->>vLLM: HTTP POST /v1/chat/completions
    vLLM-->>Qwen: 응답 + elapsed 시간
    SOP-->>Runner: 최종 답변 + elapsed
    Claude-->>Runner: 최종 답변 + elapsed
    Qwen-->>Runner: 최종 답변 + elapsed
    Runner-->>UI: SSE model_text · model_done(elapsed) 전송
    Runner->>Judge: 4개 답변 + 소요 시간 포함 프롬프트
    Judge-->>Runner: 시간 대비 최선 답변 선정 + 이유
    Runner->>Store: 비교 결과 저장
    Runner-->>UI: judge_done(best_model, best_text) 전송
```
