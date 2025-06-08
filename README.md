# 🤖 대화 기억 에이전트 (Memory Agent)

과거 대화를 기억하고 맥락을 이해하는 AI 어시스턴트입니다. ChromaDB를 사용한 벡터 검색과 Google Gemini API를 통해 개인화된 대화 경험을 제공합니다.

## ✨ 주요 기능

- **대화 기억**: 모든 대화를 벡터 데이터베이스에 저장하여 장기간 기억 유지
- **맥락 인식**: 과거 대화 내용을 바탕으로 개인화된 응답 생성
- **의미 기반 검색**: 키워드가 아닌 의미 기반으로 관련 대화 검색
- **실시간 스트리밍**: 응답을 실시간으로 스트리밍하여 빠른 사용자 경험
- **대화 통계**: 저장된 대화 수와 통계 정보 제공
- **직관적인 웹 인터페이스**: Streamlit 기반의 사용하기 쉬운 UI

## 🛠️ 기술 스택

- **벡터 데이터베이스**: ChromaDB (로컬 영구 저장)
- **임베딩 모델**: SentenceTransformer (all-MiniLM-L6-v2)
- **LLM**: Google Gemini 1.0 Pro
- **웹 프레임워크**: Streamlit
- **언어**: Python 3.8+

## 📦 설치 및 설정

### 1. 필수 패키지 설치

```bash
pip install --upgrade chromadb sentence-transformers google-generativeai streamlit
```

### 2. Google AI Studio API 키 발급

1. [Google AI Studio](https://aistudio.google.com/app/apikey)에 접속
2. API 키 생성
3. 발급받은 API 키를 안전하게 보관

### 3. 실행

```bash
streamlit run llm.py
```

## 🚀 사용 방법

### 웹 인터페이스 사용

1. 애플리케이션 실행 후 사이드바에서 Google AI Studio API 키 입력
2. 에이전트 초기화 완료 후 채팅 시작
3. 과거 대화 내용이 자동으로 참조되어 맥락있는 응답 제공

### 프로그래밍 방식 사용

```python
from llm import MemoryAgent

# 에이전트 초기화
agent = MemoryAgent(gemini_api_key="your_api_key_here")

# 대화 진행 (스트리밍)
for chunk in agent.chat("안녕하세요!"):
    print(chunk, end="")

# 기억 검색
memories = agent.search_memory("특정 주제")
for memory in memories:
    print(f"유사도: {memory['similarity']:.3f}")
    print(f"내용: {memory['content']}")
```

## 🔧 주요 클래스 및 메서드

### MemoryAgent 클래스

#### 초기화
- `__init__(gemini_api_key: str)`: 에이전트 초기화

#### 핵심 메서드
- `chat(user_input: str) -> Generator[str, None, None]`: 스트리밍 대화 처리
- `search_memory(query: str, n_results: int = 5) -> List[Dict]`: 관련 기억 검색
- `store_conversation(user_input: str, assistant_response: str)`: 대화 저장
- `get_conversation_stats() -> Dict`: 대화 통계 조회

## 📊 데이터 구조

### 대화 저장 형식
```json
{
  "id": "user_0",
  "document": "사용자 입력 텍스트",
  "embedding": [0.1, 0.2, ...],
  "metadata": {
    "type": "user_input",
    "timestamp": "2025-01-01T10:00:00",
    "conversation_id": 0
  }
}
```

### 검색 결과 형식
```python
{
  "content": "대화 내용",
  "metadata": {"type": "user_input", "timestamp": "...", "conversation_id": 0},
  "similarity": 0.85
}
```

## ⚙️ 설정 옵션

### 임베딩 모델 변경
```python
# 다른 SentenceTransformer 모델 사용
self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

### 검색 결과 수 조정
```python
# 더 많은 관련 기억 검색
memories = agent.search_memory(query, n_results=10)
```

### 유사도 임계값 조정
```python
# 더 높은 유사도만 사용
if memory['similarity'] > 0.5:  # 기본값: 0.3
    context += f"{memory['content']}\n"
```

## 🗂️ 디렉토리 구조

```
project/
│
├── llm.py              # 메인 애플리케이션 파일
├── chroma_db/          # ChromaDB 데이터 저장소 (자동 생성)
│   ├── chroma.sqlite3
│   └── ...
└── README.md
```

## 🔒 보안 고려사항

- API 키는 환경 변수나 설정 파일로 관리 권장
- 로컬 ChromaDB는 기본적으로 암호화되지 않음
- 민감한 정보 포함 시 추가 보안 조치 필요

## 🐛 문제 해결

### 일반적인 오류

**1. API 키 관련 오류**
```
해결: Google AI Studio에서 올바른 API 키 확인 및 재발급
```

**2. ChromaDB 초기화 실패**
```
해결: chroma_db 폴더 삭제 후 재시작
```

**3. 임베딩 모델 다운로드 실패**
```bash
# 수동 다운로드
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 새로운 기능 브랜치를 만듭니다 (`git checkout -b feature/AmazingFeature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/AmazingFeature`)
5. Pull Request를 생성합니다

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.


## 🙏 감사의 말

- [ChromaDB](https://www.trychroma.com/) - 벡터 데이터베이스
- [Sentence Transformers](https://www.sbert.net/) - 임베딩 모델
- [Google AI](https://ai.google/) - Gemini API
- [Streamlit](https://streamlit.io/) - 웹 프레임워크

---

**참고**: 이 프로젝트는 개인 프로젝트이며, 상업적 사용 시 각 서비스의 이용약관을 확인하시기 바랍니다.
