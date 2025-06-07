# 대화 기억 에이전트 구현
# 필요한 패키지 설치:
# pip install --upgrade chromadb sentence-transformers google-generativeai streamlit

import os
import json
import datetime
from typing import List, Dict, Any, Generator
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import streamlit as st

class MemoryAgent:
    def __init__(self, gemini_api_key: str):
        # ChromaDB 클라이언트 초기화 (로컬 저장)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # 컬렉션 생성/로드
        self.collection = self.chroma_client.get_or_create_collection(
            name="conversation_memory",
            metadata={"hnsw:space": "cosine"}
        )
        
        # 로컬 임베딩 모델 초기화
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Gemini API 설정
        genai.configure(api_key=gemini_api_key)
        
        # [수정 1] 모델 이름을 더 명시적인 'gemini-1.0-pro'로 변경
        self.llm = genai.GenerativeModel('gemini-1.0-pro')
        
        # [수정 2] 안전 설정 정의 (과도한 차단을 막기 위해 보통 수준으로 설정)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # 대화 ID 카운터
        self.conversation_id = self.collection.count() // 2
    
    def create_embedding(self, text: str) -> List[float]:
        """텍스트를 벡터로 변환"""
        return self.embedding_model.encode(text).tolist()
    
    def store_conversation(self, user_input: str, assistant_response: str):
        """대화를 벡터 DB에 저장"""
        timestamp = datetime.datetime.now().isoformat()
        
        user_id = f"user_{self.conversation_id}"
        assistant_id = f"assistant_{self.conversation_id}"

        # 사용자 입력과 어시스턴트 응답을 한 번에 추가하여 효율성 증대
        self.collection.add(
            embeddings=[
                self.create_embedding(user_input),
                self.create_embedding(assistant_response)
            ],
            documents=[user_input, assistant_response],
            metadatas=[
                {"type": "user_input", "timestamp": timestamp, "conversation_id": self.conversation_id},
                {"type": "assistant_response", "timestamp": timestamp, "conversation_id": self.conversation_id}
            ],
            ids=[user_id, assistant_id]
        )
        
        self.conversation_id += 1
    
    def search_memory(self, query: str, n_results: int = 5) -> List[Dict]:
        """관련 기억 검색"""
        query_embedding = self.create_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        memories = []
        if not results['documents'][0]:
            return []

        for i, doc in enumerate(results['documents'][0]):
            memories.append({
                'content': doc,
                'metadata': results['metadatas'][0][i],
                'similarity': 1 - results['distances'][0][i]
            })
        
        return memories
    
    # [수정 3] 스트리밍 응답을 위해 반환 타입을 Generator로 변경
    def generate_response_stream(self, user_input: str) -> Generator[str, None, None]:
        """기억을 바탕으로 응답을 스트리밍으로 생성"""
        memories = self.search_memory(user_input, n_results=3)
        
        context = "관련 과거 대화:\n"
        if memories:
            for i, memory in enumerate(memories):
                if memory['similarity'] > 0.3:
                    context += f"{i+1}. [{memory['metadata']['type']}] {memory['content']}\n"
        else:
            context += "관련된 과거 대화가 없습니다.\n"

        prompt = f"""
당신은 과거 대화를 기억하는 AI 어시스턴트입니다.

{context}

현재 사용자 질문: {user_input}

과거 대화 내용을 참고하여 자연스럽고 도움이 되는 답변을 해주세요.
만약 관련된 과거 대화가 있다면 그것을 언급하며 답변해주세요.
"""
        
        try:
            # stream=True 옵션 사용
            response_stream = self.llm.generate_content(
                prompt,
                stream=True,
                safety_settings=self.safety_settings
            )
            for chunk in response_stream:
                # 일부 chunk에 text가 없을 수 있으므로 확인
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"응답 생성 중 오류가 발생했습니다: {str(e)}"
    
    # [수정 4] 스트리밍을 처리하고 전체 응답을 반환하는 chat 메소드
    def chat(self, user_input: str) -> Generator[str, None, None]:
        """대화 처리 (기억 검색 + 스트리밍 응답 생성 + 저장)"""
        full_response_parts = []
        
        # 스트림에서 응답을 받아오면서 yield로 전달
        for chunk in self.generate_response_stream(user_input):
            full_response_parts.append(chunk)
            yield chunk

        # 전체 응답이 완료되면 대화 저장
        full_response = "".join(full_response_parts)
        if not full_response.startswith("응답 생성 중 오류가 발생했습니다"):
             self.store_conversation(user_input, full_response)
    
    def get_conversation_stats(self) -> Dict:
        """대화 통계 조회"""
        total_messages = self.collection.count()
        return {
            "total_stored_messages": total_messages,
            "conversation_pairs": total_messages // 2
        }

# Streamlit 웹 인터페이스
def main():
    st.set_page_config(page_title="대화 기억 에이전트", page_icon="🤖")
    
    st.title("🤖 대화 기억 에이전트")
    st.markdown("과거 대화를 기억하고 맥락을 이해하는 AI 어시스턴트")
    
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ""
    
    with st.sidebar:
        st.header("설정")
        api_key = st.text_input("Google AI Studio API 키", value=st.session_state.gemini_api_key, type="password", help="https://aistudio.google.com/app/apikey 에서 발급받으세요")
        
        if api_key:
            st.session_state.gemini_api_key = api_key
        
        st.markdown("---")
        
        if st.session_state.gemini_api_key:
            if 'agent' not in st.session_state:
                try:
                    st.session_state.agent = MemoryAgent(st.session_state.gemini_api_key)
                    st.success("에이전트가 성공적으로 초기화되었습니다.")
                except Exception as e:
                    st.error(f"에이전트 초기화 실패: {e}")

            if 'agent' in st.session_state:
                stats = st.session_state.agent.get_conversation_stats()
                st.metric("저장된 대화 쌍", stats['conversation_pairs'])
                st.metric("총 메시지 수", stats['total_stored_messages'])
    
    if not st.session_state.gemini_api_key:
        st.warning("Google AI Studio API 키를 입력해주세요.")
        st.info("🔗 API 키 발급: https://aistudio.google.com/app/apikey")
        return
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("메시지를 입력하세요..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            # [수정 5] st.write_stream을 사용하여 스트리밍 응답 표시
            response_generator = st.session_state.agent.chat(prompt)
            full_response = st.write_stream(response_generator)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # 사이드바 통계 즉시 새로고침
        st.rerun()

    with st.expander("🔍 기억 검색 테스트"):
        search_query = st.text_input("검색할 내용을 입력하세요:")
        if search_query and 'agent' in st.session_state:
            with st.spinner("기억 검색 중..."):
                memories = st.session_state.agent.search_memory(search_query)
                if not memories:
                    st.info("관련된 기억을 찾지 못했습니다.")
                for i, memory in enumerate(memories):
                    st.write(f"**{i+1}. 유사도: {memory['similarity']:.3f}**")
                    st.write(f"타입: {memory['metadata']['type']}")
                    st.write(f"내용: {memory['content']}")
                    st.write("---")

if __name__ == "__main__":
    main()