# 대화 기억 에이전트 구현
# 필요한 패키지 설치:
# pip install chromadb sentence-transformers google-generativeai streamlit

import os
import json
import datetime
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
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
        self.llm = genai.GenerativeModel('gemini-pro')
        
        # 대화 ID 카운터
        self.conversation_id = 0
    
    def create_embedding(self, text: str) -> List[float]:
        """텍스트를 벡터로 변환"""
        return self.embedding_model.encode(text).tolist()
    
    def store_conversation(self, user_input: str, assistant_response: str):
        """대화를 벡터 DB에 저장"""
        timestamp = datetime.datetime.now().isoformat()
        
        # 사용자 입력 저장
        user_embedding = self.create_embedding(user_input)
        user_id = f"user_{self.conversation_id}"
        
        self.collection.add(
            embeddings=[user_embedding],
            documents=[user_input],
            metadatas=[{
                "type": "user_input",
                "timestamp": timestamp,
                "conversation_id": self.conversation_id
            }],
            ids=[user_id]
        )
        
        # 어시스턴트 응답 저장
        assistant_embedding = self.create_embedding(assistant_response)
        assistant_id = f"assistant_{self.conversation_id}"
        
        self.collection.add(
            embeddings=[assistant_embedding],
            documents=[assistant_response],
            metadatas=[{
                "type": "assistant_response",
                "timestamp": timestamp,
                "conversation_id": self.conversation_id
            }],
            ids=[assistant_id]
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
        for i, doc in enumerate(results['documents'][0]):
            memories.append({
                'content': doc,
                'metadata': results['metadatas'][0][i],
                'similarity': 1 - results['distances'][0][i]  # 거리를 유사도로 변환
            })
        
        return memories
    
    def generate_response(self, user_input: str) -> str:
        """기억을 바탕으로 응답 생성"""
        # 관련 기억 검색
        memories = self.search_memory(user_input, n_results=3)
        
        # 컨텍스트 구성
        context = "관련 과거 대화:\n"
        for i, memory in enumerate(memories):
            if memory['similarity'] > 0.3:  # 유사도 임계값
                context += f"{i+1}. [{memory['metadata']['type']}] {memory['content']}\n"
        
        # 프롬프트 구성
        prompt = f"""
당신은 과거 대화를 기억하는 AI 어시스턴트입니다.

{context}

현재 사용자 질문: {user_input}

과거 대화 내용을 참고하여 자연스럽고 도움이 되는 답변을 해주세요.
만약 관련된 과거 대화가 있다면 그것을 언급하며 답변해주세요.
"""
        
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"응답 생성 중 오류가 발생했습니다: {str(e)}"
    
    def chat(self, user_input: str) -> str:
        """대화 처리 (기억 검색 + 응답 생성 + 저장)"""
        # 응답 생성
        response = self.generate_response(user_input)
        
        # 대화 저장
        self.store_conversation(user_input, response)
        
        return response
    
    def get_conversation_stats(self) -> Dict:
        """대화 통계 조회"""
        total_conversations = self.collection.count()
        return {
            "total_stored_messages": total_conversations,
            "conversation_pairs": total_conversations // 2
        }

# Streamlit 웹 인터페이스
def main():
    st.set_page_config(page_title="대화 기억 에이전트", page_icon="🤖")
    
    st.title("🤖 대화 기억 에이전트")
    st.markdown("과거 대화를 기억하고 맥락을 이해하는 AI 어시스턴트")
    
    # API 키 입력
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ""
    
    with st.sidebar:
        st.header("설정")
        api_key = st.text_input(
            "Google AI Studio API 키", 
            value=st.session_state.gemini_api_key,
            type="password",
            help="https://aistudio.google.com/app/apikey 에서 발급받으세요"
        )
        
        if api_key:
            st.session_state.gemini_api_key = api_key
        
        st.markdown("---")
        
        # 에이전트 초기화
        if st.session_state.gemini_api_key:
            if 'agent' not in st.session_state:
                st.session_state.agent = MemoryAgent(st.session_state.gemini_api_key)
            
            # 통계 표시
            if 'agent' in st.session_state:
                stats = st.session_state.agent.get_conversation_stats()
                st.metric("저장된 대화 쌍", stats['conversation_pairs'])
                st.metric("총 메시지 수", stats['total_stored_messages'])
    
    # 메인 채팅 인터페이스
    if not st.session_state.gemini_api_key:
        st.warning("Google AI Studio API 키를 입력해주세요.")
        st.info("🔗 API 키 발급: https://aistudio.google.com/app/apikey")
        return
    
    # 채팅 히스토리 초기화
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지 표시
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 어시스턴트 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("생각하는 중..."):
                response = st.session_state.agent.chat(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 기억 검색 기능 (디버깅용)
    with st.expander("🔍 기억 검색 테스트"):
        search_query = st.text_input("검색할 내용을 입력하세요:")
        if search_query and 'agent' in st.session_state:
            memories = st.session_state.agent.search_memory(search_query)
            for i, memory in enumerate(memories):
                st.write(f"**{i+1}. 유사도: {memory['similarity']:.3f}**")
                st.write(f"타입: {memory['metadata']['type']}")
                st.write(f"내용: {memory['content']}")
                st.write("---")

if __name__ == "__main__":
    main()