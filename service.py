import os
import fitz  # PyMuPDF for PDF text extraction
import faiss
import numpy as np
import requests
from typing import List
import nltk
from pykospacing import Spacing
import re

# Spacing을 위한 초기화
spacing = Spacing()

# NLTK를 위한 다운로드
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

# PDF 텍스트 추출 함수
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return spacing(text)

# 텍스트 청킹 함수
def chunk_text(text, chunk_size=400, overlap=0):
    # 문장 단위로 텍스트 분할
    sentences = re.split(r'(?<!\d)(?<![가-힣])\.(?=\s)|(?<=\n)|(?<=!)|(?<=])|(?<=>)|(?=-\s*\d+\s*-)|(?=-\s*\d+\s*/\s*\d+\s*-)|(?<=\.)\s+(?=[○])|(?=제\d+조\([^)]+\))', text)
    
    # 특정 패턴을 공백으로 대체
    sentences = [re.sub(r'-\s*\d+\s*/\s*\d+\s*-|-\s*\d+\s*-|법제처|국가법령정보센터|공익사업을 위한 토지 등의 취득 및 보상에 관..', '', sentence) for sentence in sentences]

    chunks = []
    current_chunk = ""
    current_pattern = ""

    for sentence in sentences:
        pattern_match = re.match(r'제\d+조\([^)]+\)', sentence)
        if pattern_match:
            current_pattern = pattern_match.group()

        while len(sentence) > chunk_size:
            part = sentence[:chunk_size]
            if current_chunk:
                chunks.append((current_chunk.strip(), current_pattern))
            current_chunk = part
            sentence = sentence[chunk_size:]

        if len(current_chunk) + len(sentence) > chunk_size:
            chunks.append((current_chunk.strip(), current_pattern))
            current_chunk = sentence
        else:
            current_chunk += sentence

    if current_chunk:
        chunks.append((current_chunk.strip(), current_pattern))

    # 오버랩을 고려하여 청크 조정
    final_chunks = []
    for i in range(len(chunks)):
        if i == 0:
            final_chunks.append(chunks[i])
        else:
            start_overlap = chunks[i-1][0][-overlap:]
            final_chunk = start_overlap + chunks[i][0]
            final_chunks.append((final_chunk, chunks[i][1]))

    return final_chunks

class EmbeddingService:
    def __init__(self):
        self.api_url = "http://10.200.0.12:19066/query_embeddings"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        def chunk_list(lst, chunk_size):
            """Helper function to split list into chunks."""
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]
        
        all_embeddings = []
        chunk_size = 10
        
        print(f"Embedding {len(texts)} texts via API...", flush=True)
        for chunk in chunk_list(texts, chunk_size):
            print(f"Embedding {len(chunk)} texts via API...", flush=True)
            headers = {'Content-Type': 'application/json'}
            data = {
                "sentences": chunk,
                "return_dense": True,
                "return_sparse": False,
                "return_multi": False
            }
            response = requests.post(self.api_url, headers=headers, json=data)
            if response.status_code == 200:
                dense_vecs = response.json()["response_details"]["dense_vecs"]
                embeddings = np.array([vec["dense_vecs_values"] for vec in dense_vecs])
                all_embeddings.append(embeddings)
                print(f"Generated embeddings via API: {embeddings.shape}", flush=True)
            else:
                response.raise_for_status()

        # Concatenate all the embeddings into a single numpy array
        final_embeddings = np.vstack(all_embeddings)
        print(f"Final embeddings shape: {final_embeddings.shape}", flush=True)
        return final_embeddings

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class IndexService(metaclass=SingletonMeta):
    def __init__(self):
        self.index_file_path = 'faiss_index.index'
        self.meta_data_file_path = 'meta_data.npy'
        self.index = None
        self.meta_data = []
        self.load_existing_index()

    def load_existing_index(self):
        if os.path.exists(self.index_file_path):
            self.index = faiss.read_index(self.index_file_path)
            print(f"Index loaded from {self.index_file_path}")
        else:
            print("No existing index found. Starting with an empty index.")

        if os.path.exists(self.meta_data_file_path):
            self.meta_data = np.load(self.meta_data_file_path, allow_pickle=True).tolist()
            print(f"Meta data loaded. Size: {len(self.meta_data)}")
        else:
            print("No existing meta data found. Starting with an empty meta data list.")

    def save_index_and_meta_data(self):
        # 색인 및 메타데이터 저장
        faiss.write_index(self.index, self.index_file_path)
        np.save(self.meta_data_file_path, self.meta_data)
        print(f"Index and meta data saved to {self.index_file_path} and {self.meta_data_file_path}")

    def add_text_to_index(self, text, source, chapter):
        embedding_service = EmbeddingService()
        
        if any(item['content'] == text for item in self.meta_data):
            print("Text already indexed, skipping.")
            return
        
        # 텍스트 임베딩
        embedding = embedding_service.embed_texts([text])[0]
        
        # 임베딩 차원 가져오기
        dimension = embedding.shape[0]
        
        # 색인 초기화 (필요한 경우)
        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)
        
        # 텍스트를 색인에 추가
        self.index.add(np.array([embedding], dtype=np.float32))
        
        # 메타데이터 추가
        new_meta_data = {
            "content": text,
            "source": source,
            "chapter": chapter
        }
        self.meta_data.append(new_meta_data)
        
        # 색인 및 메타데이터 저장
        self.save_index_and_meta_data()
        print("Text indexed and saved.")

    def get_index_size(self):
        if self.index is not None:
            return self.index.ntotal
        return 0

    def search(self, query: str, k: int = 5):
        embedding_service = EmbeddingService()
        query_embedding = embedding_service.embed_texts([query])
        print(f"Query embedding shape: {query_embedding.shape}")

        if self.index is None:
            raise FileNotFoundError(f"Index file {self.index_file_path} does not exist.")
        
        index_size = self.get_index_size()
        print(f"Index size: {index_size}")
        print(f"Meta data size: {len(self.meta_data)}")

        assert len(self.meta_data) == index_size, f"Mismatch: {len(self.meta_data)} meta data items, but {index_size} in index."

        distances, indices = self.index.search(query_embedding, k)
        print(f"Distances: {distances}, Indices: {indices}")

        results = []
        num = 0
        seen = set()
        for i in indices[0]:
            if i in seen:
                continue
            seen.add(i)
            print(f"Processing index: {i}")
            if i < index_size:
                content = self.meta_data[i]['content']
                corrected_content = content  # 맞춤법 보정 적용
                result = {
                    'content': corrected_content.lstrip(),
                    'document': self.meta_data[i]['source'],
                    'distance': float(distances[0][num])  # float 타입으로 변환
                }
                print(f"Appending result: {result}")  # 디버깅 메시지 추가
                results.append(result)
                num += 1
            else:
                print(f"Index {i} out of range for meta_data of size {len(self.meta_data)}")

        print('결과 조회')
        print(f"Results: {results}")

        return results[:k]

# 텍스트 추가 함수 정의
def add_text_to_index(text, source, chapter):
    index_service = IndexService()
    index_service.add_text_to_index(text, source, chapter)

# 추가할 텍스트 데이터
text_data = """
[별표 1] 부동산 신규취득 제한부서와 그 직무 관련 부동산의 범위
제한부서 : 전 부서
제한 부동산의 범위 : 「한국토지주택공사법」 제8조에 따라 공사가
사업 수행을 위해 취득 예정인 토지 및 건물,
공사가 분양‧공급하는 토지 및 건물
취득제한기간(시기) : 공사의 사업 참여 사실이 대외적으로 공표된 날
취득제한기간(종기) : 수분양자 등에게 소유권을 이전한 날
1) 본 지침 시행일 이전 매매계약(우선 분양전환 권리가 있는 주택임대차계약을 포함한다)에
따라 시행일 이후 취득하게 되는 부동산은 신규 취득 제한 부동산에서 제외한다.
2) “공사가 분양‧공급하는 토지 및 건물” 중 공사가 공사 홈페이지, 일간신문 또는 관할 시‧군
‧자치구의 홈페이지를 통해 추첨제 분양 또는 추첨에 의한 동호지정순번을 결정하여 그 결
과에 따라 계약하여 취득하는 부동산은 제외한다.
3) “공사의 사업 참여 사실이 대외적으로 공표된 날”이란 사업이 주민공고‧공람, 지구지정 등
으로 고시되거나 일간신문, 홈페이지 등에 안내되어 대외에 최초로 공개된 날을 말한다. 다
만, 대외적으로 공표되기 전이라도 제한대상자 등이 직무 관련 부동산임을 알게 된 경우 해
당 부동산은 취득제한에 해당하며, 예외적 취득 시 제한대상자는 제6조에 따라 신고하여야
한다.
"""

# 텍스트 색인에 추가
add_text_to_index(text_data, "부동산 신규취득 제한 및 신고에 관한 지침", "별표 1")

