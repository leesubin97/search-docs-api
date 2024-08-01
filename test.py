import os
import fitz  # PyMuPDF for PDF text extraction
import json
import re
from typing import List
import nltk
from pykospacing import Spacing

# 맞춤법 교정을 위해 Spacing 모듈 사용
spacing = Spacing()

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
def chunk_text(text, chunk_size=400, overlap=100):
    # 문장 단위로 텍스트 분할
    sentences = re.split(r'(?<=\.)\s+|(?<=\n)', text)
    
    chunks = []
    current_chunk = ""
    current_pattern = ""

    for sentence in sentences:
        pattern_match = re.match(r'제\d+조', sentence)
        if pattern_match:
            if current_chunk:
                chunks.append((current_chunk.strip(), current_pattern))
            current_pattern = pattern_match.group()
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
        
        while len(current_chunk) > chunk_size:
            part = current_chunk[:chunk_size]
            chunks.append((part.strip(), current_pattern))
            current_chunk = current_chunk[chunk_size - overlap:].strip()

    if current_chunk:
        chunks.append((current_chunk.strip(), current_pattern))

    return chunks

# JSON으로 저장하는 함수
def save_clauses_to_json(clauses, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clauses, f, ensure_ascii=False, indent=4)

# PDF 파일 경로
file_path = 'doc/산업안전보건법 시행령(대통령령)(제34304호)(20240312).pdf'
# JSON 출력 파일 경로
output_path = '/mnt/data/clauses.json'

# PDF에서 텍스트 추출
pdf_text = extract_text_from_pdf(file_path)

# 조항 추출 및 청킹
chunks = chunk_text(pdf_text)

# JSON으로 저장할 데이터 구조로 변환
clauses = [{"title": chunk[1], "content": chunk[0]} for chunk in chunks]

# JSON으로 저장
save_clauses_to_json(clauses, output_path)

print(f"Extracted clauses saved to {output_path}")
