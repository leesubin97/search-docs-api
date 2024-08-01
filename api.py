from fastapi import FastAPI, HTTPException, Depends
from models import Item, Message, FilePathModel, QueryModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from service import IndexService
import os
import openai
from dotenv import load_dotenv

import requests
import json


load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# 종속성 주입 함수
def get_index_service():
    return IndexService()

@app.post("/chat")
async def chat_message(message: Message, index_service: IndexService = Depends(get_index_service)):
    try:
        # 검색 수행
        query_model = QueryModel(query=message.message, k=5)
        results = index_service.search(query_model.query, query_model.k)

        print('채팅 검색')
        print(results)

        # 검색 결과 텍스트 형식으로 변환
        results_text = "\n".join(
            [
                f"[문서{index + 1}]\n{result['content']}\n" if index < len(results) - 1 else f"[문서{index + 1}]\n{result['content']}"
                for index, result in enumerate(results)
            ]
        )

        print('텍스트형식변환')
        print(results_text)
        print(query_model.query)
        
        # 세션 할당
        session_request_body = {
            "user_id": "lh-demo"
        }

        response = requests.post("http://211.109.9.151:15003/session/open", json=session_request_body, stream=True)

        api_key = response.json().get('api_key')

        print(f"세선할당 : {api_key}")

        # 요청 본문 생성
        request_body = {
            "user_name": "사용자",
            "agent_name": "루시아",
            "api_key": api_key,
            "prompt": query_model.query,
            "stream": True,
            "overflow": False,
            "document": results_text,
            "use_history": False,
            "history": [],
            "history_count": 5,
            "best_of": 1,
            "max_new_tokens": 512,
            "repetition_penalty": 1.1,
            "temperature": 0.1,
            "top_p": 0.95,
            "do_mapreduce": False,
            "chunk_size": 1000,
            "doc_max_token": 1500,
            "max_token_length": 1500,
            "stop": [],
            "stop_token_ids": []
        }

        # HTTP 요청
        response = requests.post("http://211.109.9.151:15003/completion", json=request_body, stream=True)

        # 스트리밍 응답 처리
        generated_text = ""
        generated_texts = []
        for line in response.iter_lines():
            if line:
                # 데이터 추출
                data = line.decode('utf-8').replace("data:", "")
                json_data = json.loads(data)
                if 'token' in json_data and 'text' in json_data['token']:
                    generated_text += json_data['token']['text']
                    generated_texts.append(json_data['token']['text'])

        # distance가 0.3 이하인 결과가 있는지 확인
        distance_threshold = 0.3
        has_low_distance = any(result['distance'] <= distance_threshold for result in results)

        if has_low_distance:
            return JSONResponse(content={"response": generated_texts, "full_response" : generated_text})
        else:
            return JSONResponse(content={"response": generated_texts, "results": results, "full_response" : generated_text})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/upload")
async def upload_file(file_path: FilePathModel, index_service: IndexService = Depends(get_index_service)):
    try:
        index_service.create_index(file_path.path)
        return {"message": "Index created and saved successfully"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(query_model: QueryModel, index_service: IndexService = Depends(get_index_service)):
    try:
        results = index_service.search(query_model.query, query_model.k)
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
