from pydantic import BaseModel

# 요청 모델
class Item(BaseModel):
    # API 파라미터 지정
    prompt: str

# 응답 모델
class ItemResponse(BaseModel):
    id: int
    name: str
    description: str
    total_price: float

# open ai 모델
class Message(BaseModel):
    message: str

class FilePathModel(BaseModel):
    path: str

class QueryModel(BaseModel):
    query: str
    k: int = 5