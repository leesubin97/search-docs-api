import requests

# API URL과 헤더
url = "http://10.200.0.142:8000/chat"
headers = {
    "Content-Type": "application/json"
}

# 요청 본문
data = {
    "message": "하수도법에서의 공공하수처리시설에 대해 설명해보세요."
}

# POST 요청 보내기
response = requests.post(url, json=data, headers=headers)

# 응답 출력
if response.status_code == 200:
    print("Success!")
    print("Response JSON:", response.json())
else:
    print("Failed!")
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)
