import requests
try:
    response = requests.post("http://127.0.0.1:8000/api/v1/auth/signup", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "password123"
    })
    print(response.status_code)
    print(response.text)
except Exception as e:
    print(e)
