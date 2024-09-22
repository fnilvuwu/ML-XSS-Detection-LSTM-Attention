import requests

url = "http://localhost:5000/predict"
data = {"text": "http://example.com/?id=<script>alert('xss')</script>"}
response = requests.post(url, json=data)
print(response.json())
