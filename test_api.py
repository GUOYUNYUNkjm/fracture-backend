import requests

url = "https://parking-dissuade-revenue.ngrok-free.dev/predict"

files = {
    "file": open("test.jpg", "rb")
}

res = requests.post(url, files=files)

print(res.text)