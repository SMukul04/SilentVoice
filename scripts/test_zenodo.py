import requests

url = "https://zenodo.org/api/records/4010759"

response = requests.get(url, timeout=30)

print("Status Code:", response.status_code)

data = response.json()

print("Record ID:", data["id"])

print("Number of downloadable files:", len(data["files"]))