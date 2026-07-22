import requests

ZENODO_RECORD = "https://zenodo.org/api/records/4010759"

response = requests.get(ZENODO_RECORD, timeout=30)
response.raise_for_status()

record = response.json()
files = record["files"]

print(f"Total archives: {len(files)}\n")

for i, file in enumerate(files, start=1):
    name = file["key"]

    size_mb = file["size"] / (1024 * 1024)

    download_url = file["links"]["self"]

    print(f"{i:02d}. {name}")
    print(f"    Size : {size_mb:.2f} MB")
    print(f"    URL  : {download_url}\n")

    include50 = train[train["include_50"]]

print(include50["parent_label"].value_counts())