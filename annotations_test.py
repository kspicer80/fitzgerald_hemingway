import requests
from bs4 import BeautifulSoup
import json

url = "http://gutenberg.net.au/fsf/BABYLON-REVISITED.html"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
text = soup.get_text()

text_chunks = text.split("\n\n")

data = []

for i, chunk in enumerate(text_chunks):
    data.append("label": chunk_{}.format(i), "text": chunk)

with open("babylon.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")