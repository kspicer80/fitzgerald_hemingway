import requests
from bs4 import BeautifulSoup

# Define the URL of the main page
url = "https://gutenberg.org/ebooks/author/420"

# Send a request to the website
response = requests.get(url)

# Parse the HTML content of the page
soup = BeautifulSoup(response.content, "html.parser")

# Find all the links to the texts
text_links = [link["href"] for link in soup.find_all("a") if link.get("href").startswith("/ebooks/")]

# Download and print the texts
for link in text_links:
    text_url = "https://gutenberg.org" + link
    response = requests.get(text_url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()
    print(text)

fitzgerald_text_links = ['https://gutenberg.org/cache/epub/64317/pg64317.txt',
    'https://gutenberg.org/files/9830/9830-0.txt',
    'https://gutenberg.org/cache/epub/4368/pg4368.txt',
    'https://gutenberg.org/files/60962/60962-0.txt',
   ' https://gutenberg.org/files/805/805-0.txt',
    'https://gutenberg.org/files/6695/6695-0.txt',
    'https://gutenberg.org/cache/epub/68229/pg68229.txt'
    ]

hemingway_text_links = ['https://gutenberg.org/cache/epub/67138/pg67138.txt',
    'https://gutenberg.org/files/61085/61085-0.txt',
    'https://gutenberg.org/files/59603/59603-0.txt',
    'https://gutenberg.org/ebooks/69683']

for link in fitzgerald_text_links:
    response = requests.get(link)
    with open(link.split("/")[-1], "wb") as file:
        file.write(response.content)

for link in hemingway_text_links:
    response = requests.get(link)
    with open(link.split("/")[-1], "wb") as file:
        file.write(response.content)