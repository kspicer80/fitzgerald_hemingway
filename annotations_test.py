import requests
from bs4 import BeautifulSoup
import json
import nltk
import spacy
nlp = spacy.load('en_core_web_lg')

'''
url = "http://gutenberg.net.au/fsf/BABYLON-REVISITED.html"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
text = soup.get_text()
# Let's try NLTK's sentence tokenizer
#sentences = nltk.sent_tokenize(text)
#data = []
#for i, sentence in enumerate(sentences):
    #clean_sentence = sentence.replace("\\", "")
    #data.append({"label": "", "sentence_number": i, "text": clean_sentence})

#with open("fitzgerald_babylon_annotated.jsonl", "w") as f:
    #for item in data:
        #f.write(json.dumps(item) + "\n")


# That one doesn't work fantastically—reads Mr. Campbell and splits in there on the period ... let's try spaCy instead
with open(r'data\fitzgerald\bablyon_revisited.txt', 'r') as f:
    text = f.read()

doc = nlp(text)

data = []
for i, sentence in enumerate(doc.sents, 1):
    # Check if the sentence ends with a comma followed by a lowercase letter
    if sentence[-2].text == "," and sentence[-1].text[0].islower():
        # Merge the current sentence with the next sentence
        next_sentence = next(doc.sents)
        new_sentence = sentence[:-1] + next_sentence
        doc.merge(new_sentence)
        clean_sentence = str(new_sentence).replace("\n", "")
        data.append({"label": "", "sentence_number": i, "text": clean_sentence})
    elif sentence[-1].text == "?":
        # Check if the next sentence starts with a lowercase letter
        next_sentence = next(doc.sents)
        if next_sentence[0].text[0].islower():
            # Merge the current sentence with the next sentence
            new_sentence = sentence + next_sentence
            doc.merge(new_sentence)
            clean_sentence = str(new_sentence).replace("\n\n", "")
            data.append({"label": "", "sentence_number": i, "text": clean_sentence})
        else:
            clean_sentence = str(sentence).replace("\n\n", "")
            data.append({"label": "", "sentence_number": i, "text": clean_sentence})
    else:
        clean_sentence = str(sentence).replace("\n\n", "")
        data.append({"label": "", "sentence_number": i, "text": clean_sentence})

print(len(data))

with open("fitzgerald_babylon_annotated.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
'''

filename = "fitzgerald_babylon_annotated_v0.jsonl"
new_filename = "fitzgerald_babylon_annotated_v1.jsonl"

with open(filename, "r") as f:
    with open(new_filename, "w") as f_new:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            data["sentence_number"] = i + 1
            f_new.write(json.dumps(data) + "\n")
