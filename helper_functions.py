import numpy
import os
from statistics import mean
import spacy
nlp = spacy.load('en_core_web_lg')
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import transformers

class MetaphorCounter(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('\n>>>>>>>>>>>init() called.\n')

    def fit(self, X, y=None):
        print('\n>>>>>>>>>>>fit() called.\n')
        return self

    def process_text_with_hugging_face(self, texts: list):
        metaphor_counts = []
        for text in texts:
            label_list = ['literal', 'metaphoric']
            label_dict_relations = {i: l for i, l in enumerate(label_list)}

            model = AutoModelForTokenClassification.from_pretrained("lwachowiak/Metaphor-Detection-XLMR")
            tokenizer = AutoTokenizer.from_pretrained("lwachowiak/Metaphor-Detection-XLMR")
            metaphor_pipeline_ = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
            processed_text = metaphor_pipeline_(text)
            count = self.count_label_1(processed_text)
            metaphor_counts.append(count)
        return metaphor_counts

    def count_label_1(self, entities):
        count = 0
        for entity in entities:
            if entity['entity_group'] == 'LABEL_1':
                count += 1
        return count

    def transform(self, X, y=None):
        print('\n>>>>>>>>>>>transform(*) called.\n')
        metaphor_counts = self.process_text_with_hugging_face(X)
        return np.array(metaphor_counts).reshape(-1, 1)

def load_data(folder_path):
    root_folder = folder_path

    #filenames = []
    text_data = [] # Replace with the text data
    labels = [] # Replace with the corresponding labels (0 or 1)

    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        if subfolder == 'fitzgerald':
            label = 0
        else:
            label = 1

        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            print("Processing file:", file)
            with open(file_path, 'r', encoding="utf-8") as f:
                text = f.read()
            #filenames.append(file_path)
            text_data.append(text)
            labels.append(label)
    return text_data, labels

def calculate_word_stats(texts, filenames):
    word_stats = {}
    for text, filename in zip(texts, filenames):
        doc = nlp(text)
        sentences = [sent for sent in doc.sents]
        num_words = [len([token for token in sent if not (token.is_punct | token.is_space)]) for sent in doc.sents]
        avg_sentence_length = sum([len(sent) for sent in sentences])/len(sentences)
        word_stats[filename] = (len(sentences), avg_sentence_length)
    return word_stats

text_data, filenames = load_data('data')
word_stats_dict = calculate_word_stats(text_data, filenames)

import json

with open('word_stats', 'w') as f:
    json.dump(word_stats_dict, f)