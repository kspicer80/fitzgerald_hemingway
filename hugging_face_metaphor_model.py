from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import os
import json

'''
def process_text_with_hugging_face(texts: list):
    metaphor_counts = {}
    for i, text in enumerate(texts):
        label_list = ['literal', 'metaphoric']
        label_dict_relations = {i: l for i, l in enumerate(label_list)}
        model = AutoModelForTokenClassification.from_pretrained("lwachowiak/Metaphor-Detection-XLMR")
        tokenizer = AutoTokenizer.from_pretrained("lwachowiak/Metaphor-Detection-XLMR")
        metaphor_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        processed_text = metaphor_pipeline(text)
        count = count_label_1(processed_text)
        metaphor_counts[i] = count
    return metaphor_counts
'''

def process_text_with_hugging_face(texts, filenames):
    metaphor_counts = {}
    for i, text in enumerate(texts):
        label_list = ['literal', 'metaphoric']
        label_dict_relations = {i: l for i, l in enumerate(label_list)}
        model = AutoModelForTokenClassification.from_pretrained("lwachowiak/Metaphor-Detection-XLMR")
        tokenizer = AutoTokenizer.from_pretrained("lwachowiak/Metaphor-Detection-XLMR")
        metaphor_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        processed_text = metaphor_pipeline(text)
        count = count_label_1(processed_text)
        file_name = filenames[i]
        metaphor_counts[file_name] = count
    return metaphor_counts

def count_label_1(entities):
    count = 0
    for entity in entities:
        if entity['entity_group'] == 'LABEL_1':
            count += 1
    return count

def load_data(folder_path):
    root_folder = folder_path

    filenames = []
    text_data = []

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
            file_for_end = file_path.split('.')[0]
            filenames.append(file_for_end)
            text_data.append(text)

    return text_data, filenames

loaded_data, filenames = load_data('data')
full_metaphor_counts = process_text_with_hugging_face(loaded_data, filenames)

with open('results.json', 'w') as outfile:
    json.dump(full_metaphor_counts, outfile)

