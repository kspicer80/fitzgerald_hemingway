import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import transformers
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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
            #file_for_end = file_path.split('.')[0]
            #filenames.append(file_for_end)
            text_data.append(text)
            labels.append(label)
    return text_data, labels

text_data, labels = load_data('data')
metaphor_counter = MetaphorCounter()
classifier = SVC(kernel='linear', max_iter=1000)

#classifier = LinearSVC()

X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)

pipe1 = sklearn.pipeline.Pipeline([
    ('metaphor_counter', metaphor_counter),
    ('classifier', classifier)
])

pipe1.fit(X_train, y_train)

y_pred = pipe1.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for this LinearSVC model utilizing TfidfVectorizer is:", accuracy)
#print(classification_report(y_test, y_pred))#, target_names=target_names))
print(classifier.dual_coef_)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Set1')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
plt.savefig('svc_with_metaphor_counts.png')

