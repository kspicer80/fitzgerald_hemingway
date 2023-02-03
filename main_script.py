# Import the required libraries
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import spacy
nlp = spacy.load('en_core_web_lg')

def count_metaphors_similes(text):
    doc = nlp(text)
    metaphor_simile_count = 0
    for token in doc:
        if token.pos_ in ['ADJ', 'ADV']:
            if "compound" in [child.dep_ for child in token.children]:
                metaphor_simile_count += 1
    return metaphor_simile_count

# Load the data
root_folder = 'data'

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

        text_data.append(text)
        labels.append(label)

# Let's try a different vectorizer:

X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2)

metaphor_count = [count_metaphors_similes(text) for text in X_train]
metaphor_count = np.array(metaphor_count).reshape(-1, 1)

X_traied
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

classifier = LinearSVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for this LinearSVC model utilizing TfidfVectorizer is:", accuracy)

'''
# Convert the text data into numerical representations
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

X_test = [str(text).lower() for text in X_test]
X_test = vectorizer.transform(X_test)

#X_test = vectorizer.transform(X_test)

# Train a binary classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict the labels on the test data
predictions = model.predict(X_test)
predictions = [1 if prediction > 0.5 else 0 for prediction in predictions]

# Evaluate the classifier

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


for i in range(len(X_test)):
    print("Text:", X_test[i][:100], "...")
    print("Actual label:", labels_test[i])
    print("Predicted label:", predictions[i])


# Convert the predictions from numeric to binary format
predictions = [1 if prediction > 0.5 else 0 for prediction in predictions]

# Calculate the accuracy of the classifier
accuracy = accuracy_score(labels, predictions)
print("Accuracy for this model utilizing the CountVectorizer is:", accuracy)
'''