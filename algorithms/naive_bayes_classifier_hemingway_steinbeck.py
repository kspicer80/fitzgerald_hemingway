import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import spacy
nlp = spacy.load('en_core_web_lg')
nlp.max_length = 1399867

def preprocess(text):
    doc = nlp(text)
    # Remove stop words and punctuation
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def load_just_hemingway_and_steinbeck(folder_path):
    root_folder = folder_path

    #filenames = []
    text_data = [] # Replace with the text data
    labels = [] # Replace with the corresponding labels (0 or 1)

    for subfolder in os.listdir(root_folder):
        if subfolder == 'fitzgerald':
            continue
        subfolder_path = os.path.join(root_folder, subfolder)

        if subfolder == 'hemingway':
            label = 1
        else:
            label = 0

        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            print("Processing file:", file)
            with open(file_path, 'r', encoding="utf-8") as f:
                text = f.read()
            #filenames.append(file_path)
            text_data.append(text)
            labels.append(label)
    return text_data, labels

def load_just_hemingway_and_steinbeck_v1(folder_path, subfolders):
    root_folder = folder_path

    text_data = []
    labels = []

    for subfolder in subfolders:
        subfolder_path = os.path.join(root_folder, subfolder)

        if subfolder == 'hemingway':
            label = 1
        else:
            label = 0

        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            print("Processing file:", file)
            with open(file_path, 'r', encoding="utf-8") as f:
                text = f.read()
            text_data.append(text)
            labels.append(label)
    return text_data, labels

# Load the data
subfolders = ['hemingway', 'steinbeck']
text_data, labels = load_just_hemingway_and_steinbeck_v1('data', subfolders)
df = pd.DataFrame(list(zip(text_data, labels)), columns=['text_data', 'label'])

X = df['text_data']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a bag-of-words representation of the text data
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

# Evaluate the model on the testing set
accuracy = nb.score(X_test_vec, y_test)
print(f"The accuracy score for this NaiveBayes Classifier is: {accuracy}")

# Predict the class labels for the test set and print out a classification report
y_pred = nb.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=['hemingway', 'steinbeck']))
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# create a list of class labels
classes = ['hemingway', 'steinbeck']

# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=classes, yticklabels=classes)

# add axis labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Define a new text sample to classify—Hemingway's "The Snows of Kilimanjaro"
with open(r'C:\Users\KSpicer\Documents\GitHub\fitzgerald_hemingway\test_data\steinbeck_in_dubious_battle.txt') as f:
    new_text =  f.read()

# Transform the new text sample into a bag-of-words representation
new_counts = vectorizer.transform([new_text])

# Use the trained model to predict the label of the new text sample
new_pred = nb.predict(new_counts)

# Print the predicted label
if new_pred == 1:
    print("The model predicts that this text is by Hemingway ...")
else:
    print("The model predicts that this text is by Steinbeck ...")

