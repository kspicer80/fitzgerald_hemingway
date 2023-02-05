# Import the required libraries
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the data
root_folder = 'data'

filenames = []
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
        file_for_end = file_path.split('.')[0]
        filenames.append(file_for_end)
        text_data.append(text)
        labels.append(label)

print(filenames)
vectorizer = TfidfVectorizer()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

classifier = LinearSVC()
classifier.fit(X_train, y_train)
training_predictions = classifier.predict(X_train)
y_train_pred = classifier.predict(X_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for this LinearSVC model utilizing TfidfVectorizer is:", accuracy)

train_results = dict(zip(filenames[:X_train.shape[0]], y_train_pred))
#results = dict(zip(filenames[len(X_train):], y_pred))
for filename, prediction in train_results.items():
    print(f"File: {filename}, Prediction: {prediction}")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Set1')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
plt.savefig('base_svc_model.png')
'''
# Here's a simple NaiveBayes classifier:

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