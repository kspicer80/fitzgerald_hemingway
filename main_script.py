# Import the required libraries
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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
        with open(file_path, 'r') as f:
            text = f.read()

        text_data.append(text)
        labels.append(label)

#print(len(text_data))
#print(len(labels))

# Split the data into training and testing sets
text_train, text_test, labels_train, labels_test = train_test_split(text_data, labels, test_size=0.2)

# Convert the text data into numerical representations
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(text_train)
X_test = vectorizer.transform(text_test)

# Train a binary classifier
classifier = MultinomialNB()
classifier.fit(X_train, labels_train)

# Predict the labels on the test data
predictions = classifier.predict(X_test)
predictions = [1 if prediction > 0.5 else 0 for prediction in predictions]
accuracy = accuracy_score(labels_test, predictions)

# Evaluate the classifier
accuracy = accuracy_score(labels_test, predictions)
print("Accuracy:", accuracy)

for i in range(len(X_test)):
    print("Text:", X_test[i][:100], "...")
    print("Actual label:", labels_test[i])
    print("Predicted label:", predictions[i])

# Convert the sparse matrix to a dense matrix
X_test = X_test.toarray()

# Make predictions on the test data
predictions = model.predict(X_test)

# Convert the predictions from numeric to binary format
predictions = [1 if prediction > 0.5 else 0 for prediction in predictions]

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
