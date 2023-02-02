# Import the required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the data
text_data = [...] # Replace with the text data
labels = [...] # Replace with the corresponding labels (0 or 1)

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

# Evaluate the classifier
accuracy = accuracy_score(labels_test, predictions)
print("Accuracy:", accuracy)
