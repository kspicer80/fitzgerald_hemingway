from helper_functions import load_data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import spacy
nlp = spacy.load('en_core_web_lg')

def preprocess(text):
    doc = nlp(text)
    # Remove stop words and punctuation
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Load the data
text_data, labels = load_data('data')
df = pd.DataFrame(list(zip(text_data, labels)), columns=['text_data', 'label'])

X = df['text_data']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Linear SVM model
sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)

# Evaluate the model on the testing set
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=[0, 1]))

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# create a list of class labels
classes = ['fitzgerald', 'hemingway']

# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=classes, yticklabels=classes)

# add axis labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

'''
# Define a new text sample to classify—Hemingway's "The Snows of Kilimanjaro"
with open(r'test_data\hemingway_snows.txt') as f:
    new_text =  f.read()

# Transform the new text sample into a bag-of-words representation
new_counts = CountVectorizer.transform([new_text])

# Use the trained model to predict the label of the new text sample
new_pred = sgd.predict(new_text)

# Print the predicted label
if new_pred == 1:
    print("The model predicts that this text is by Hemingway ...")
else:
    print("The model predicts that this text is by Fitzgerald ...")
'''

