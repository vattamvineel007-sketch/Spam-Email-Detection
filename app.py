from flask import Flask, request, jsonify,render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

app = Flask(__name__)
data = pd.read_csv("spam.csv", encoding='latin-1', on_bad_lines='skip')

data = data[['label', 'message']]

data = data.dropna()

data['label'] = data['label'].astype(str)
data['label'] = data['label'].str.replace('"', '')
data['label'] = data['label'].str.strip().str.lower()

print("Unique labels BEFORE map:", data['label'].unique())

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

print("After map:", data['label'].unique())

data = data.dropna()

emails = data['message']
labels = data['label']

print("Rows:", len(emails))
print("Sample:", emails.head(5).tolist())

if len(emails) == 0:
    raise ValueError("Dataset is EMPTY")

vectorizer = TfidfVectorizer(stop_words=None)
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    return jsonify({"prediction": int(result)})

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)