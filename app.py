from flask import Flask, request, jsonify,render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

app = Flask(__name__)
data = pd.read_csv("spam.csv", encoding='latin-1')

# just pick columns
data = data[['label', 'message']]

# remove nulls only
data = data.dropna()

# basic cleaning (NO over filtering)
data['label'] = data['label'].str.lower().str.strip()

# map labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# remove rows where mapping failed
data = data.dropna() 

emails = data['message']
labels = data['label']

print("Rows:", len(emails))
print("Sample:", emails.head(10).tolist())

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