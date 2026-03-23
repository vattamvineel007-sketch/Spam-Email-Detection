from flask import Flask, request, jsonify,render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

app = Flask(__name__)
data = pd.read_csv("spam.csv", encoding='latin-1')

# remove extra columns if exist
data = data.iloc[:, :2]
data.columns = ['label', 'message']

# clean
data = data.dropna()

data['label'] = data['label'].astype(str).str.strip().str.lower()
data['message'] = data['message'].astype(str).str.strip()

# remove empty messages
data = data[data['message'] != ""]

# keep only valid labels
data = data[data['label'].isin(['ham', 'spam'])]

# convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# FINAL check
data = data.dropna()

print("Final rows:", len(data))

emails = data['message']
labels = data['label']

vectorizer = TfidfVectorizer()
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))