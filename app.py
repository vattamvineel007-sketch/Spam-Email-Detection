from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import os

app = Flask(__name__)

# ===== LOAD DATA =====
data = pd.read_csv("clean_spam.csv", encoding='latin-1')

# keep only required columns
data = data.iloc[:, :2]
data.columns = ['label', 'message']

# remove nulls
data = data.dropna()

# clean labels
data['label'] = data['label'].astype(str).str.strip().str.lower()

# keep only valid labels
data = data[data['label'].isin(['ham', 'spam'])]

# convert labels → numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# final clean
data = data.dropna()

emails = data['message']
labels = data['label']

print("Rows:", len(emails))
print("Labels:", labels.value_counts())

# ===== MODEL =====
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

# ===== ROUTES =====
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]

    if result == 1:
        output = "Spam"
    else:
        output = "Not Spam"

    return jsonify({"prediction": output})

# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)