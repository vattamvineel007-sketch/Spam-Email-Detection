import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# load dataset
data = pd.read_csv("emails.csv", encoding="latin-1")

# keep only needed columns
data = data[['text', 'spam']]

# remove nulls
data = data.dropna()

# 🔥 CLEAN TEXT (IMPORTANT)
data['text'] = data['text'].str.replace("Subject:", "", regex=False)
data['text'] = data['text'].str.lower()

# features & labels
emails = data['text']
labels = data['spam']

print("Rows:", len(emails))
print(labels.value_counts())

# 🔥 BETTER VECTORIZER
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9
)

X = vectorizer.fit_transform(emails)

# 🔥 MODEL (STRONGER THAN DEFAULT)
model = MultinomialNB(alpha=0.5)

model.fit(X, labels)

# 🔥 TEST BEFORE SAVING
test_samples = [
    "win money now click here",
    "free prize claim now",
    "meeting schedule tomorrow",
    "project discussion"
]

test_vec = vectorizer.transform(test_samples)
print("Test Predictions:", model.predict(test_vec))

# save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model & vectorizer saved ✅")