import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# load dataset
data = pd.read_csv("emails.csv", encoding='latin-1')

# keep only 2 columns
data = data.iloc[:, :2]
data.columns = ['text', 'spam']

# remove nulls
data = data.dropna()

# clean labels
data['spam'] = data['spam'].astype(int)

# features & labels
emails = data['text']
labels = data['spam']

print("Rows:", len(emails))
print(labels.value_counts())

# vectorize
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(emails)

# model
model = MultinomialNB()
model.fit(X, labels)

# save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model & Vectorizer saved ✅")