import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# LOAD DATASET
data = pd.read_csv("emails.csv")

# KEEP ONLY REQUIRED COLUMNS
data = data[['text', 'spam']]
data = data.dropna()

# 🔥 CLEAN TEXT (VERY IMPORTANT)
data['text'] = data['text'].str.replace("Subject:", "", regex=False)
data['text'] = data['text'].str.lower()

# SPLIT
X = data['text']
y = data['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# VECTORIZE
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# MODEL
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# SAVE FILES
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model & Vectorizer saved ✅")