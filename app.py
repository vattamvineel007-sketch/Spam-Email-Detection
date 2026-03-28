from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)
print("NEW CODE RUNNING")

# load trained files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get("text")

    print("INPUT TEXT:", text)

    # 🔥 SAME CLEANING AS TRAINING
    if not text or not text.strip():
        return jsonify({"prediction": "Enter valid text"})

    text = text.lower().replace("subject:", "")

    vec = vectorizer.transform([text])

    result = model.predict(vec)[0]
    print("MODEL OUTPUT:", result)

    if result == 1:
        output = "Spam"
    else:
        output = "Not Spam"

    return jsonify({"prediction": output})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)