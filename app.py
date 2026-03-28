from flask import Flask, request, jsonify, render_template
import pickle
 
app = Flask(__name__) 
 
# LOAD MODEL
model = pickle.load(open("model.pkl", "rb")) 
vectorizer = pickle.load(open("vectorizer.pkl", "rb")) 
 
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    # 🔥 SAME CLEANING (IMPORTANT)
    text = text.replace("Subject:", "")
    text = text.lower()

    vec = vectorizer.transform([text]) 
    result = model.predict(vec)[0] 

    return jsonify({"prediction": int(result)})

if __name__ == "__main__":
    app.run(debug=True)