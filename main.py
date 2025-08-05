from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model/iris_model.joblib")

@app.route("/")
def index():
    return " Iris Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
model = joblib.load("model/iris_model.joblib")

@app.route("/")
def index():
    return " Iris Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("data", [])
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    try:
        preds = model.predict(np.array(data)).tolist()
        return jsonify({"predictions": preds})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

    data = request.json.get("data", [])
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    try:
        preds = model.predict(np.array(data)).tolist()
        return jsonify({"predictions": preds})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
