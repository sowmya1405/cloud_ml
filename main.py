from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model/iris_model.joblib")

@app.route("/")
def index():
    return "🌸 Iris Classifier API is running!"

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
    app.run(debug=True)
