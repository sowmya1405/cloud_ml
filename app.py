from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return '🌸 Iris Classifier API is running!'
