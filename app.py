from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load trained KNN model
model = joblib.load("knn_model.pkl")

@app.route('/')
def home():
    return "KNN Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']  # Expecting a list of features
        prediction = model.predict([np.array(data)])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
