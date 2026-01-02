import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load trained model
try:
    model = joblib.load("logistic_regression_model.joblib")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


@app.route("/")
def home():
    return "✅ Diabetes Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Convert input JSON to DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]

        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return jsonify({
            "prediction": int(prediction),
            "result": result
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
