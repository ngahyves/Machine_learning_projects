import traceback
import logging
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Configure logging for detailed debugging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)

# Load the saved SVM model
try:
    model = joblib.load("svm_best.pkl")
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error("Error loading the model: %s", str(e))
    model = None  # Prevent API from crashing if model fails to load

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Debugging: Print request headers & data
        logging.debug("Received request headers: %s", request.headers)
        data = request.get_json()
        logging.debug("Received JSON data: %s", data)

        # Validate JSON input
        if data is None or "features" not in data:
            logging.warning("Invalid JSON format or missing 'features'")
            return jsonify({"error": "Invalid JSON format or missing 'features'"}), 400

        # Convert features to NumPy array & reshape
        features = np.array(data["features"]).reshape(1, -1)
        logging.debug("Processed feature input: %s", features)

        # Ensure model is loaded before making predictions
        if model is None:
            logging.error("Model is not loaded! Cannot make predictions.")
            return jsonify({"error": "Model not found. Please reload the API."}), 500

        # Make predictions
        prediction = model.predict(features)[0]
        logging.info("Prediction generated successfully: %s", prediction)

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        logging.error("Error occurred during prediction: %s", str(e))
        logging.debug("Full Traceback: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Run the Flask application
if __name__ == "__main__":
    logging.info("Starting Flask API...")
    app.run(debug=True)
