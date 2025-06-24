from flask import Flask, request, jsonify
import joblib
import numpy as np

# Charger le modèle SVM sauvegardé
model = joblib.load("svm_best.pkl")

# Initialiser l’API Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Récupérer les données envoyées
    features = np.array(data["features"]).reshape(1, -1)  # Formater les données

    prediction = model.predict(features)[0]  # Prédiction
    probability = model.predict_proba(features)[0].tolist()  # Probabilités

    return jsonify({"prediction": int(prediction), "probabilities": probability})

if __name__ == "__main__":
    app.run(debug=True)
