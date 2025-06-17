# main.py

import joblib
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import logging

# Configure the logging
logging.basicConfig(filename='predictions.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# ...

# Dans votre fonction predict, juste avant le 'return'
@app.post("/predict")
def predict(transaction: Transaction):
    
    # Logging prediction
    log_message = f"INPUT: {transaction.dict()} | PREDICTION: {prediction_proba[0]:.4f} | DECISION: {'Fraude' if is_fraud else 'No-Fraud'}"
    logging.info(log_message)
    
    return {
        #The answer...
    }

# 1. Initiate FastAPI
app = FastAPI(title="API Fraud detection", version="1.0")

# 2. Load the model components
try:
    model = joblib.load('fraud_detection_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model and scaler load.")
except Exception as e:
    print(f"error loading model components : {e}")
    model = None
    scaler = None

# 3. Define data format
#    
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# 4. Endpoint of predictions
@app.post("/predict")
def predict(transaction: Transaction):
    """
    Receiving data
    """
    if not model or not scaler:
        return {"error": "Model or scaler not load."}

    # Convert entry data into DataFrame
    data = pd.DataFrame([transaction.dict()])
    
    # Scale the data
    data_scaled = scaler.transform(data)
    
    # Probability predictions
    prediction_proba = model.predict_proba(data_scaled)[:, 1]
    
    # thereshold
    thereshold = 0.5 
    
    # Formater la réponse
    is_fraud = bool(prediction_proba[0] > thereshold)
    
    return {
        "prediction": "Fraude" if is_fraud else "No-Fraud",
        "is_fraud": is_fraud,
        "scoring_fraud": float(prediction_proba[0])
    }

# Lauching the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
