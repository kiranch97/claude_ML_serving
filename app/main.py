from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.tensorflow
import numpy as np
import time
import os

app = FastAPI(
    title="ML Model Serving API",
    description="FastAPI inference server — claude_ML_serving",
    version="1.0.0"
)

# MLflow config
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.mlflow.svc.cluster.local:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "SpamClassifier")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None
model_info = {}

def load_model():
    global model, model_info
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not versions:
            print(f"No model found for {MODEL_NAME} in stage {MODEL_STAGE}")
            return False
        mv = versions[0]
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.tensorflow.load_model(model_uri)
        model_info = {
            "name": MODEL_NAME,
            "version": mv.version,
            "stage": MODEL_STAGE,
            "run_id": mv.run_id
        }
        print(f"Model loaded: {MODEL_NAME} v{mv.version}")
        return True
    except Exception as e:
        print(f"Model load error: {e}")
        return False

# Load on startup
load_model()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    model_version: str
    inference_ms: float

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_info": model_info
    }

@app.get("/model-info")
def get_model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return model_info

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    start = time.time()
    # Tokenize and pad (simplified — adjust for your tokenizer)
    tokens = np.zeros((1, 100))  # placeholder
    proba = float(model.predict(tokens, verbose=0)[0][0])
    elapsed = (time.time() - start) * 1000
    label = "spam" if proba > 0.5 else "ham"
    return PredictResponse(
        prediction=label,
        confidence=round(proba if label == "spam" else 1 - proba, 4),
        probabilities={"ham": round(1 - proba, 4), "spam": round(proba, 4)},
        model_version=f"{MODEL_NAME}/{model_info.get('version', 'unknown')}",
        inference_ms=round(elapsed, 2)
    )

@app.post("/reload")
def reload_model():
    success = load_model()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reload model")
    return {"status": "reloaded", "model_info": model_info}
