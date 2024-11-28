from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
import joblib
import os
from src.api.config import Config

# Initialize API Router
router = APIRouter()

# Path to the trained model
MODEL_PATH = Config.MODEL_PATH

# Load the model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


# Request and Response schemas
class InferenceRequest(BaseModel):
    features: List[float]


class InferenceResponse(BaseModel):
    prediction: int
    probabilities: Dict[str, float]


@router.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    """
    Perform inference using the trained model.
    """
    if not model:
        return {"error": "Model not found. Train the model first."}

    # Reshape input for prediction
    features = [request.features]
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # Format probabilities for output
    prob_dict = {f"class_{i}": prob for i, prob in enumerate(probabilities)}
    return {"prediction": prediction, "probabilities": prob_dict}


@router.get("/health")
def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {"status": "healthy", "model_loaded": model is not None}
