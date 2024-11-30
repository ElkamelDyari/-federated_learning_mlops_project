from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
from pydantic import BaseModel
from typing import List
import mlflow
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define input schema for prediction requests
class PredictionInput(BaseModel):
    features: List[float]

# Define output schema for prediction responses
class PredictionOutput(BaseModel):
    prediction: int

# Initialize FastAPI app
app = FastAPI()

# MLflow setup
mlflow.set_tracking_uri("https://dagshub.com/ElkamelDyari/-federated_learning_mlops_project.mlflow")

# Load the preprocessing pipeline
pipeline_path = "artifacts/preprocessing_pipeline.pkl"
try:
    pipeline = joblib.load(pipeline_path)
    logger.info("Preprocessing pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load preprocessing pipeline: {e}")
    raise HTTPException(status_code=500, detail="Preprocessing pipeline not found.")

def preprocess_input(input_features: List[float]):
    """
    Preprocess input features for prediction using the saved pipeline.

    Parameters:
        input_features (List[float]): List of raw input features.

    Returns:
        pd.DataFrame: Preprocessed features ready for prediction.
    """
    try:
        features_df = pd.DataFrame([input_features])
        transformed_features = pipeline.transform(features_df)
        return pd.DataFrame(transformed_features, columns=[f'PC{i + 1}' for i in range(transformed_features.shape[1])])
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail="Preprocessing failed.")

# Load the model from MLflow
MODEL_NAME = os.getenv("MODEL_NAME", "Final_FL")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")

def backend_model(model_name: str, stage: str = "Staging"):
    """
    Retrieve the model from the MLflow Model Registry.

    Parameters:
        model_name (str): The name of the registered model.
        stage (str): The stage of the model (e.g., 'Staging', 'Production').

    Returns:
        The loaded MLflow model.
    """
    try:
        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model.")

try:
    loaded_model = backend_model(MODEL_NAME, MODEL_STAGE)
    logger.info(f"Model '{MODEL_NAME}' loaded successfully from stage '{MODEL_STAGE}'.")
except HTTPException as e:
    logger.error(e.detail)
    raise e

@app.get("/")
async def serve_homepage():
    return FileResponse("src/static/index.html")

# API route for making predictions
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Predict endpoint for the model.

    Parameters:
        input_data (PredictionInput): Input data containing features.

    Returns:
        PredictionOutput: The model's prediction.
    """
    try:
        preprocessed_data = preprocess_input(input_data.features)
        prediction = loaded_model.predict(preprocessed_data)
        return PredictionOutput(prediction=int(prediction[0]))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")
