from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from typing import List
import mlflow
import joblib

# Define input schema for prediction requests
class PredictionInput(BaseModel):
    features: List[float]

# Define output schema for prediction responses
class PredictionOutput(BaseModel):
    prediction: int

# Initialize FastAPI app
app = FastAPI()

mlflow.set_tracking_uri("https://dagshub.com/ElkamelDyari/-federated_learning_mlops_project.mlflow")

# Load the pipeline
pipeline = joblib.load('artifacts/preprocessing_pipeline.pkl')

def preprocess_input(input_features: List[float]):
    """
    Preprocess input features for prediction using the saved pipeline.

    Parameters:
        input_features (List[float]): List of raw input features.

    Returns:
        pd.DataFrame: Preprocessed features ready for prediction.
    """
    try:
        # Convert input to DataFrame
        features_df = pd.DataFrame([input_features])

        # Transform features using the loaded pipeline
        transformed_features = pipeline.transform(features_df)

        # Return preprocessed features as a DataFrame
        return pd.DataFrame(transformed_features, columns=[f'PC{i + 1}' for i in range(transformed_features.shape[1])])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

# Load the registered model
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
        print(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Load the model
MODEL_NAME = "Final_FL"
MODEL_STAGE = "Staging"
try:
    loaded_model = backend_model(MODEL_NAME, MODEL_STAGE)
    print(f"Model '{MODEL_NAME}' loaded successfully from stage '{MODEL_STAGE}'.")
except HTTPException as e:
    print(e.detail)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Federated Learning API"}

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
        # Preprocess input data
        preprocessed_data = preprocess_input(input_data.features)

        # Make prediction
        prediction = loaded_model.predict(preprocessed_data)
        return PredictionOutput(prediction=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Test function to make a prediction using the X variable
def test_prediction():
    X = [[ 5.30000000e+01,  4.91360000e+04,  1.00000000e+00,
         1.00000000e+00,  4.50000000e+01,  7.70000000e+01,
         4.50000000e+01,  4.50000000e+01,  4.50000000e+01,
         0.00000000e+00,  7.70000000e+01,  7.70000000e+01,
         7.70000000e+01,  0.00000000e+00,  2.48290450e+03,
         4.07033540e+01,  4.91360000e+04,  0.00000000e+00,
         4.91360000e+04,  4.91360000e+04,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  2.00000000e+01,
         2.00000000e+01,  2.03516770e+01,  2.03516770e+01,
         4.50000000e+01,  7.70000000e+01,  5.56666680e+01,
         1.84752080e+01,  3.41333340e+02,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  1.00000000e+00,  8.35000000e+01,
         4.50000000e+01,  7.70000000e+01,  2.00000000e+01,
         1.00000000e+00,  4.50000000e+01,  1.00000000e+00,
         7.70000000e+01, -1.00000000e+00, -1.00000000e+00,
         0.00000000e+00,  2.00000000e+01,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00],
       [ 4.43000000e+02,  1.16429251e+08,  2.10000000e+01,
         1.90000000e+01,  8.15000000e+02,  5.35000000e+03,
         3.72000000e+02,  0.00000000e+00,  3.88095250e+01,
         8.95173800e+01,  1.46000000e+03,  0.00000000e+00,
         2.81578950e+02,  5.11337830e+02,  5.29506100e+01,
         3.43556280e-01,  2.98536550e+06,  4.53535650e+06,
         1.00080760e+07,  2.00000000e+00,  1.16429251e+08,
         5.82146250e+06,  4.90239500e+06,  1.00315510e+07,
         4.00000000e+00,  1.11074742e+08,  6.17081900e+06,
         4.97572900e+06,  1.00369960e+07,  4.40000000e+01,
         0.00000000e+00,  0.00000000e+00,  4.32000000e+02,
         5.24000000e+02,  1.80367050e-01,  1.63189230e-01,
         0.00000000e+00,  1.46000000e+03,  1.50365860e+02,
         3.70058560e+02,  1.36943340e+05,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  1.54125000e+02,
         3.88095250e+01,  2.81578950e+02,  4.32000000e+02,
         2.10000000e+01,  8.15000000e+02,  1.90000000e+01,
         5.35000000e+03,  8.19200000e+03,  9.80000000e+02,
         2.00000000e+01,  2.00000000e+01,  8.96746640e+04,
         2.24485670e+05,  8.02471000e+05,  2.29090000e+04,
         9.61276300e+06,  1.34829890e+06,  1.00080760e+07,
         5.33134900e+06,  0.00000000e+00],
       [ 8.00000000e+01,  2.14539000e+05,  4.00000000e+00,
         5.00000000e+00,  4.29000000e+02,  2.26700000e+03,
         4.29000000e+02,  0.00000000e+00,  1.07250000e+02,
         2.14500000e+02,  1.44800000e+03,  0.00000000e+00,
         4.53400000e+02,  6.59470800e+02,  1.25664795e+04,
         4.19504130e+01,  2.68173750e+04,  5.11047300e+04,
         1.47988000e+05,  1.00000000e+00,  2.14529000e+05,
         7.15096640e+04,  6.62332800e+04,  1.47988000e+05,
         3.28880000e+04,  1.81911000e+05,  4.54777500e+04,
         9.03144700e+04,  1.80948000e+05,  1.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  1.36000000e+02,
         1.68000000e+02,  1.86446290e+01,  2.33057860e+01,
         0.00000000e+00,  1.44800000e+03,  2.69600000e+02,
         4.97176970e+02,  2.47184940e+05,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  1.00000000e+00,  2.99555540e+02,
         1.07250000e+02,  4.53400000e+02,  1.36000000e+02,
         4.00000000e+00,  4.29000000e+02,  5.00000000e+00,
         2.26700000e+03,  2.92000000e+04,  2.35000000e+02,
         1.00000000e+00,  3.20000000e+01,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00],
       [ 5.30000000e+01,  1.34464000e+05,  2.00000000e+00,
         2.00000000e+00,  9.20000000e+01,  2.28000000e+02,
         4.60000000e+01,  4.60000000e+01,  4.60000000e+01,
         0.00000000e+00,  1.14000000e+02,  1.14000000e+02,
         1.14000000e+02,  0.00000000e+00,  2.37981900e+03,
         2.97477400e+01,  4.48213320e+04,  7.75800000e+04,
         1.34403000e+05,  1.30000000e+01,  1.30000000e+01,
         1.30000000e+01,  0.00000000e+00,  1.30000000e+01,
         1.30000000e+01,  4.80000000e+01,  4.80000000e+01,
         0.00000000e+00,  4.80000000e+01,  4.80000000e+01,
         0.00000000e+00,  0.00000000e+00,  6.40000000e+01,
         6.40000000e+01,  1.48738700e+01,  1.48738700e+01,
         4.60000000e+01,  1.14000000e+02,  7.32000000e+01,
         3.72451320e+01,  1.38720000e+03,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  1.00000000e+00,  9.15000000e+01,
         4.60000000e+01,  1.14000000e+02,  6.40000000e+01,
         2.00000000e+00,  9.20000000e+01,  2.00000000e+00,
         2.28000000e+02, -1.00000000e+00, -1.00000000e+00,
         1.00000000e+00,  3.20000000e+01,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00],
       [ 5.34700000e+04,  5.20000000e+01,  1.00000000e+00,
         1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         3.84615400e+04,  5.20000000e+01,  0.00000000e+00,
         5.20000000e+01,  5.20000000e+01,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  3.20000000e+01,
         3.20000000e+01,  1.92307700e+04,  1.92307700e+04,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00,  1.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  3.20000000e+01,
         1.00000000e+00,  0.00000000e+00,  1.00000000e+00,
         0.00000000e+00,  1.18000000e+02,  3.05000000e+02,
         0.00000000e+00,  3.20000000e+01,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00]]
    input_data = PredictionInput(features=X[0])
    prediction = predict(input_data)
    print(f"Test Prediction: {prediction}")


if __name__ == "__main__":
    test_prediction()