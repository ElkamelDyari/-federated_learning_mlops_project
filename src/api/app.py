from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
from src.data.feature_engineering import process_pred
import dagshub
import pandas as pd
import mlflow


app = FastAPI()

# Load the model from MLflow Model Registry
MODEL_NAME = "FederatedGlobalModel"
STAGE = "Production"

mlflow.set_tracking_uri("https://dagshub.com/ElkamelDyari/-federated_learning_mlops_project.mlflow")
dagshub.init(repo_owner='ElkamelDyari', repo_name='-federated_learning_mlops_project', mlflow=True)
model_name = "federated_final_model"
stage = "Production"

# Load the model from the registry
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

data = pd.read_csv("data/processed/data20.csv")
@app.get("/")
def read_root():
    return {"message": "Federated Learning Model API is up and running"}

@app.post("/predict")
def predict(X):

    try:

        X = process_pred(X)
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
