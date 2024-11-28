import os

class Config:
    """
    Configuration class for global settings.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "models", "final_model.joblib")
    LOG_DIR = os.path.join(BASE_DIR, "artifacts", "logs")
    METRICS_DIR = os.path.join(BASE_DIR, "artifacts", "metrics")
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    DEBUG = True
