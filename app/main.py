from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pydantic import BaseModel  
from typing import List, Optional, Dict
import uvicorn
import logging
import os
import glob
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommentRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    toxicity_score: float
    prediction_details: Optional[dict] = None

class BatchCommentRequest(BaseModel):
    comments: List[str]

spark = None
models: Dict[str, PipelineModel] = {}

def initialize_spark():
    spark = SparkSession.builder \
        .appName("ToxicCommentClassifier") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_models(base_model_path: str = "./model"):
    """Load all available models and sort them by creation timestamp"""
    model_dirs = glob.glob(os.path.join(base_model_path, "Logistic_Regression_*"))
    
    sorted_dirs = sorted(model_dirs, key=lambda x: re.findall(r'\d{8}_\d{6}', x)[0])
    
    loaded_models = {}
    for i, model_dir in enumerate(sorted_dirs):
        version = f"v1.{i}"
        try:
            model = PipelineModel.load(model_dir)
            loaded_models[version] = {
                'model': model,
                'path': model_dir,
                'timestamp': re.findall(r'\d{8}_\d{6}', model_dir)[0]
            }
            logger.info(f"Loaded model version {version} from {model_dir}")
        except Exception as e:
            logger.error(f"Error loading model from {model_dir}: {str(e)}")
    
    return loaded_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    global spark, models
    try:
        logger.info("Initializing Spark session...")
        spark = initialize_spark()
        
        logger.info("Loading models...")
        models = load_models()
        
        if not models:
            raise Exception("No models were loaded successfully")
            
        logger.info(f"Loaded {len(models)} model versions")
        
        yield
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    finally:
        if spark:
            spark.stop()
            logger.info("Spark session stopped")

app = FastAPI(
    title="Toxic Comment Classifier",
    description="API for predicting toxicity in comments using PySpark ML model",
    version="1.0.0",
    lifespan=lifespan
)

def get_model_info():
    """Get information about available model versions"""
    return {
        version: {
            'path': info['path'],
            'timestamp': info['timestamp']
        }
        for version, info in models.items()
    }

async def predict_with_model(text: str, model_version: str):
    """Make prediction using specified model version"""
    if model_version not in models:
        raise HTTPException(status_code=404, detail=f"Model version {model_version} not found")
    
    try:
        data = [(text,)]
        df = spark.createDataFrame(data, ["comment_text"])
        
        model = models[model_version]['model']
        result = model.transform(df)
        
        prediction_row = result.select("comment_text", "prediction", "probability").first()
        
        return PredictionResponse(
            toxicity_score=float(prediction_row["prediction"]),
            prediction_details={
                "probability_scores": prediction_row["probability"].toArray().tolist(),
                "model_version": model_version,
                "model_timestamp": models[model_version]['timestamp']
            }
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{version}", response_model=PredictionResponse)
async def predict_toxicity(version: str, request: CommentRequest):
    """Predict toxicity for a single comment using specified model version"""
    return await predict_with_model(request.text, version)

@app.post("/predict/{version}/batch", response_model=List[PredictionResponse])
async def batch_predict_toxicity(version: str, request: BatchCommentRequest):
    """Predict toxicity for multiple comments using specified model version"""
    try:
        predictions = []
        for text in request.comments:
            prediction = await predict_with_model(text, version)
            predictions.append(prediction)
        return predictions
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List all available model versions"""
    return get_model_info()

@app.get("/health-check")
async def health_check():
    return {
        "status": "healthy",
        "spark_active": spark is not None,
        "loaded_models": list(models.keys())
    }

@app.get("/")
async def root():
    """Welcome endpoint with available model versions"""
    return {
        "message": "Welcome to the Toxic Comment Classifier API!",
        "available_models": get_model_info()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)