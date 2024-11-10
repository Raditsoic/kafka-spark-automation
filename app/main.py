from fastapi import FastAPI, HTTPException
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

app = FastAPI(
    title="Toxic Comment Classifier",
    description="API for predicting toxicity in comments using PySpark ML model",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommentRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    toxicity_score: float
    prediction_details: Optional[dict] = None

class BatchCommentRequest(BaseModel):
    comments: List[str]

spark = None
model = None

def initialize_spark():
    global spark
    spark = SparkSession.builder \
        .appName("ToxicCommentClassifier") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_model(model_path: str):
    global model
    try:
        model = PipelineModel.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    global spark, model
    try:
        spark = initialize_spark()
        model_path = "./model/Logistic_Regression_20241109_091937"  
        model = load_model(model_path)
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    global spark
    if spark:
        spark.stop()
        logger.info("Spark session stopped")

@app.post("/predict", response_model=PredictionResponse)
async def predict_toxicity(request: CommentRequest):
    """Predict toxicity for a single comment"""
    try:
        # Create DataFrame from input text
        data = [(request.text,)]
        df = spark.createDataFrame(data, ["comment_text"])
        
        # Make prediction
        result = model.transform(df)
        
        # Extract prediction and probability
        prediction_row = result.select("comment_text", "prediction", "probability").first()
        
        return PredictionResponse(
            text=prediction_row["comment_text"],
            toxicity_score=float(prediction_row["prediction"]),
            prediction_details={
                "probability_scores": prediction_row["probability"].toArray().tolist()
            }
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def batch_predict_toxicity(request: BatchCommentRequest):
    """Predict toxicity for multiple comments"""
    try:
        data = [(text,) for text in request.comments]
        df = spark.createDataFrame(data, ["comment_text"])
        
        results = model.transform(df)
        
        predictions = []
        for row in results.collect():
            predictions.append(
                PredictionResponse(
                    text=row["comment_text"],
                    toxicity_score=float(row["prediction"]),
                    prediction_details={
                        "probability_scores": row["probability"].toArray().tolist()
                    }
                )
            )
        return predictions
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health-check")
async def health_check():
    return {
        "status": "healthy",
        "spark_active": spark is not None,
        "model_loaded": model is not None
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the Toxic Comment Classifier API!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)