"""
GuardianAI ML Service - FastAPI Application
Fraud detection microservice with <200ms latency
"""

import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ml_core import GuardianAIMLCore
from db import get_conn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GuardianAI ML Service",
    description="ML fraud detection microservice with <200ms latency",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML core
ml_core = GuardianAIMLCore()

# Pydantic models for request/response validation
class TransactionHistory(BaseModel):
    amount: float
    merchant_id: str
    device_id: str
    ip_country: str
    timestamp: str

class FraudCheckRequest(BaseModel):
    user_id: str
    amount: float
    merchant_id: str
    device_id: str
    ip_country: str
    timestamp: str
    transaction_id: str
    user_transaction_history: List[TransactionHistory] = []

class FraudCheckResponse(BaseModel):
    transaction_id: str
    user_id: str
    risk_score: float
    risk_level: str
    action: str
    anomaly_score: float
    explanation: str
    breakdown: Dict[str, float]
    latency_ms: float
    timestamp: str

class TrainingFeature(BaseModel):
    features: Optional[List[float]] = None
    label: str = "normal"
    user_id: Optional[str] = None
    amount: Optional[float] = None
    merchant_id: Optional[str] = None
    device_id: Optional[str] = None
    ip_country: Optional[str] = None
    timestamp: Optional[str] = None
    user_history: Optional[List[Dict[str, Any]]] = []

class TrainRequest(BaseModel):
    training_data: List[TrainingFeature]
    force_retrain: bool = False

class TrainResponse(BaseModel):
    status: str
    model_version: str
    training_samples: int
    accuracy: float
    fraud_precision: float
    fraud_recall: float
    trained_at: str

class ModelStatusResponse(BaseModel):
    model_version: str
    model_trained_at: Optional[str]
    model_accuracy: float
    users_learned: int
    training_samples: int
    next_retrain_scheduled: str
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    dependencies: Dict[str, str]

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time*1000:.1f}ms"
    )
    
    return response

# API Endpoints

@app.post("/fraud-check", response_model=FraudCheckResponse)
async def fraud_check(request: FraudCheckRequest):
    """
    Check if a transaction is fraudulent
    Target latency: <200ms
    """
    try:
        start_time = time.time()
        
        # Convert request to dict format
        transaction = {
            "user_id": request.user_id,
            "amount": request.amount,
            "merchant_id": request.merchant_id,
            "device_id": request.device_id,
            "ip_country": request.ip_country,
            "timestamp": request.timestamp,
            "transaction_id": request.transaction_id
        }
        
        # Convert history to dict format
        user_history = [
            {
                "amount": tx.amount,
                "merchant_id": tx.merchant_id,
                "device_id": tx.device_id,
                "ip_country": tx.ip_country,
                "timestamp": tx.timestamp
            }
            for tx in request.user_transaction_history
        ]
        
        # Process transaction through ML pipeline
        result = ml_core.process_transaction(transaction, user_history)
        
                # Save result to InsForge transactions table
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO transactions
                            (razorpay_id, amount, merchant_id, vpa, notes,
                             risk_score, action, explanation, phishing_flags, created_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        RETURNING id
                        """,
                        (
                            transaction["transaction_id"],            # razorpay_id / txn id
                            transaction["amount"],
                            transaction["merchant_id"],
                            transaction["user_id"],                   # store user_id in vpa column for now
                            "",                                       # notes – you can map later if needed
                            result["risk_score"],
                            result["action"],
                            result.get("explanation", ""),
                            result.get("breakdown", {}).get("phishing_flags", 0),
                            datetime.utcnow(),
                        ),
                    )
                    row = cur.fetchone()
                    result["txn_id"] = row["id"]
        except Exception as db_err:
            logger.error(f"Failed to log to InsForge: {db_err}")

        # Check latency requirement
        total_latency = (time.time() - start_time) * 1000
        if total_latency > 200:
            logger.warning(f"Latency exceeded 200ms: {total_latency:.1f}ms")
        
        return FraudCheckResponse(**result)
        
    except Exception as e:
        logger.error(f"Fraud check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fraud detection failed: {str(e)}")

@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Retrain ML models on new data
    """
    try:
        # Convert request to training data format
        training_data = []
        for sample in request.training_data:
            sample_dict = {
                "label": sample.label,
                "user_id": sample.user_id
            }
            
            if sample.features:
                sample_dict["features"] = sample.features
            else:
                # Build transaction from individual fields
                sample_dict.update({
                    "amount": sample.amount,
                    "merchant_id": sample.merchant_id,
                    "device_id": sample.device_id,
                    "ip_country": sample.ip_country,
                    "timestamp": sample.timestamp,
                    "user_history": sample.user_history or []
                })
            
            training_data.append(sample_dict)
        
        # Train the model
        result = ml_core.train(training_data)
        
        if result["status"] == "success":
            return TrainResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Training failed"))
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/model-status", response_model=ModelStatusResponse)
async def get_model_status():
    """
    Get current model version and metadata
    """
    try:
        status = ml_core.get_model_status()
        return ModelStatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    try:
        health = ml_core.get_health_status()
        return HealthResponse(**health)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("GuardianAI ML Service starting up...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Log service status
    health = ml_core.get_health_status()
    logger.info(f"Service initialized: {health}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("GuardianAI ML Service shutting down...")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8007)),
        reload=False,
        log_level="info"
    )