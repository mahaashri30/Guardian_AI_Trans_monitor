"""
Anomaly Detection Module for GuardianAI ML Service
Uses Isolation Forest to detect statistical outliers in transaction patterns
"""

import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class AnomalyDetectionAgent:
    """Detects anomalous transactions using Isolation Forest"""
    
    def __init__(self):
        """Initialize Isolation Forest model with optimized parameters"""
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expect ~10% outliers
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, training_features: np.ndarray) -> Dict[str, float]:
        """
        Train the anomaly detection model on historical normal transactions
        
        Args:
            training_features: 2D array of shape (n_samples, 15)
            
        Returns:
            Training metrics
        """
        logger.info(f"Training anomaly detector on {len(training_features)} samples")
        
        # Scale features for better performance
        scaled_features = self.scaler.fit_transform(training_features)
        
        # Train Isolation Forest
        self.isolation_forest.fit(scaled_features)
        self.is_trained = True
        
        # Calculate training metrics
        train_scores = self.isolation_forest.decision_function(scaled_features)
        train_anomaly_scores = self._normalize_scores(train_scores)
        
        metrics = {
            'training_samples': len(training_features),
            'mean_anomaly_score': float(np.mean(train_anomaly_scores)),
            'std_anomaly_score': float(np.std(train_anomaly_scores)),
            'outlier_threshold': 60.0
        }
        
        logger.info(f"Training completed: {metrics}")
        return metrics
    
    def compute_anomaly_score(self, feature_vector: np.ndarray) -> float:
        """
        Compute anomaly score for a single transaction
        
        Args:
            feature_vector: 1D array of 15 features
            
        Returns:
            Anomaly score (0-100, higher = more anomalous)
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning default score")
            return 10.0
        
        # Reshape for single prediction
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        
        # Scale features
        scaled_features = self.scaler.transform(feature_vector)
        
        # Get isolation forest score (-0.5 to 0)
        raw_score = self.isolation_forest.decision_function(scaled_features)[0]
        
        # Normalize to 0-100 scale
        anomaly_score = self._normalize_scores([raw_score])[0]
        
        return float(anomaly_score)
    
    def is_anomaly(self, feature_vector: np.ndarray, threshold: float = 60.0) -> bool:
        """
        Check if transaction is anomalous
        
        Args:
            feature_vector: 1D array of 15 features
            threshold: Anomaly threshold (default 60)
            
        Returns:
            True if anomalous
        """
        anomaly_score = self.compute_anomaly_score(feature_vector)
        return anomaly_score >= threshold
    
    def _normalize_scores(self, raw_scores: List[float]) -> List[float]:
        """
        Normalize Isolation Forest scores to 0-100 scale
        
        Isolation Forest returns scores from ~-0.5 (normal) to ~0 (anomalous)
        We map this to 0-100 where higher = more anomalous
        """
        # Clip extreme values
        clipped_scores = np.clip(raw_scores, -0.5, 0.1)
        
        # Map to 0-100: score_100 = (score + 0.5) * 166.67
        normalized = (clipped_scores + 0.5) * 166.67
        
        return normalized.tolist()
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.isolation_forest = model_data['isolation_forest']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            
        except FileNotFoundError:
            logger.warning(f"Model file not found: {filepath}")
            self.is_trained = False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_trained = False