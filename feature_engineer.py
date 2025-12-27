"""
Feature Engineering Module for GuardianAI ML Service
Extracts 15 features from transaction data for fraud detection
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Extracts 15 features from transaction data for ML fraud detection"""
    
    def compute_transaction_features(self, transaction: Dict[str, Any], 
                                   user_transaction_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract 15 features from transaction and user history
        
        Args:
            transaction: Current transaction data
            user_transaction_history: List of user's historical transactions
            
        Returns:
            Dict with 15 features for ML model
        """
        features = {}
        
        # Parse current transaction
        current_amount = float(transaction['amount'])
        current_time = datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))
        merchant_id = transaction['merchant_id']
        device_id = transaction['device_id']
        ip_country = transaction['ip_country']
        
        # Handle empty history for new users
        if not user_transaction_history:
            return self._get_default_features(transaction)
        
        # Extract historical data
        amounts = [float(tx['amount']) for tx in user_transaction_history]
        merchants = [tx['merchant_id'] for tx in user_transaction_history]
        devices = [tx['device_id'] for tx in user_transaction_history[-20:]]  # Last 20 devices
        countries = [tx['ip_country'] for tx in user_transaction_history[-20:]]  # Last 20 countries
        
        # AMOUNT FEATURES (3)
        user_avg = np.mean(amounts)
        user_std = np.std(amounts) if len(amounts) > 1 else 1.0
        
        features['amount_zscore'] = (current_amount - user_avg) / user_std if user_std > 0 else 0
        features['amount_percentile'] = self._calculate_percentile(current_amount, amounts)
        features['amount_exceeds_3x_avg'] = 1.0 if current_amount > user_avg * 3 else 0.0
        
        # VELOCITY FEATURES (3)
        features['txns_in_last_hour'] = self._count_transactions_in_window(
            user_transaction_history, current_time, minutes=60)
        features['txns_in_last_minute'] = self._count_transactions_in_window(
            user_transaction_history, current_time, minutes=1)
        features['high_velocity_burst'] = 1.0 if features['txns_in_last_minute'] >= 2 else 0.0
        
        # MERCHANT FEATURES (3)
        features['new_merchant'] = 1.0 if merchant_id not in merchants else 0.0
        features['merchant_tx_count'] = merchants.count(merchant_id)
        
        merchant_amounts = [float(tx['amount']) for tx in user_transaction_history 
                          if tx['merchant_id'] == merchant_id]
        merchant_avg = np.mean(merchant_amounts) if merchant_amounts else current_amount
        features['amount_vs_merchant_avg'] = current_amount / merchant_avg if merchant_avg > 0 else 1.0
        
        # LOCATION FEATURES (3)
        features['new_device'] = 1.0 if device_id not in devices else 0.0
        features['new_country'] = 1.0 if ip_country not in countries else 0.0
        features['new_location_flag'] = 1.0 if (features['new_device'] and features['new_country']) else 0.0
        
        # TIME FEATURES (3)
        hour = current_time.hour
        weekday = current_time.weekday()
        
        features['is_unusual_hour'] = 1.0 if hour < 6 or hour > 23 else 0.0
        features['hour_of_day'] = float(hour)
        features['is_weekend'] = 1.0 if weekday >= 5 else 0.0
        
        return features
    
    def _get_default_features(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Return default features for new users with no history"""
        current_time = datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))
        hour = current_time.hour
        weekday = current_time.weekday()
        
        return {
            'amount_zscore': 0.0,
            'amount_percentile': 50.0,
            'amount_exceeds_3x_avg': 0.0,
            'txns_in_last_hour': 0.0,
            'txns_in_last_minute': 0.0,
            'high_velocity_burst': 0.0,
            'new_merchant': 1.0,  # New user = new merchant
            'merchant_tx_count': 0.0,
            'amount_vs_merchant_avg': 1.0,
            'new_device': 1.0,  # New user = new device
            'new_country': 0.0,
            'new_location_flag': 0.0,
            'is_unusual_hour': 1.0 if hour < 6 or hour > 23 else 0.0,
            'hour_of_day': float(hour),
            'is_weekend': 1.0 if weekday >= 5 else 0.0
        }
    
    def _calculate_percentile(self, value: float, historical_values: List[float]) -> float:
        """Calculate percentile rank of value in historical values"""
        if not historical_values:
            return 50.0
        
        sorted_values = sorted(historical_values)
        rank = sum(1 for v in sorted_values if v <= value)
        return (rank / len(sorted_values)) * 100
    
    def _count_transactions_in_window(self, history: List[Dict[str, Any]], 
                                    current_time: datetime, minutes: int) -> float:
        """Count transactions within time window"""
        window_start = current_time - timedelta(minutes=minutes)
        count = 0
        
        for tx in history:
            tx_time = datetime.fromisoformat(tx['timestamp'].replace('Z', '+00:00'))
            if window_start <= tx_time <= current_time:
                count += 1
        
        return float(count)
    
    def get_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert features dict to numpy array for ML model"""
        feature_order = [
            'amount_zscore', 'amount_percentile', 'amount_exceeds_3x_avg',
            'txns_in_last_hour', 'txns_in_last_minute', 'high_velocity_burst',
            'new_merchant', 'merchant_tx_count', 'amount_vs_merchant_avg',
            'new_device', 'new_country', 'new_location_flag',
            'is_unusual_hour', 'hour_of_day', 'is_weekend'
        ]
        
        return np.array([features[key] for key in feature_order])