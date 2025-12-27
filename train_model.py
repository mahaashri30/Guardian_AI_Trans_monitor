"""
Model Training Script for GuardianAI ML Service
Generates synthetic training data and trains the initial model
"""

import numpy as np
import json
from datetime import datetime, timedelta
import random
from ml_core import GuardianAIMLCore

def generate_training_data(num_samples: int = 1000) -> list:
    """Generate synthetic training data for fraud detection"""
    
    training_data = []
    base_time = datetime.now()
    
    # Generate normal transactions (80%)
    for i in range(int(num_samples * 0.8)):
        # Normal user patterns
        user_avg_amount = random.uniform(1000, 10000)
        
        features = [
            random.uniform(-1, 1),      # amount_zscore (normal range)
            random.uniform(20, 80),     # amount_percentile
            0,                          # amount_exceeds_3x_avg
            random.uniform(0, 5),       # txns_in_last_hour
            0,                          # txns_in_last_minute
            0,                          # high_velocity_burst
            random.choice([0, 1]) * 0.3, # new_merchant (mostly known)
            random.uniform(1, 20),      # merchant_tx_count
            random.uniform(0.5, 2),     # amount_vs_merchant_avg
            random.choice([0, 1]) * 0.2, # new_device (mostly same)
            0,                          # new_country (same country)
            0,                          # new_location_flag
            random.choice([0, 1]) * 0.1, # is_unusual_hour (mostly normal hours)
            random.uniform(8, 20),      # hour_of_day (business hours)
            random.choice([0, 1]) * 0.3  # is_weekend
        ]
        
        training_data.append({
            "features": features,
            "label": "normal",
            "user_id": f"normal_user_{i}"
        })
    
    # Generate fraud transactions (20%)
    for i in range(int(num_samples * 0.2)):
        fraud_type = random.choice(['card_testing', 'account_takeover', 'amount_fraud'])
        
        if fraud_type == 'card_testing':
            # Small amounts, rapid transactions, new merchants
            features = [
                random.uniform(-3, -1),     # amount_zscore (small amounts)
                random.uniform(0, 20),      # amount_percentile (low)
                0,                          # amount_exceeds_3x_avg
                random.uniform(10, 30),     # txns_in_last_hour (high)
                random.uniform(2, 5),       # txns_in_last_minute (burst)
                1,                          # high_velocity_burst
                1,                          # new_merchant
                0,                          # merchant_tx_count (new)
                random.uniform(0.1, 0.5),   # amount_vs_merchant_avg
                random.choice([0, 1]),      # new_device
                random.choice([0, 1]),      # new_country
                random.choice([0, 1]),      # new_location_flag
                1,                          # is_unusual_hour (late night)
                random.choice([2, 3, 23]),  # hour_of_day (unusual)
                random.choice([0, 1])       # is_weekend
            ]
            
        elif fraud_type == 'account_takeover':
            # New device + country, large amounts
            features = [
                random.uniform(3, 8),       # amount_zscore (large amounts)
                random.uniform(80, 100),    # amount_percentile (high)
                1,                          # amount_exceeds_3x_avg
                random.uniform(1, 3),       # txns_in_last_hour
                0,                          # txns_in_last_minute
                0,                          # high_velocity_burst
                1,                          # new_merchant
                0,                          # merchant_tx_count
                random.uniform(5, 20),      # amount_vs_merchant_avg (much higher)
                1,                          # new_device
                1,                          # new_country
                1,                          # new_location_flag
                1,                          # is_unusual_hour
                random.choice([1, 2, 23]),  # hour_of_day
                random.choice([0, 1])       # is_weekend
            ]
            
        else:  # amount_fraud
            # Extreme amounts, otherwise normal
            features = [
                random.uniform(5, 15),      # amount_zscore (extreme)
                random.uniform(95, 100),    # amount_percentile
                1,                          # amount_exceeds_3x_avg
                random.uniform(0, 2),       # txns_in_last_hour
                0,                          # txns_in_last_minute
                0,                          # high_velocity_burst
                random.choice([0, 1]) * 0.5, # new_merchant
                random.uniform(0, 5),       # merchant_tx_count
                random.uniform(10, 50),     # amount_vs_merchant_avg
                random.choice([0, 1]) * 0.3, # new_device
                0,                          # new_country
                0,                          # new_location_flag
                random.choice([0, 1]) * 0.2, # is_unusual_hour
                random.uniform(9, 18),      # hour_of_day
                random.choice([0, 1])       # is_weekend
            ]
        
        training_data.append({
            "features": features,
            "label": "fraud",
            "user_id": f"fraud_user_{i}"
        })
    
    # Shuffle the data
    random.shuffle(training_data)
    return training_data

def train_initial_model():
    """Train the initial GuardianAI model"""
    
    print("Generating training data...")
    training_data = generate_training_data(1000)
    
    print(f"Generated {len(training_data)} training samples")
    fraud_count = sum(1 for sample in training_data if sample['label'] == 'fraud')
    print(f"Normal: {len(training_data) - fraud_count}, Fraud: {fraud_count}")
    
    # Initialize ML core
    print("Initializing ML core...")
    ml_core = GuardianAIMLCore()
    
    # Train the model
    print("Training model...")
    result = ml_core.train(training_data)
    
    print("Training Results:")
    if result['status'] == 'success':
        print(f"Status: {result['status']}")
        print(f"Model Version: {result['model_version']}")
        print(f"Training Samples: {result['training_samples']}")
        print(f"Accuracy: {result['accuracy']:.3f}")
        print(f"Fraud Precision: {result['fraud_precision']:.3f}")
        print(f"Fraud Recall: {result['fraud_recall']:.3f}")
    else:
        print(f"Status: {result['status']}")
        print(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    # Test the model with sample transactions
    print("\nTesting model with sample transactions...")
    
    # Test 1: Normal transaction
    normal_tx = {
        "user_id": "test_user_1",
        "amount": 5000,
        "merchant_id": "supplier_a",
        "device_id": "phone_001",
        "ip_country": "IN",
        "timestamp": datetime.now().isoformat() + "Z",
        "transaction_id": "test_normal"
    }
    
    normal_history = [
        {
            "amount": 4500,
            "merchant_id": "supplier_a",
            "device_id": "phone_001",
            "ip_country": "IN",
            "timestamp": (datetime.now() - timedelta(hours=1)).isoformat() + "Z"
        }
    ]
    
    result1 = ml_core.process_transaction(normal_tx, normal_history)
    print(f"Normal Transaction: Risk={result1['risk_score']}, Action={result1['action']}")
    
    # Test 2: Suspicious transaction
    suspicious_tx = {
        "user_id": "test_user_2",
        "amount": 75000,
        "merchant_id": "unknown_merchant",
        "device_id": "new_device",
        "ip_country": "XX",
        "timestamp": "2025-12-27T02:00:00Z",
        "transaction_id": "test_suspicious"
    }
    
    suspicious_history = [
        {
            "amount": 2000,
            "merchant_id": "supplier_a",
            "device_id": "phone_001",
            "ip_country": "IN",
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat() + "Z"
        }
    ]
    
    result2 = ml_core.process_transaction(suspicious_tx, suspicious_history)
    print(f"Suspicious Transaction: Risk={result2['risk_score']}, Action={result2['action']}")
    
    print("\nModel training completed successfully!")
    print("The model is ready for deployment.")

if __name__ == "__main__":
    train_initial_model()