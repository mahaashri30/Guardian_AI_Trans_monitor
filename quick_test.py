"""
Quick Test Script for GuardianAI ML Service
Tests all endpoints and validates responses
"""

import requests
import json
import time
from datetime import datetime

def test_service():
    """Test all service endpoints"""
    base_url = "http://localhost:8003"
    
    print("=" * 60)
    print("GuardianAI ML Service - Quick Test")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1. Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✓ Health Status: {health['status']}")
            print(f"✓ Model Loaded: {health['model_loaded']}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False
    
    # Test 2: Model Status
    print("\n2. Testing Model Status...")
    try:
        response = requests.get(f"{base_url}/model-status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✓ Model Version: {status['model_version']}")
            print(f"✓ Model Status: {status['status']}")
            print(f"✓ Training Samples: {status['training_samples']}")
        else:
            print(f"✗ Model status failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Model status error: {e}")
    
    # Test 3: Normal Transaction
    print("\n3. Testing Normal Transaction...")
    normal_tx = {
        "user_id": "test_user_1",
        "amount": 5000,
        "merchant_id": "supplier_a",
        "device_id": "phone_001",
        "ip_country": "IN",
        "timestamp": datetime.now().isoformat() + "Z",
        "transaction_id": "test_normal_001",
        "user_transaction_history": [
            {
                "amount": 4500,
                "merchant_id": "supplier_a",
                "device_id": "phone_001",
                "ip_country": "IN",
                "timestamp": "2025-12-27T09:00:00Z"
            }
        ]
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/fraud-check", json=normal_tx, timeout=10)
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Risk Score: {result['risk_score']}")
            print(f"✓ Risk Level: {result['risk_level']}")
            print(f"✓ Action: {result['action']}")
            print(f"✓ Latency: {latency:.1f}ms (reported: {result['latency_ms']}ms)")
            print(f"✓ Explanation: {result['explanation']}")
            
            if latency < 200:
                print("✓ Latency requirement met (<200ms)")
            else:
                print("⚠ Latency exceeds 200ms requirement")
        else:
            print(f"✗ Normal transaction test failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"✗ Normal transaction error: {e}")
    
    # Test 4: Suspicious Transaction
    print("\n4. Testing Suspicious Transaction...")
    suspicious_tx = {
        "user_id": "test_user_2",
        "amount": 75000,
        "merchant_id": "unknown_merchant",
        "device_id": "new_device_001",
        "ip_country": "XX",
        "timestamp": "2025-12-27T02:00:00Z",
        "transaction_id": "test_suspicious_001",
        "user_transaction_history": [
            {
                "amount": 2000,
                "merchant_id": "supplier_a",
                "device_id": "phone_001",
                "ip_country": "IN",
                "timestamp": "2025-12-26T10:00:00Z"
            }
        ]
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/fraud-check", json=suspicious_tx, timeout=10)
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Risk Score: {result['risk_score']}")
            print(f"✓ Risk Level: {result['risk_level']}")
            print(f"✓ Action: {result['action']}")
            print(f"✓ Latency: {latency:.1f}ms")
            print(f"✓ Explanation: {result['explanation']}")
            
            if result['risk_score'] > 50:
                print("✓ Correctly identified as high risk")
            else:
                print("⚠ Risk score lower than expected for suspicious transaction")
        else:
            print(f"✗ Suspicious transaction test failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Suspicious transaction error: {e}")
    
    # Test 5: Training Endpoint
    print("\n5. Testing Training Endpoint...")
    training_data = {
        "training_data": [
            {
                "features": [0.5, 45.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0, 1.2, 0.0, 0.0, 0.0, 0.0, 14.0, 0.0],
                "label": "normal",
                "user_id": "train_user_1"
            },
            {
                "features": [5.2, 95.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 8.5, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0],
                "label": "fraud",
                "user_id": "train_user_2"
            }
        ],
        "force_retrain": False
    }
    
    try:
        response = requests.post(f"{base_url}/train", json=training_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Training Status: {result['status']}")
            print(f"✓ Model Version: {result['model_version']}")
            print(f"✓ Accuracy: {result['accuracy']:.3f}")
        else:
            print(f"✗ Training test failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Training error: {e}")
    
    print("\n" + "=" * 60)
    print("Quick Test Complete!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_service()