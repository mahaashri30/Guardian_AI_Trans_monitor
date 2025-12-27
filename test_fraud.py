import requests
import json
from datetime import datetime

def test_fraud_detection():
    base_url = "http://localhost:8005"
    
    # Test normal transaction
    transaction = {
        "user_id": "user_123",
        "amount": 5000,
        "merchant_id": "supplier_a", 
        "device_id": "phone_001",
        "ip_country": "IN",
        "timestamp": datetime.now().isoformat() + "Z",
        "transaction_id": "test_001",
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
    
    print("Testing fraud detection...")
    
    try:
        response = requests.post(f"{base_url}/fraud-check", json=transaction, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: Fraud detection working!")
            print(f"Risk Score: {result['risk_score']}")
            print(f"Action: {result['action']}")
            print(f"Latency: {result['latency_ms']}ms")
            print(f"Explanation: {result['explanation']}")
            return True
        else:
            print(f"FAILED: Status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"ERROR: {e}")
    
    return False

if __name__ == "__main__":
    test_fraud_detection()