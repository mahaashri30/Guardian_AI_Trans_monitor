import requests
import json

def test_service():
    base_url = "http://localhost:8005"
    
    print("Testing GuardianAI ML Service...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("SUCCESS: Service is running!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"FAILED: Status {response.status_code}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    return False

if __name__ == "__main__":
    test_service()