import requests
import json

# Train model
training_data = {
    "training_data": [
        {"features": [0.5, 45.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0, 1.2, 0.0, 0.0, 0.0, 0.0, 14.0, 0.0], "label": "normal", "user_id": "user1"},
        {"features": [5.2, 95.0, 1.0, 15.0, 3.0, 1.0, 1.0, 0.0, 8.5, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0], "label": "fraud", "user_id": "user2"}
    ],
    "force_retrain": True
}

print("Training model...")
response = requests.post("http://localhost:8005/train", json=training_data)
print("Training response:", response.status_code)
if response.status_code == 200:
    print("Training result:", response.json())

print("\nChecking model status...")
status_response = requests.get("http://localhost:8005/model-status")
print("Model status:", status_response.json())