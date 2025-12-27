"""
Unit Tests for GuardianAI ML Service
Tests all modules: feature engineering, anomaly detection, risk scoring, ML core
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from feature_engineer import FeatureEngineer
from anomaly_detector import AnomalyDetectionAgent
from risk_scorer import RiskScoringAgent
from ml_core import GuardianAIMLCore

class TestFeatureEngineer(unittest.TestCase):
    """Test feature engineering module"""
    
    def setUp(self):
        self.engineer = FeatureEngineer()
        self.base_time = datetime.now()
        
    def test_compute_features_new_user(self):
        """Test feature extraction for new user with no history"""
        transaction = {
            "amount": 5000,
            "merchant_id": "supplier_a",
            "device_id": "phone_001",
            "ip_country": "IN",
            "timestamp": self.base_time.isoformat() + "Z"
        }
        
        features = self.engineer.compute_transaction_features(transaction, [])
        
        # Should have all 15 features
        self.assertEqual(len(features), 15)
        
        # New user defaults
        self.assertEqual(features['new_merchant'], 1.0)
        self.assertEqual(features['new_device'], 1.0)
        self.assertEqual(features['merchant_tx_count'], 0.0)
        
    def test_compute_features_with_history(self):
        """Test feature extraction with transaction history"""
        transaction = {
            "amount": 10000,  # 2x average
            "merchant_id": "supplier_a",
            "device_id": "phone_001",
            "ip_country": "IN",
            "timestamp": self.base_time.isoformat() + "Z"
        }
        
        history = [
            {
                "amount": 5000,
                "merchant_id": "supplier_a",
                "device_id": "phone_001",
                "ip_country": "IN",
                "timestamp": (self.base_time - timedelta(hours=1)).isoformat() + "Z"
            }
        ]
        
        features = self.engineer.compute_transaction_features(transaction, history)
        
        # Known merchant
        self.assertEqual(features['new_merchant'], 0.0)
        self.assertEqual(features['merchant_tx_count'], 1.0)
        
        # Amount features
        self.assertGreater(features['amount_zscore'], 0)  # Above average
        
    def test_velocity_features(self):
        """Test velocity feature calculation"""
        current_time = self.base_time
        transaction = {
            "amount": 1000,
            "merchant_id": "test",
            "device_id": "phone",
            "ip_country": "IN",
            "timestamp": current_time.isoformat() + "Z"
        }
        
        # Create rapid transactions
        history = []
        for i in range(3):
            tx_time = current_time - timedelta(seconds=30 + i*10)
            history.append({
                "amount": 1000,
                "merchant_id": "test",
                "device_id": "phone",
                "ip_country": "IN",
                "timestamp": tx_time.isoformat() + "Z"
            })
        
        features = self.engineer.compute_transaction_features(transaction, history)
        
        self.assertGreater(features['txns_in_last_hour'], 0)
        self.assertGreater(features['txns_in_last_minute'], 0)

class TestAnomalyDetector(unittest.TestCase):
    """Test anomaly detection module"""
    
    def setUp(self):
        self.detector = AnomalyDetectionAgent()
        
    def test_training(self):
        """Test model training"""
        # Generate synthetic normal data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 15))
        
        metrics = self.detector.train(normal_data)
        
        self.assertTrue(self.detector.is_trained)
        self.assertEqual(metrics['training_samples'], 100)
        
    def test_anomaly_scoring(self):
        """Test anomaly score computation"""
        # Train on normal data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 15))
        self.detector.train(normal_data)
        
        # Test normal transaction
        normal_tx = np.random.normal(0, 1, 15)
        normal_score = self.detector.compute_anomaly_score(normal_tx)
        
        # Test anomalous transaction
        anomalous_tx = np.random.normal(5, 1, 15)  # Far from training data
        anomalous_score = self.detector.compute_anomaly_score(anomalous_tx)
        
        # Anomalous should have higher score
        self.assertGreater(anomalous_score, normal_score)
        
    def test_save_load_model(self):
        """Test model persistence"""
        # Train model
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (50, 15))
        self.detector.train(normal_data)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.detector.save_model(tmp.name)
            
            # Create new detector and load
            new_detector = AnomalyDetectionAgent()
            new_detector.load_model(tmp.name)
            
            self.assertTrue(new_detector.is_trained)
            
            # Cleanup
            os.unlink(tmp.name)

class TestRiskScorer(unittest.TestCase):
    """Test risk scoring module"""
    
    def setUp(self):
        self.scorer = RiskScoringAgent()
        
    def test_low_risk_scoring(self):
        """Test low risk transaction scoring"""
        features = {
            'amount_zscore': 0.5,
            'txns_in_last_minute': 0,
            'txns_in_last_hour': 1,
            'new_device': 0,
            'new_country': 0,
            'new_merchant': 0,
            'amount_exceeds_3x_avg': 0,
            'is_unusual_hour': 0,
            'merchant_tx_count': 5
        }
        
        risk_score, risk_level, action, breakdown = self.scorer.score_risk(10.0, features)
        
        self.assertLess(risk_score, 30)
        self.assertEqual(risk_level, "LOW")
        self.assertEqual(action, "ALLOW")
        
    def test_high_risk_scoring(self):
        """Test high risk transaction scoring"""
        features = {
            'amount_zscore': 6.0,  # Very high amount
            'txns_in_last_minute': 3,  # Rapid transactions
            'txns_in_last_hour': 15,  # High velocity
            'new_device': 1,
            'new_country': 1,
            'new_merchant': 1,
            'amount_exceeds_3x_avg': 1,
            'is_unusual_hour': 1,
            'merchant_tx_count': 0
        }
        
        risk_score, risk_level, action, breakdown = self.scorer.score_risk(90.0, features)
        
        self.assertGreater(risk_score, 70)
        self.assertEqual(action, "BLOCK")
        
    def test_explanation_generation(self):
        """Test explanation generation"""
        features = {
            'txns_in_last_minute': 3,
            'amount_zscore': -2,  # Small amount
            'new_device': 1,
            'new_country': 1
        }
        
        breakdown = {'anomaly_layer': 30, 'behavior_layer': 20, 'amount_layer': 5, 'merchant_layer': 5}
        explanation = self.scorer.generate_explanation(60, features, breakdown)
        
        self.assertIn("Card testing", explanation)

class TestMLCore(unittest.TestCase):
    """Test ML core integration"""
    
    def setUp(self):
        # Create temporary model path
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")
        self.core = GuardianAIMLCore(self.model_path)
        
    def tearDown(self):
        # Cleanup
        if os.path.exists(self.model_path):
            os.unlink(self.model_path)
        os.rmdir(self.temp_dir)
        
    def test_process_transaction_normal(self):
        """Test normal transaction processing"""
        transaction = {
            "user_id": "user_123",
            "amount": 5000,
            "merchant_id": "supplier_a",
            "device_id": "phone_001",
            "ip_country": "IN",
            "timestamp": datetime.now().isoformat() + "Z",
            "transaction_id": "txn_123"
        }
        
        history = [
            {
                "amount": 4500,
                "merchant_id": "supplier_a",
                "device_id": "phone_001",
                "ip_country": "IN",
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat() + "Z"
            }
        ]
        
        result = self.core.process_transaction(transaction, history)
        
        # Check result structure
        self.assertIn("transaction_id", result)
        self.assertIn("risk_score", result)
        self.assertIn("action", result)
        self.assertIn("latency_ms", result)
        
        # Should be low risk for normal pattern
        self.assertLess(result["risk_score"], 50)
        
    def test_process_transaction_suspicious(self):
        """Test suspicious transaction processing"""
        transaction = {
            "user_id": "user_123",
            "amount": 50000,  # Very high amount
            "merchant_id": "unknown_merchant",
            "device_id": "new_device",
            "ip_country": "XX",  # New country
            "timestamp": "2025-12-27T03:00:00Z",  # Unusual hour
            "transaction_id": "txn_suspicious"
        }
        
        history = [
            {
                "amount": 1000,
                "merchant_id": "supplier_a",
                "device_id": "phone_001",
                "ip_country": "IN",
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat() + "Z"
            }
        ]
        
        result = self.core.process_transaction(transaction, history)
        
        # Should be high risk
        self.assertGreater(result["risk_score"], 50)
        self.assertIn(result["action"], ["REVIEW", "BLOCK"])
        
    def test_training(self):
        """Test model training"""
        # Generate training data
        training_data = []
        
        # Normal transactions
        for i in range(50):
            training_data.append({
                "features": np.random.normal(0, 1, 15).tolist(),
                "label": "normal",
                "user_id": f"user_{i}"
            })
        
        # Fraud transactions
        for i in range(10):
            training_data.append({
                "features": np.random.normal(3, 1, 15).tolist(),  # Anomalous
                "label": "fraud",
                "user_id": f"fraud_user_{i}"
            })
        
        result = self.core.train(training_data)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["training_samples"], 60)
        self.assertGreater(result["accuracy"], 0.5)
        
    def test_model_status(self):
        """Test model status retrieval"""
        status = self.core.get_model_status()
        
        self.assertIn("model_version", status)
        self.assertIn("status", status)
        
    def test_health_check(self):
        """Test health check"""
        health = self.core.get_health_status()
        
        self.assertEqual(health["status"], "healthy")
        self.assertIn("dependencies", health)

class TestScenarios(unittest.TestCase):
    """Test specific fraud scenarios"""
    
    def setUp(self):
        self.core = GuardianAIMLCore()
        
    def test_scenario_normal_transaction(self):
        """Scenario 1: Normal Transaction"""
        transaction = {
            "user_id": "user_123",
            "amount": 5100,
            "merchant_id": "supplier_a",
            "device_id": "phone_001",
            "ip_country": "IN",
            "timestamp": "2025-12-27T10:00:00Z",
            "transaction_id": "txn_normal"
        }
        
        history = [
            {
                "amount": 5000,
                "merchant_id": "supplier_a",
                "device_id": "phone_001",
                "ip_country": "IN",
                "timestamp": "2025-12-26T10:00:00Z"
            }
        ]
        
        result = self.core.process_transaction(transaction, history)
        
        self.assertLess(result["risk_score"], 20)
        self.assertEqual(result["action"], "ALLOW")
        
    def test_scenario_card_testing(self):
        """Scenario 2: Card Testing"""
        transaction = {
            "user_id": "user_456",
            "amount": 50,  # Small amount
            "merchant_id": "unknown_merchant_1",
            "device_id": "phone_002",
            "ip_country": "IN",
            "timestamp": "2025-12-27T03:00:00Z",  # Unusual hour
            "transaction_id": "txn_testing"
        }
        
        # Recent rapid small transactions
        history = []
        base_time = datetime.fromisoformat("2025-12-27T02:59:00")
        for i in range(4):
            history.append({
                "amount": 50,
                "merchant_id": f"unknown_merchant_{i}",
                "device_id": "phone_002",
                "ip_country": "IN",
                "timestamp": (base_time - timedelta(seconds=i*15)).isoformat() + "Z"
            })
        
        result = self.core.process_transaction(transaction, history)
        
        self.assertGreater(result["risk_score"], 40)
        self.assertIn(result["action"], ["REVIEW", "BLOCK"])
        
    def test_scenario_account_compromise(self):
        """Scenario 3: Account Compromise"""
        transaction = {
            "user_id": "user_789",
            "amount": 75000,  # Very large amount
            "merchant_id": "new_merchant",
            "device_id": "new_device_001",  # New device
            "ip_country": "AE",  # New country
            "timestamp": "2025-12-27T23:30:00Z",  # Late night
            "transaction_id": "txn_compromise"
        }
        
        # Normal historical pattern
        history = [
            {
                "amount": 2000,
                "merchant_id": "supplier_a",
                "device_id": "phone_001",
                "ip_country": "IN",
                "timestamp": "2025-12-26T10:00:00Z"
            }
        ]
        
        result = self.core.process_transaction(transaction, history)
        
        self.assertGreater(result["risk_score"], 70)
        self.assertEqual(result["action"], "BLOCK")

if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)