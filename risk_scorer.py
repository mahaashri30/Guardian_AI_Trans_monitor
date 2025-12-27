"""
Risk Scoring Module for GuardianAI ML Service
Implements 4-layer risk scoring system with behavioral rules
"""

from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class RiskScoringAgent:
    """Scores transaction risk using 4-layer weighted system"""
    
    def __init__(self):
        """Initialize risk scoring thresholds and rules"""
        self.high_risk_merchants = {'jewelry', 'forex', 'gambling', 'crypto'}
        
    def score_risk(self, anomaly_score: float, features: Dict[str, float]) -> Tuple[float, str, str, Dict[str, float]]:
        """
        Calculate risk score using 4-layer weighted system
        
        Args:
            anomaly_score: Anomaly score (0-100)
            features: Dict of 15 transaction features
            
        Returns:
            Tuple of (risk_score, risk_level, action, breakdown)
        """
        # LAYER 1 - Anomaly Contribution (weight: 40%)
        points_1 = self._score_anomaly_layer(anomaly_score)
        
        # LAYER 2 - Behavioral Rules (weight: 30%)
        points_2 = self._score_behavior_layer(features)
        
        # LAYER 3 - Amount Checks (weight: 20%)
        points_3 = self._score_amount_layer(features)
        
        # LAYER 4 - Merchant Trust (weight: 10%)
        points_4 = self._score_merchant_layer(features)
        
        # Calculate weighted final score
        risk_score = (points_1 * 0.40) + (points_2 * 0.30) + (points_3 * 0.20) + (points_4 * 0.10)
        risk_score = min(100.0, max(0.0, risk_score))  # Clamp to 0-100
        
        # Get risk level and action
        risk_level = self.get_risk_level(risk_score)
        action = self.get_action(risk_score)
        
        # Breakdown for transparency
        breakdown = {
            'anomaly_layer': round(points_1, 1),
            'behavior_layer': round(points_2, 1),
            'amount_layer': round(points_3, 1),
            'merchant_layer': round(points_4, 1)
        }
        
        return risk_score, risk_level, action, breakdown
    
    def _score_anomaly_layer(self, anomaly_score: float) -> float:
        """Layer 1: Anomaly contribution (max 40 points)"""
        if anomaly_score < 40:
            return 0.0
        elif anomaly_score < 60:
            return 10.0
        elif anomaly_score < 80:
            return 20.0
        else:
            return 40.0
    
    def _score_behavior_layer(self, features: Dict[str, float]) -> float:
        """Layer 2: Behavioral rules (max 30 points)"""
        points = 0.0
        
        # Card testing pattern: small amounts + rapid transactions
        if features.get('amount_zscore', 0) < -1 and features.get('txns_in_last_minute', 0) >= 2:
            points += 15.0
        
        # Velocity attack: too many transactions per hour
        if features.get('txns_in_last_hour', 0) > 10:
            points += 10.0
        
        # Account compromise: new device + new country
        if features.get('new_device', 0) and features.get('new_country', 0):
            points += 8.0
        
        # Suspicious new device with large amount
        if features.get('new_device', 0) and features.get('amount_exceeds_3x_avg', 0):
            points += 5.0
        
        # Unusual time patterns
        if features.get('is_unusual_hour', 0):
            points += 3.0
        
        return min(30.0, points)
    
    def _score_amount_layer(self, features: Dict[str, float]) -> float:
        """Layer 3: Amount checks (max 20 points)"""
        amount_zscore = features.get('amount_zscore', 0)
        
        if amount_zscore > 5:
            return 15.0
        elif amount_zscore > 3:
            return 8.0
        elif amount_zscore > 2:
            return 4.0
        else:
            return 0.0
    
    def _score_merchant_layer(self, features: Dict[str, float]) -> float:
        """Layer 4: Merchant trust (max 10 points)"""
        points = 0.0
        
        # New merchant penalty
        if features.get('new_merchant', 0):
            points += 4.0
        
        # Low merchant familiarity
        if features.get('merchant_tx_count', 0) < 3:
            points += 2.0
        
        # High-risk merchant categories (would need merchant category in features)
        # For now, use merchant transaction count as proxy
        if features.get('merchant_tx_count', 0) == 0:  # Completely new merchant
            points += 3.0
        
        return min(10.0, points)
    
    def get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score <= 20:
            return "LOW"
        elif risk_score <= 40:
            return "MEDIUM"
        elif risk_score <= 70:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def get_action(self, risk_score: float) -> str:
        """Determine action based on risk score"""
        if risk_score <= 40:
            return "ALLOW"
        elif risk_score <= 70:
            return "REVIEW"
        else:
            return "BLOCK"
    
    def generate_explanation(self, risk_score: float, features: Dict[str, float], 
                           breakdown: Dict[str, float]) -> str:
        """Generate human-readable explanation"""
        
        if risk_score <= 20:
            return "✓ Low risk - Normal transaction pattern"
        
        explanations = []
        
        # Check for specific patterns
        if features.get('txns_in_last_minute', 0) >= 2 and features.get('amount_zscore', 0) < -1:
            explanations.append("🔴 Card testing pattern: rapid small transactions")
        
        if features.get('new_device', 0) and features.get('new_country', 0):
            explanations.append("🛑 Account compromise: new device + new country")
        
        if features.get('amount_exceeds_3x_avg', 0):
            explanations.append("⚠️ Extreme amount: 3x above user average")
        
        if features.get('txns_in_last_hour', 0) > 10:
            explanations.append("🚨 Velocity attack: excessive transaction rate")
        
        if features.get('new_merchant', 0) and features.get('amount_zscore', 0) > 2:
            explanations.append("🔍 New merchant with unusual amount")
        
        if features.get('is_unusual_hour', 0):
            explanations.append("🌙 Unusual time: late night transaction")
        
        # High anomaly score
        if breakdown.get('anomaly_layer', 0) >= 20:
            explanations.append("📊 Statistical anomaly detected")
        
        if not explanations:
            if risk_score <= 40:
                return "⚠️ Medium risk - Some unusual patterns detected"
            elif risk_score <= 70:
                return "🔍 High risk - Multiple risk factors present"
            else:
                return "🛑 Critical risk - Strong fraud indicators"
        
        return " | ".join(explanations[:2])  # Limit to 2 main reasons