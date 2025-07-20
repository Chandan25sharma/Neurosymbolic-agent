import random
from typing import Dict, List, Any
import logging
import re

logger = logging.getLogger(__name__)

class TextClassifier:
    """
    Mock neural text classifier for demonstration purposes.
    Simulates text classification without requiring heavy ML libraries.
    """

    def __init__(self, model_name: str = "mock-distilbert-text-classifier"):
        """
        Initialize the mock text classifier

        Args:
            model_name: Mock model name for text classification
        """
        self.model_name = model_name
        self.device = "cpu"

        # Mock initialization
        logger.info(f"TextClassifier initialized with mock model: {model_name}")

        # Keywords for different classifications
        self.toxicity_keywords = ["toxic", "poison", "dangerous", "harmful", "lethal", "hazardous"]
        self.medical_keywords = ["medicine", "drug", "treatment", "therapy", "patient", "doctor", "symptom"]
        self.allergy_keywords = ["allergy", "allergic", "reaction", "peanut", "shellfish"]
        self.safety_keywords = ["safe", "secure", "protection", "approved", "tested"]

    async def predict(self, text: str) -> Dict[str, Any]:
        """
        Mock prediction for input text

        Args:
            text: Input text to classify

        Returns:
            Dictionary containing mock prediction results
        """
        try:
            text_lower = text.lower()

            # Determine label based on keywords
            label = "GENERAL_NEUTRAL"
            confidence = 0.6 + random.random() * 0.3  # Random confidence between 0.6-0.9

            # Check for specific patterns
            if any(keyword in text_lower for keyword in self.toxicity_keywords):
                label = "TOXICITY_WARNING"
                confidence = 0.8 + random.random() * 0.15
            elif any(keyword in text_lower for keyword in self.allergy_keywords):
                label = "ALLERGY_ALERT"
                confidence = 0.85 + random.random() * 0.1
            elif any(keyword in text_lower for keyword in self.medical_keywords):
                label = "MEDICAL_POSITIVE"
                confidence = 0.75 + random.random() * 0.2
            elif any(keyword in text_lower for keyword in self.safety_keywords):
                label = "SAFETY_POSITIVE"
                confidence = 0.7 + random.random() * 0.25
            elif any(word in text_lower for word in ["risk", "hazard", "unsafe"]):
                label = "RISK_IDENTIFIED"
                confidence = 0.7 + random.random() * 0.2

            # Extract mock entities
            entities = self._extract_mock_entities(text)

            return {
                "text": text,
                "label": label,
                "confidence": confidence,
                "raw_prediction": {"label": label, "score": confidence},
                "entities": entities,
                "model_used": self.model_name
            }

        except Exception as e:
            logger.error(f"Mock prediction failed: {str(e)}")
            raise e

    def _extract_mock_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mock entities from text

        Args:
            text: Input text

        Returns:
            List of mock detected entities
        """
        entities = []
        text_lower = text.lower()

        # Mock chemical entities
        chemical_patterns = ["h2o", "co2", "nacl", "compound", "molecule", "chemical"]
        for pattern in chemical_patterns:
            if pattern in text_lower:
                entities.append({
                    "word": pattern,
                    "label": "CHEMICAL",
                    "score": 0.8 + random.random() * 0.15,
                    "start": text_lower.find(pattern),
                    "end": text_lower.find(pattern) + len(pattern)
                })

        # Mock medical entities
        medical_patterns = ["patient", "doctor", "medicine", "treatment", "symptom"]
        for pattern in medical_patterns:
            if pattern in text_lower:
                entities.append({
                    "word": pattern,
                    "label": "MEDICAL",
                    "score": 0.85 + random.random() * 0.1,
                    "start": text_lower.find(pattern),
                    "end": text_lower.find(pattern) + len(pattern)
                })

        return entities[:5]  # Limit to 5 entities

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the mock model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_parameters": 125000000,  # Mock parameter count
            "model_type": "text_classification",
            "is_mock": True
        }
