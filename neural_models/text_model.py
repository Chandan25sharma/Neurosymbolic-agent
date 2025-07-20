import random
from typing import Dict, List, Any
import logging
import re

logger = logging.getLogger(__name__)

class TextClassifier:
    """
    Neural text classifier using pre-trained transformers for pattern recognition.
    Converts raw text into classified labels with confidence scores.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the text classifier

        Args:
            model_name: HuggingFace model name for text classification
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            # Create pipeline for easier inference
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )

            logger.info(f"TextClassifier initialized with model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize TextClassifier: {str(e)}")
            raise e

    async def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict classification for input text

        Args:
            text: Input text to classify

        Returns:
            Dictionary containing prediction results
        """
        try:
            # Run classification
            results = self.classifier(text)

            # Extract top prediction
            top_prediction = results[0] if isinstance(results, list) else results

            # Map to our domain-specific categories
            domain_label = self._map_to_domain(top_prediction['label'], text)

            # Extract entities for additional context
            entities = self._extract_entities(text)

            return {
                "text": text,
                "label": domain_label,
                "confidence": float(top_prediction['score']),
                "raw_prediction": top_prediction,
                "entities": entities,
                "model_used": self.model_name
            }

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise e

    def _map_to_domain(self, raw_label: str, text: str) -> str:
        """
        Map generic model labels to domain-specific categories

        Args:
            raw_label: Original model prediction label
            text: Input text for context

        Returns:
            Domain-specific label
        """
        # Example domain mapping for different use cases
        domain_mappings = {
            "POSITIVE": self._classify_positive_context(text),
            "NEGATIVE": self._classify_negative_context(text),
            "LABEL_0": "NEUTRAL",
            "LABEL_1": "POSITIVE_SENTIMENT"
        }

        return domain_mappings.get(raw_label, raw_label)

    def _classify_positive_context(self, text: str) -> str:
        """Classify positive sentiment into specific domains"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["medicine", "drug", "treatment", "therapy"]):
            return "MEDICAL_POSITIVE"
        elif any(word in text_lower for word in ["food", "eat", "nutrition", "diet"]):
            return "FOOD_POSITIVE"
        elif any(word in text_lower for word in ["safe", "secure", "protection"]):
            return "SAFETY_POSITIVE"
        else:
            return "GENERAL_POSITIVE"

    def _classify_negative_context(self, text: str) -> str:
        """Classify negative sentiment into specific domains"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["toxic", "poison", "dangerous", "harmful"]):
            return "TOXICITY_WARNING"
        elif any(word in text_lower for word in ["allergy", "allergic", "reaction"]):
            return "ALLERGY_ALERT"
        elif any(word in text_lower for word in ["risk", "hazard", "unsafe"]):
            return "RISK_IDENTIFIED"
        else:
            return "GENERAL_NEGATIVE"

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text for additional context

        Args:
            text: Input text

        Returns:
            List of detected entities
        """
        # Simple keyword-based entity extraction
        # In a production system, you'd use a proper NER model
        entities = []

        # Chemical/molecular entities
        chemical_patterns = ["molecule", "compound", "chemical", "acid", "base"]
        for pattern in chemical_patterns:
            if pattern in text.lower():
                entities.append({
                    "text": pattern,
                    "label": "CHEMICAL",
                    "confidence": 0.8
                })

        # Medical entities
        medical_patterns = ["symptom", "disease", "treatment", "medication"]
        for pattern in medical_patterns:
            if pattern in text.lower():
                entities.append({
                    "text": pattern,
                    "label": "MEDICAL",
                    "confidence": 0.8
                })

        return entities

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_type": "text_classification"
        }
