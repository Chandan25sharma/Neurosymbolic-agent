import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import Dict, List, Any, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class EntityRecognizer:
    """
    Neural entity recognizer using pre-trained NER models.
    Extracts named entities and relationships from text for symbolic reasoning.
    """

    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        """
        Initialize the entity recognizer

        Args:
            model_name: HuggingFace model name for NER
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            # Create NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )

            # Define domain-specific patterns
            self.domain_patterns = self._initialize_domain_patterns()

            logger.info(f"EntityRecognizer initialized with model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize EntityRecognizer: {str(e)}")
            raise e

    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for domain-specific entity recognition"""
        return {
            "CHEMICAL": [
                r'\b[A-Z][a-z]*\d*\b',  # Chemical formulas like H2O, NaCl
                r'\b\w+ine\b',  # Compounds ending in -ine
                r'\b\w+ate\b',  # Compounds ending in -ate
                r'\b\w+yl\b',   # Compounds ending in -yl
            ],
            "MOLECULAR": [
                r'\bmolecule\w*\b',
                r'\bcompound\w*\b',
                r'\bprotein\w*\b',
                r'\benzyme\w*\b',
            ],
            "MEDICAL": [
                r'\b\w+pathy\b',  # Medical conditions ending in -pathy
                r'\b\w+osis\b',   # Medical conditions ending in -osis
                r'\b\w+itis\b',   # Inflammatory conditions ending in -itis
                r'\bsymptom\w*\b',
                r'\bdiagnos\w*\b',
            ],
            "SAFETY": [
                r'\btoxic\w*\b',
                r'\bhazard\w*\b',
                r'\bdanger\w*\b',
                r'\brisk\w*\b',
            ]
        }

    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text using both neural NER and pattern matching

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing extracted entities and relationships
        """
        try:
            # Run neural NER
            ner_results = self.ner_pipeline(text)

            # Run domain-specific pattern matching
            domain_entities = self._extract_domain_entities(text)

            # Extract relationships between entities
            relationships = self._extract_relationships(text, ner_results + domain_entities)

            # Combine and structure results
            entities = self._consolidate_entities(ner_results, domain_entities)

            return {
                "text": text,
                "entities": entities,
                "relationships": relationships,
                "entity_count": len(entities),
                "confidence_scores": self._calculate_entity_confidence(entities)
            }

        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            raise e

    def _extract_domain_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract domain-specific entities using pattern matching

        Args:
            text: Input text

        Returns:
            List of domain entities found
        """
        domain_entities = []

        for entity_type, patterns in self.domain_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    entity = {
                        "entity_group": entity_type,
                        "score": 0.85,  # Fixed confidence for pattern matches
                        "word": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "extraction_method": "pattern_matching"
                    }
                    domain_entities.append(entity)

        return domain_entities

    def _extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between identified entities

        Args:
            text: Original text
            entities: List of extracted entities

        Returns:
            List of relationships between entities
        """
        relationships = []

        # Define relationship patterns
        relationship_patterns = {
            "CAUSES": [r"causes?", r"leads? to", r"results? in", r"triggers?"],
            "CONTAINS": [r"contains?", r"includes?", r"has", r"comprises?"],
            "INTERACTS_WITH": [r"interacts? with", r"reacts? with", r"binds? to"],
            "INCREASES": [r"increases?", r"raises?", r"elevates?", r"boosts?"],
            "DECREASES": [r"decreases?", r"reduces?", r"lowers?", r"inhibits?"],
        }

        # Look for relationships between entity pairs
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i >= j:  # Avoid duplicate pairs and self-relationships
                    continue

                # Extract text between entities
                start_pos = min(entity1["end"], entity2["end"])
                end_pos = max(entity1["start"], entity2["start"])

                if start_pos < end_pos:
                    between_text = text[start_pos:end_pos]
                else:
                    between_text = text[entity1["end"]:entity2["start"]]

                # Check for relationship patterns
                for rel_type, patterns in relationship_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, between_text, re.IGNORECASE):
                            relationship = {
                                "subject": entity1["word"],
                                "predicate": rel_type,
                                "object": entity2["word"],
                                "confidence": 0.7,
                                "context": between_text.strip()
                            }
                            relationships.append(relationship)
                            break

        return relationships

    def _consolidate_entities(self, ner_entities: List[Dict[str, Any]],
                            domain_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Consolidate entities from different extraction methods, removing duplicates

        Args:
            ner_entities: Entities from neural NER
            domain_entities: Entities from pattern matching

        Returns:
            Consolidated list of unique entities
        """
        all_entities = ner_entities + domain_entities
        consolidated = []
        seen_spans = set()

        # Sort by position in text
        all_entities.sort(key=lambda x: x["start"])

        for entity in all_entities:
            span = (entity["start"], entity["end"])

            # Check for overlapping entities
            overlapping = False
            for seen_span in seen_spans:
                if (span[0] < seen_span[1] and span[1] > seen_span[0]):
                    overlapping = True
                    break

            if not overlapping:
                seen_spans.add(span)

                # Enhance entity with additional information
                enhanced_entity = {
                    **entity,
                    "id": f"entity_{len(consolidated)}",
                    "extraction_method": entity.get("extraction_method", "neural_ner"),
                    "normalized_form": self._normalize_entity(entity["word"]),
                    "semantic_type": self._classify_semantic_type(entity)
                }
                consolidated.append(enhanced_entity)

        return consolidated

    def _normalize_entity(self, entity_text: str) -> str:
        """Normalize entity text for better matching"""
        # Convert to lowercase and remove extra whitespace
        normalized = entity_text.lower().strip()

        # Remove common suffixes/prefixes for better matching
        suffixes_to_remove = ["s", "ing", "ed", "er", "est"]
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 2:
                normalized = normalized[:-len(suffix)]
                break

        return normalized

    def _classify_semantic_type(self, entity: Dict[str, Any]) -> str:
        """Classify entity into semantic categories"""
        entity_group = entity.get("entity_group", "")
        entity_text = entity["word"].lower()

        # Domain-specific classification
        if entity_group in ["CHEMICAL", "MOLECULAR"]:
            return "SUBSTANCE"
        elif entity_group == "MEDICAL":
            return "MEDICAL_CONCEPT"
        elif entity_group == "SAFETY":
            return "RISK_FACTOR"
        elif entity_group in ["PER", "PERSON"]:
            return "PERSON"
        elif entity_group in ["ORG", "ORGANIZATION"]:
            return "ORGANIZATION"
        elif entity_group in ["LOC", "LOCATION"]:
            return "LOCATION"
        else:
            return "GENERAL"

    def _calculate_entity_confidence(self, entities: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall confidence statistics for extracted entities"""
        if not entities:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}

        scores = [entity["score"] for entity in entities]

        return {
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "count": len(scores)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_type": "named_entity_recognition",
            "supported_entities": list(self.domain_patterns.keys())
        }
