from typing import Dict, List, Any, Tuple, Set
import logging
import json
from .grounding_rules import GroundingRules

logger = logging.getLogger(__name__)

class SymbolMapper:
    """
    Maps neural network outputs to symbolic representations for reasoning.
    This is the critical bridge between subsymbolic (neural) and symbolic processing.
    """

    def __init__(self):
        """Initialize the symbol mapper with grounding rules"""
        self.grounding_rules = GroundingRules()
        self.symbol_registry = self._initialize_symbol_registry()
        self.confidence_threshold = 0.5

        logger.info("SymbolMapper initialized")

    def _initialize_symbol_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize registry of symbols with their properties and relationships

        Returns:
            Dictionary mapping symbol names to their properties
        """
        return {
            # Substance symbols
            "SUBSTANCE_TOXIC": {
                "type": "substance",
                "properties": ["dangerous", "harmful", "requires_caution"],
                "relationships": ["CAUSES_HARM", "REQUIRES_PROTECTION"],
                "domain": "safety"
            },
            "SUBSTANCE_SAFE": {
                "type": "substance",
                "properties": ["safe", "non_toxic", "approved"],
                "relationships": ["SAFE_FOR_USE"],
                "domain": "safety"
            },
            "SUBSTANCE_UNKNOWN": {
                "type": "substance",
                "properties": ["unknown_safety", "requires_analysis"],
                "relationships": ["NEEDS_EVALUATION"],
                "domain": "safety"
            },

            # Medical symbols
            "MEDICAL_CONDITION": {
                "type": "condition",
                "properties": ["medical", "health_related", "requires_attention"],
                "relationships": ["AFFECTS_HEALTH", "REQUIRES_TREATMENT"],
                "domain": "medical"
            },
            "ALLERGY_RISK": {
                "type": "risk",
                "properties": ["allergic", "immunological", "individual_specific"],
                "relationships": ["TRIGGERS_REACTION", "REQUIRES_AVOIDANCE"],
                "domain": "medical"
            },

            # Risk symbols
            "HIGH_RISK": {
                "type": "risk_level",
                "properties": ["dangerous", "immediate_attention", "critical"],
                "relationships": ["REQUIRES_INTERVENTION", "CAUSES_CONCERN"],
                "domain": "risk"
            },
            "MEDIUM_RISK": {
                "type": "risk_level",
                "properties": ["moderate", "caution_advised", "monitor"],
                "relationships": ["REQUIRES_MONITORING"],
                "domain": "risk"
            },
            "LOW_RISK": {
                "type": "risk_level",
                "properties": ["minimal", "acceptable", "safe"],
                "relationships": ["ACCEPTABLE_LEVEL"],
                "domain": "risk"
            },

            # Entity symbols
            "ENTITY_PERSON": {
                "type": "entity",
                "properties": ["human", "individual", "decision_maker"],
                "relationships": ["MAKES_DECISIONS", "HAS_PREFERENCES"],
                "domain": "entity"
            },
            "ENTITY_ORGANIZATION": {
                "type": "entity",
                "properties": ["institutional", "collective", "authoritative"],
                "relationships": ["PROVIDES_GUIDANCE", "SETS_STANDARDS"],
                "domain": "entity"
            }
        }

    def map_to_symbols(self, neural_output: Dict[str, Any]) -> List[str]:
        """
        Convert neural network outputs to symbolic representations

        Args:
            neural_output: Output from neural model containing predictions and confidence

        Returns:
            List of symbolic representations
        """
        try:
            symbols = []

            # Extract primary label symbol
            primary_symbol = self._map_label_to_symbol(neural_output)
            if primary_symbol:
                symbols.append(primary_symbol)

            # Extract entity-based symbols
            if "entities" in neural_output:
                entity_symbols = self._map_entities_to_symbols(neural_output["entities"])
                symbols.extend(entity_symbols)

            # Extract confidence-based symbols
            confidence_symbols = self._map_confidence_to_symbols(neural_output.get("confidence", 0.0))
            symbols.extend(confidence_symbols)

            # Extract visual feature symbols (for image inputs)
            if "visual_features" in neural_output:
                visual_symbols = self._map_visual_features_to_symbols(neural_output["visual_features"])
                symbols.extend(visual_symbols)

            # Apply grounding rules to refine symbols
            grounded_symbols = self.grounding_rules.apply_grounding(symbols, neural_output)

            # Remove duplicates while preserving order
            unique_symbols = list(dict.fromkeys(grounded_symbols))

            logger.info(f"Mapped neural output to {len(unique_symbols)} symbols: {unique_symbols}")
            return unique_symbols

        except Exception as e:
            logger.error(f"Symbol mapping failed: {str(e)}")
            return ["MAPPING_ERROR"]

    def _map_label_to_symbol(self, neural_output: Dict[str, Any]) -> str:
        """
        Map the primary classification label to a symbol

        Args:
            neural_output: Neural model output

        Returns:
            Primary symbol or None
        """
        label = neural_output.get("label", "")
        confidence = neural_output.get("confidence", 0.0)

        # Only map if confidence is above threshold
        if confidence < self.confidence_threshold:
            return "LOW_CONFIDENCE_PREDICTION"

        # Domain-specific label mapping
        label_mappings = {
            # Safety/toxicity labels
            "TOXICITY_WARNING": "SUBSTANCE_TOXIC",
            "ALLERGY_ALERT": "ALLERGY_RISK",
            "RISK_IDENTIFIED": "HIGH_RISK",
            "SAFETY_POSITIVE": "SUBSTANCE_SAFE",

            # Medical labels
            "MEDICAL_POSITIVE": "MEDICAL_CONDITION",
            "MEDICAL_NEGATIVE": "MEDICAL_CONDITION",

            # General sentiment to risk mapping
            "GENERAL_NEGATIVE": "MEDIUM_RISK",
            "GENERAL_POSITIVE": "LOW_RISK",

            # Visual classifications
            "MOLECULAR_STRUCTURE": "SUBSTANCE_UNKNOWN",
            "CHEMICAL_COMPOUND": "SUBSTANCE_UNKNOWN",
            "BIOLOGICAL_ENTITY": "MEDICAL_CONDITION",
            "SAFETY_HAZARD": "HIGH_RISK"
        }

        return label_mappings.get(label, "UNKNOWN_CLASSIFICATION")

    def _map_entities_to_symbols(self, entities: List[Dict[str, Any]]) -> List[str]:
        """
        Map extracted entities to symbols

        Args:
            entities: List of entities from entity recognition

        Returns:
            List of entity-based symbols
        """
        entity_symbols = []

        for entity in entities:
            entity_type = entity.get("label", entity.get("entity_group", ""))
            confidence = entity.get("score", 0.0)

            if confidence >= self.confidence_threshold:
                # Map entity types to symbols
                entity_mappings = {
                    "CHEMICAL": "SUBSTANCE_UNKNOWN",
                    "MOLECULAR": "SUBSTANCE_UNKNOWN",
                    "MEDICAL": "MEDICAL_CONDITION",
                    "SAFETY": "HIGH_RISK",
                    "PERSON": "ENTITY_PERSON",
                    "PER": "ENTITY_PERSON",
                    "ORG": "ENTITY_ORGANIZATION",
                    "ORGANIZATION": "ENTITY_ORGANIZATION"
                }

                symbol = entity_mappings.get(entity_type, "ENTITY_UNKNOWN")
                entity_symbols.append(symbol)

        return entity_symbols

    def _map_confidence_to_symbols(self, confidence: float) -> List[str]:
        """
        Map confidence levels to symbols for reasoning about uncertainty

        Args:
            confidence: Confidence score from neural model

        Returns:
            List of confidence-related symbols
        """
        confidence_symbols = []

        if confidence >= 0.9:
            confidence_symbols.append("HIGH_CONFIDENCE")
        elif confidence >= 0.7:
            confidence_symbols.append("MEDIUM_CONFIDENCE")
        elif confidence >= 0.5:
            confidence_symbols.append("LOW_CONFIDENCE")
        else:
            confidence_symbols.append("VERY_LOW_CONFIDENCE")

        return confidence_symbols

    def _map_visual_features_to_symbols(self, visual_features: Dict[str, Any]) -> List[str]:
        """
        Map visual features to symbols for image-based reasoning

        Args:
            visual_features: Visual features extracted from image

        Returns:
            List of visual feature symbols
        """
        visual_symbols = []

        # Map complexity features
        complexity = visual_features.get("complexity", {})
        edge_density = complexity.get("edge_density", 0)

        if edge_density > 50:
            visual_symbols.append("COMPLEX_STRUCTURE")
        elif edge_density > 20:
            visual_symbols.append("MODERATE_STRUCTURE")
        else:
            visual_symbols.append("SIMPLE_STRUCTURE")

        # Map color analysis
        color_analysis = visual_features.get("color_analysis", {})
        brightness = color_analysis.get("mean_brightness", 128)

        if brightness > 200:
            visual_symbols.append("BRIGHT_IMAGE")
        elif brightness < 50:
            visual_symbols.append("DARK_IMAGE")

        return visual_symbols

    def get_symbol_properties(self, symbol: str) -> Dict[str, Any]:
        """
        Get properties and metadata for a symbol

        Args:
            symbol: Symbol name

        Returns:
            Symbol properties and metadata
        """
        return self.symbol_registry.get(symbol, {
            "type": "unknown",
            "properties": [],
            "relationships": [],
            "domain": "general"
        })

    def get_related_symbols(self, symbol: str) -> List[str]:
        """
        Get symbols related to the given symbol

        Args:
            symbol: Symbol to find relations for

        Returns:
            List of related symbols
        """
        symbol_props = self.get_symbol_properties(symbol)
        symbol_domain = symbol_props.get("domain", "")

        # Find symbols in the same domain
        related = []
        for sym_name, sym_props in self.symbol_registry.items():
            if sym_name != symbol and sym_props.get("domain") == symbol_domain:
                related.append(sym_name)

        return related

    def validate_symbol_mapping(self, symbols: List[str], neural_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that symbol mapping is consistent and reasonable

        Args:
            symbols: Generated symbols
            neural_output: Original neural output

        Returns:
            Validation results
        """
        validation = {
            "valid": True,
            "warnings": [],
            "confidence_check": True,
            "consistency_check": True
        }

        # Check confidence consistency
        neural_confidence = neural_output.get("confidence", 0.0)
        has_high_conf_symbol = any("HIGH_CONFIDENCE" in symbol for symbol in symbols)

        if neural_confidence < 0.7 and has_high_conf_symbol:
            validation["warnings"].append("High confidence symbol with low neural confidence")
            validation["confidence_check"] = False

        # Check for conflicting symbols
        has_safe = any("SAFE" in symbol for symbol in symbols)
        has_toxic = any("TOXIC" in symbol for symbol in symbols)

        if has_safe and has_toxic:
            validation["warnings"].append("Conflicting safety symbols detected")
            validation["consistency_check"] = False

        validation["valid"] = validation["confidence_check"] and validation["consistency_check"]

        return validation
