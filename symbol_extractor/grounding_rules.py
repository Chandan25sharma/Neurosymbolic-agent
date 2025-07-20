from typing import Dict, List, Any, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class GroundingRules:
    """
    Applies grounding rules to refine symbol mapping from neural outputs.
    Grounding connects raw neural predictions to meaningful symbolic representations.
    """

    def __init__(self):
        """Initialize grounding rules"""
        self.context_rules = self._initialize_context_rules()
        self.refinement_rules = self._initialize_refinement_rules()
        self.domain_knowledge = self._initialize_domain_knowledge()

        logger.info("GroundingRules initialized")

    def _initialize_context_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Initialize context-based grounding rules

        Returns:
            Dictionary of context rules by domain
        """
        return {
            "medical": [
                {
                    "condition": lambda text: any(word in text.lower() for word in ["patient", "treatment", "diagnosis"]),
                    "action": "enhance_medical_context",
                    "priority": 1
                },
                {
                    "condition": lambda text: any(word in text.lower() for word in ["allergy", "allergic", "reaction"]),
                    "action": "emphasize_allergy_risk",
                    "priority": 2
                }
            ],
            "chemical": [
                {
                    "condition": lambda text: re.search(r'\b[A-Z][a-z]*\d+\b', text),
                    "action": "identify_chemical_formula",
                    "priority": 1
                },
                {
                    "condition": lambda text: any(word in text.lower() for word in ["toxic", "hazardous", "dangerous"]),
                    "action": "enhance_toxicity_warning",
                    "priority": 2
                }
            ],
            "safety": [
                {
                    "condition": lambda text: any(word in text.lower() for word in ["warning", "caution", "danger"]),
                    "action": "elevate_risk_level",
                    "priority": 1
                },
                {
                    "condition": lambda text: any(word in text.lower() for word in ["safe", "approved", "tested"]),
                    "action": "confirm_safety_status",
                    "priority": 1
                }
            ]
        }

    def _initialize_refinement_rules(self) -> List[Dict[str, Any]]:
        """
        Initialize symbol refinement rules

        Returns:
            List of refinement rules
        """
        return [
            {
                "name": "confidence_refinement",
                "condition": lambda symbols, output: output.get("confidence", 0) < 0.6,
                "action": self._apply_low_confidence_refinement,
                "description": "Refine symbols for low confidence predictions"
            },
            {
                "name": "entity_consistency",
                "condition": lambda symbols, output: len(symbols) > 1,
                "action": self._apply_entity_consistency,
                "description": "Ensure consistency between multiple entities"
            },
            {
                "name": "domain_coherence",
                "condition": lambda symbols, output: True,  # Always apply
                "action": self._apply_domain_coherence,
                "description": "Ensure symbols are coherent within their domain"
            },
            {
                "name": "risk_escalation",
                "condition": lambda symbols, output: any("TOXIC" in s or "DANGER" in s for s in symbols),
                "action": self._apply_risk_escalation,
                "description": "Escalate risk levels for dangerous substances"
            }
        ]

    def _initialize_domain_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize domain-specific knowledge for grounding

        Returns:
            Dictionary of domain knowledge
        """
        return {
            "chemical_safety": {
                "known_toxic": ["benzene", "asbestos", "mercury", "lead"],
                "known_safe": ["water", "oxygen", "nitrogen", "carbon dioxide"],
                "warning_indicators": ["skull", "crossbones", "biohazard", "toxic"],
                "safety_indicators": ["fda approved", "generally recognized as safe", "food grade"]
            },
            "medical_knowledge": {
                "serious_conditions": ["cancer", "heart disease", "stroke", "diabetes"],
                "common_allergies": ["peanut", "shellfish", "dairy", "gluten"],
                "emergency_keywords": ["anaphylaxis", "severe", "emergency", "critical"],
                "mild_conditions": ["cold", "headache", "minor cut", "bruise"]
            },
            "risk_assessment": {
                "high_risk_indicators": ["fatal", "lethal", "severe", "critical", "emergency"],
                "medium_risk_indicators": ["caution", "warning", "moderate", "significant"],
                "low_risk_indicators": ["mild", "minor", "slight", "minimal"]
            }
        }

    def apply_grounding(self, symbols: List[str], neural_output: Dict[str, Any]) -> List[str]:
        """
        Apply grounding rules to refine symbol mapping

        Args:
            symbols: Initial symbols from mapping
            neural_output: Original neural network output

        Returns:
            Refined list of symbols
        """
        try:
            # Start with original symbols
            refined_symbols = symbols.copy()

            # Apply context-based grounding
            context_refined = self._apply_context_grounding(refined_symbols, neural_output)

            # Apply refinement rules
            for rule in self.refinement_rules:
                if rule["condition"](context_refined, neural_output):
                    context_refined = rule["action"](context_refined, neural_output)

            # Apply domain knowledge
            final_symbols = self._apply_domain_knowledge_grounding(context_refined, neural_output)

            logger.info(f"Grounding refined {len(symbols)} -> {len(final_symbols)} symbols")
            return final_symbols

        except Exception as e:
            logger.error(f"Grounding failed: {str(e)}")
            return symbols  # Return original symbols if grounding fails

    def _apply_context_grounding(self, symbols: List[str], neural_output: Dict[str, Any]) -> List[str]:
        """Apply context-based grounding rules"""
        refined_symbols = symbols.copy()
        text = neural_output.get("text", "")

        if not text:
            return refined_symbols

        # Apply context rules for each domain
        for domain, rules in self.context_rules.items():
            for rule in sorted(rules, key=lambda x: x["priority"]):
                if rule["condition"](text):
                    refined_symbols = self._execute_context_action(
                        rule["action"], refined_symbols, neural_output
                    )

        return refined_symbols

    def _execute_context_action(self, action: str, symbols: List[str],
                               neural_output: Dict[str, Any]) -> List[str]:
        """Execute a specific context action"""
        if action == "enhance_medical_context":
            # Add medical context symbol
            if "MEDICAL_CONDITION" not in symbols:
                symbols.append("MEDICAL_CONTEXT")

        elif action == "emphasize_allergy_risk":
            # Replace general risk with allergy-specific risk
            symbols = [s for s in symbols if "MEDIUM_RISK" not in s and "LOW_RISK" not in s]
            if "ALLERGY_RISK" not in symbols:
                symbols.append("ALLERGY_RISK")

        elif action == "identify_chemical_formula":
            # Add chemical structure symbol
            if "CHEMICAL_STRUCTURE_DETECTED" not in symbols:
                symbols.append("CHEMICAL_STRUCTURE_DETECTED")

        elif action == "enhance_toxicity_warning":
            # Elevate to toxic substance
            symbols = [s if "SUBSTANCE_UNKNOWN" not in s else "SUBSTANCE_TOXIC" for s in symbols]

        elif action == "elevate_risk_level":
            # Increase risk level
            risk_mapping = {
                "LOW_RISK": "MEDIUM_RISK",
                "MEDIUM_RISK": "HIGH_RISK"
            }
            symbols = [risk_mapping.get(s, s) for s in symbols]

        elif action == "confirm_safety_status":
            # Confirm safety
            symbols = [s if "SUBSTANCE_UNKNOWN" not in s else "SUBSTANCE_SAFE" for s in symbols]

        return symbols

    def _apply_low_confidence_refinement(self, symbols: List[str],
                                       neural_output: Dict[str, Any]) -> List[str]:
        """Refine symbols for low confidence predictions"""
        confidence = neural_output.get("confidence", 0.0)

        if confidence < 0.4:
            # Very low confidence - add uncertainty symbols
            symbols.append("UNCERTAIN_PREDICTION")
            symbols.append("REQUIRES_HUMAN_REVIEW")
        elif confidence < 0.6:
            # Low confidence - add caution symbol
            symbols.append("LOW_CONFIDENCE_RESULT")

        return symbols

    def _apply_entity_consistency(self, symbols: List[str],
                                neural_output: Dict[str, Any]) -> List[str]:
        """Ensure consistency between multiple entities"""
        # Check for conflicting symbols
        has_safe = any("SAFE" in s for s in symbols)
        has_toxic = any("TOXIC" in s for s in symbols)

        if has_safe and has_toxic:
            # Conflict detected - prefer more cautious option
            symbols = [s for s in symbols if "SAFE" not in s]
            symbols.append("CONFLICTING_SAFETY_SIGNALS")

        return symbols

    def _apply_domain_coherence(self, symbols: List[str],
                              neural_output: Dict[str, Any]) -> List[str]:
        """Ensure symbols are coherent within their domain"""
        # Group symbols by domain
        domains = {}
        for symbol in symbols:
            if "MEDICAL" in symbol:
                domains.setdefault("medical", []).append(symbol)
            elif "SUBSTANCE" in symbol or "CHEMICAL" in symbol:
                domains.setdefault("chemical", []).append(symbol)
            elif "RISK" in symbol:
                domains.setdefault("risk", []).append(symbol)
            else:
                domains.setdefault("general", []).append(symbol)

        # Check for coherence within each domain
        coherent_symbols = []
        for domain, domain_symbols in domains.items():
            if len(domain_symbols) > 1:
                # Apply domain-specific coherence rules
                coherent_symbols.extend(self._ensure_domain_coherence(domain, domain_symbols))
            else:
                coherent_symbols.extend(domain_symbols)

        return coherent_symbols

    def _ensure_domain_coherence(self, domain: str, domain_symbols: List[str]) -> List[str]:
        """Ensure coherence within a specific domain"""
        if domain == "risk":
            # Only keep the highest risk level
            risk_hierarchy = ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]
            highest_risk = None

            for symbol in domain_symbols:
                for risk in risk_hierarchy:
                    if risk in symbol:
                        highest_risk = symbol
                        break

            return [highest_risk] if highest_risk else domain_symbols

        return domain_symbols

    def _apply_risk_escalation(self, symbols: List[str],
                             neural_output: Dict[str, Any]) -> List[str]:
        """Escalate risk levels for dangerous substances"""
        # If any toxic/dangerous symbols are present, ensure high risk
        has_danger = any(keyword in " ".join(symbols).lower()
                        for keyword in ["toxic", "danger", "hazard", "harmful"])

        if has_danger:
            # Remove lower risk levels
            symbols = [s for s in symbols if "LOW_RISK" not in s and "MEDIUM_RISK" not in s]

            # Add high risk if not present
            if not any("HIGH_RISK" in s for s in symbols):
                symbols.append("HIGH_RISK")

        return symbols

    def _apply_domain_knowledge_grounding(self, symbols: List[str],
                                        neural_output: Dict[str, Any]) -> List[str]:
        """Apply domain-specific knowledge for final grounding"""
        text = neural_output.get("text", "").lower()
        enhanced_symbols = symbols.copy()

        # Check against known chemical safety knowledge
        chemical_knowledge = self.domain_knowledge["chemical_safety"]

        # Check for known toxic substances
        for toxic_substance in chemical_knowledge["known_toxic"]:
            if toxic_substance in text:
                enhanced_symbols = [s if "SUBSTANCE_UNKNOWN" not in s else "SUBSTANCE_TOXIC"
                                  for s in enhanced_symbols]
                enhanced_symbols.append("KNOWN_TOXIC_SUBSTANCE")
                break

        # Check for known safe substances
        for safe_substance in chemical_knowledge["known_safe"]:
            if safe_substance in text:
                enhanced_symbols = [s if "SUBSTANCE_UNKNOWN" not in s else "SUBSTANCE_SAFE"
                                  for s in enhanced_symbols]
                enhanced_symbols.append("KNOWN_SAFE_SUBSTANCE")
                break

        # Check medical knowledge
        medical_knowledge = self.domain_knowledge["medical_knowledge"]

        # Check for serious medical conditions
        for condition in medical_knowledge["serious_conditions"]:
            if condition in text:
                enhanced_symbols.append("SERIOUS_MEDICAL_CONDITION")
                break

        # Check for emergency keywords
        for keyword in medical_knowledge["emergency_keywords"]:
            if keyword in text:
                enhanced_symbols.append("MEDICAL_EMERGENCY")
                break

        return enhanced_symbols

    def validate_grounding(self, original_symbols: List[str],
                         grounded_symbols: List[str],
                         neural_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that grounding has improved symbol quality

        Args:
            original_symbols: Symbols before grounding
            grounded_symbols: Symbols after grounding
            neural_output: Original neural output

        Returns:
            Validation results
        """
        validation = {
            "improved": True,
            "changes_made": len(grounded_symbols) != len(original_symbols),
            "added_symbols": list(set(grounded_symbols) - set(original_symbols)),
            "removed_symbols": list(set(original_symbols) - set(grounded_symbols)),
            "confidence_appropriate": True,
            "domain_coherent": True
        }

        # Check if confidence-related changes are appropriate
        confidence = neural_output.get("confidence", 0.0)
        has_uncertainty_symbols = any("UNCERTAIN" in s or "LOW_CONFIDENCE" in s
                                    for s in grounded_symbols)

        if confidence < 0.6 and not has_uncertainty_symbols:
            validation["confidence_appropriate"] = False

        # Check domain coherence
        domains_present = set()
        for symbol in grounded_symbols:
            if "MEDICAL" in symbol:
                domains_present.add("medical")
            elif "CHEMICAL" in symbol or "SUBSTANCE" in symbol:
                domains_present.add("chemical")
            elif "RISK" in symbol:
                domains_present.add("risk")

        # Too many domains might indicate incoherence
        if len(domains_present) > 2:
            validation["domain_coherent"] = False

        validation["improved"] = (validation["confidence_appropriate"] and
                                validation["domain_coherent"])

        return validation
