from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class DomainRules:
    """
    Domain-specific reasoning rules for the neurosymbolic framework.
    Defines logical rules for chemical safety, medical assessment, and risk evaluation.
    """

    def __init__(self):
        """Initialize domain-specific rules"""
        self.chemical_safety_rules = self._initialize_chemical_safety_rules()
        self.medical_assessment_rules = self._initialize_medical_assessment_rules()
        self.risk_evaluation_rules = self._initialize_risk_evaluation_rules()
        self.interaction_rules = self._initialize_interaction_rules()

        logger.info("DomainRules initialized")

    def _initialize_chemical_safety_rules(self) -> List[Dict[str, Any]]:
        """Initialize chemical safety reasoning rules"""
        return [
            {
                "id": "toxic_substance_high_risk",
                "description": "If substance is toxic, then risk level is high",
                "conditions": ["SUBSTANCE_TOXIC"],
                "conclusion": "HIGH_RISK_ASSESSMENT",
                "confidence": 0.9,
                "priority": 1,
                "type": "deductive",
                "domain": "chemical_safety"
            },
            {
                "id": "unknown_substance_caution",
                "description": "If substance safety is unknown, then caution is required",
                "conditions": ["SUBSTANCE_UNKNOWN"],
                "conclusion": "CAUTION_REQUIRED",
                "confidence": 0.7,
                "priority": 2,
                "type": "deductive",
                "domain": "chemical_safety"
            },
            {
                "id": "safe_substance_low_risk",
                "description": "If substance is confirmed safe, then risk level is low",
                "conditions": ["SUBSTANCE_SAFE"],
                "conclusion": "LOW_RISK_ASSESSMENT",
                "confidence": 0.8,
                "priority": 1,
                "type": "deductive",
                "domain": "chemical_safety"
            },
            {
                "id": "chemical_structure_analysis_needed",
                "description": "If chemical structure is detected, then analysis is needed",
                "conditions": ["CHEMICAL_STRUCTURE_DETECTED"],
                "conclusion": "CHEMICAL_ANALYSIS_REQUIRED",
                "confidence": 0.8,
                "priority": 2,
                "type": "deductive",
                "domain": "chemical_safety"
            },
            {
                "id": "known_toxic_immediate_action",
                "description": "If known toxic substance is identified, then immediate protective action is required",
                "conditions": ["KNOWN_TOXIC_SUBSTANCE"],
                "conclusion": "IMMEDIATE_PROTECTIVE_ACTION_REQUIRED",
                "confidence": 0.95,
                "priority": 1,
                "type": "deductive",
                "domain": "chemical_safety"
            },
            {
                "id": "multiple_toxic_indicators",
                "description": "If multiple toxicity indicators are present, then elevated risk",
                "conditions": [
                    {
                        "type": "count",
                        "pattern": "TOXIC",
                        "threshold": 2
                    }
                ],
                "conclusion": "ELEVATED_TOXICITY_RISK",
                "confidence": 0.85,
                "priority": 2,
                "type": "inductive",
                "domain": "chemical_safety"
            }
        ]

    def _initialize_medical_assessment_rules(self) -> List[Dict[str, Any]]:
        """Initialize medical assessment reasoning rules"""
        return [
            {
                "id": "allergy_risk_avoidance",
                "description": "If allergy risk is identified, then avoidance is recommended",
                "conditions": ["ALLERGY_RISK"],
                "conclusion": "AVOIDANCE_RECOMMENDED",
                "confidence": 0.9,
                "priority": 1,
                "type": "deductive",
                "domain": "medical"
            },
            {
                "id": "medical_condition_consultation",
                "description": "If medical condition is detected, then consultation is advised",
                "conditions": ["MEDICAL_CONDITION"],
                "conclusion": "MEDICAL_CONSULTATION_ADVISED",
                "confidence": 0.8,
                "priority": 2,
                "type": "deductive",
                "domain": "medical"
            },
            {
                "id": "serious_medical_condition_urgent",
                "description": "If serious medical condition is detected, then urgent attention is needed",
                "conditions": ["SERIOUS_MEDICAL_CONDITION"],
                "conclusion": "URGENT_MEDICAL_ATTENTION_REQUIRED",
                "confidence": 0.95,
                "priority": 1,
                "type": "deductive",
                "domain": "medical"
            },
            {
                "id": "medical_emergency_protocol",
                "description": "If medical emergency is detected, then emergency protocol should be activated",
                "conditions": ["MEDICAL_EMERGENCY"],
                "conclusion": "ACTIVATE_EMERGENCY_PROTOCOL",
                "confidence": 0.98,
                "priority": 1,
                "type": "deductive",
                "domain": "medical"
            },
            {
                "id": "medical_context_enhanced_assessment",
                "description": "If medical context is present, then enhanced assessment is warranted",
                "conditions": ["MEDICAL_CONTEXT"],
                "conclusion": "ENHANCED_MEDICAL_ASSESSMENT_WARRANTED",
                "confidence": 0.7,
                "priority": 3,
                "type": "deductive",
                "domain": "medical"
            },
            {
                "id": "multiple_medical_indicators",
                "description": "If multiple medical indicators are present, then comprehensive evaluation needed",
                "conditions": [
                    {
                        "type": "count",
                        "pattern": "MEDICAL",
                        "threshold": 2
                    }
                ],
                "conclusion": "COMPREHENSIVE_MEDICAL_EVALUATION_NEEDED",
                "confidence": 0.8,
                "priority": 2,
                "type": "inductive",
                "domain": "medical"
            }
        ]

    def _initialize_risk_evaluation_rules(self) -> List[Dict[str, Any]]:
        """Initialize risk evaluation reasoning rules"""
        return [
            {
                "id": "high_risk_immediate_mitigation",
                "description": "If high risk is identified, then immediate mitigation is required",
                "conditions": ["HIGH_RISK"],
                "conclusion": "IMMEDIATE_RISK_MITIGATION_REQUIRED",
                "confidence": 0.9,
                "priority": 1,
                "type": "deductive",
                "domain": "risk"
            },
            {
                "id": "medium_risk_monitoring",
                "description": "If medium risk is identified, then ongoing monitoring is advised",
                "conditions": ["MEDIUM_RISK"],
                "conclusion": "ONGOING_MONITORING_ADVISED",
                "confidence": 0.8,
                "priority": 2,
                "type": "deductive",
                "domain": "risk"
            },
            {
                "id": "low_risk_standard_precautions",
                "description": "If low risk is identified, then standard precautions are sufficient",
                "conditions": ["LOW_RISK"],
                "conclusion": "STANDARD_PRECAUTIONS_SUFFICIENT",
                "confidence": 0.7,
                "priority": 3,
                "type": "deductive",
                "domain": "risk"
            },
            {
                "id": "conflicting_safety_signals_investigation",
                "description": "If conflicting safety signals are detected, then further investigation is needed",
                "conditions": ["CONFLICTING_SAFETY_SIGNALS"],
                "conclusion": "FURTHER_INVESTIGATION_REQUIRED",
                "confidence": 0.85,
                "priority": 2,
                "type": "deductive",
                "domain": "risk"
            },
            {
                "id": "uncertain_prediction_expert_review",
                "description": "If prediction is uncertain, then expert review is recommended",
                "conditions": ["UNCERTAIN_PREDICTION"],
                "conclusion": "EXPERT_REVIEW_RECOMMENDED",
                "confidence": 0.8,
                "priority": 2,
                "type": "deductive",
                "domain": "risk"
            },
            {
                "id": "multiple_risk_factors_escalation",
                "description": "If multiple risk factors are present, then risk escalation is warranted",
                "conditions": [
                    {
                        "type": "or",
                        "conditions": ["HIGH_RISK", "SUBSTANCE_TOXIC", "ALLERGY_RISK"]
                    },
                    {
                        "type": "count",
                        "pattern": "RISK",
                        "threshold": 2
                    }
                ],
                "conclusion": "RISK_ESCALATION_WARRANTED",
                "confidence": 0.85,
                "priority": 1,
                "type": "inductive",
                "domain": "risk"
            }
        ]

    def _initialize_interaction_rules(self) -> List[Dict[str, Any]]:
        """Initialize rules for interactions between different domains"""
        return [
            {
                "id": "toxic_substance_medical_risk",
                "description": "If substance is toxic and medical condition is present, then heightened medical risk",
                "conditions": [
                    {
                        "type": "and",
                        "conditions": ["SUBSTANCE_TOXIC", "MEDICAL_CONDITION"]
                    }
                ],
                "conclusion": "HEIGHTENED_MEDICAL_RISK",
                "confidence": 0.9,
                "priority": 1,
                "type": "deductive",
                "domain": "interaction"
            },
            {
                "id": "allergy_unknown_substance_high_caution",
                "description": "If allergy risk exists and substance is unknown, then high caution is needed",
                "conditions": [
                    {
                        "type": "and",
                        "conditions": ["ALLERGY_RISK", "SUBSTANCE_UNKNOWN"]
                    }
                ],
                "conclusion": "HIGH_CAUTION_ALLERGY_UNKNOWN_SUBSTANCE",
                "confidence": 0.85,
                "priority": 1,
                "type": "deductive",
                "domain": "interaction"
            },
            {
                "id": "low_confidence_high_risk_verification",
                "description": "If confidence is low but risk appears high, then verification is critical",
                "conditions": [
                    {
                        "type": "and",
                        "conditions": ["LOW_CONFIDENCE", "HIGH_RISK"]
                    }
                ],
                "conclusion": "CRITICAL_VERIFICATION_NEEDED",
                "confidence": 0.9,
                "priority": 1,
                "type": "deductive",
                "domain": "interaction"
            },
            {
                "id": "human_review_required_complex",
                "description": "If situation involves human factors and high risk, then human review is required",
                "conditions": [
                    {
                        "type": "and",
                        "conditions": ["ENTITY_PERSON", "HIGH_RISK"]
                    }
                ],
                "conclusion": "HUMAN_REVIEW_REQUIRED",
                "confidence": 0.85,
                "priority": 2,
                "type": "deductive",
                "domain": "interaction"
            },
            {
                "id": "organizational_guidance_needed",
                "description": "If organization is involved and risk is present, then organizational guidance is needed",
                "conditions": [
                    {
                        "type": "and",
                        "conditions": ["ENTITY_ORGANIZATION", "MEDIUM_RISK"]
                    }
                ],
                "conclusion": "ORGANIZATIONAL_GUIDANCE_NEEDED",
                "confidence": 0.7,
                "priority": 3,
                "type": "deductive",
                "domain": "interaction"
            }
        ]

    def get_rules_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get rules for a specific domain

        Args:
            domain: Domain name (chemical_safety, medical, risk, interaction)

        Returns:
            List of rules for the specified domain
        """
        domain_mapping = {
            "chemical_safety": self.chemical_safety_rules,
            "medical": self.medical_assessment_rules,
            "risk": self.risk_evaluation_rules,
            "interaction": self.interaction_rules
        }

        return domain_mapping.get(domain, [])

    def get_all_rules(self) -> List[Dict[str, Any]]:
        """
        Get all domain-specific rules

        Returns:
            List of all rules across all domains
        """
        all_rules = []
        all_rules.extend(self.chemical_safety_rules)
        all_rules.extend(self.medical_assessment_rules)
        all_rules.extend(self.risk_evaluation_rules)
        all_rules.extend(self.interaction_rules)

        return all_rules

    def get_rule_by_id(self, rule_id: str) -> Dict[str, Any]:
        """
        Get a specific rule by its ID

        Args:
            rule_id: ID of the rule to retrieve

        Returns:
            Rule dictionary or empty dict if not found
        """
        all_rules = self.get_all_rules()

        for rule in all_rules:
            if rule.get("id") == rule_id:
                return rule

        return {}

    def get_rules_by_priority(self, min_priority: int = 1) -> List[Dict[str, Any]]:
        """
        Get rules with priority >= min_priority

        Args:
            min_priority: Minimum priority level

        Returns:
            List of rules meeting priority criteria
        """
        all_rules = self.get_all_rules()
        return [rule for rule in all_rules if rule.get("priority", 0) >= min_priority]

    def get_domain_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the rule set

        Returns:
            Dictionary with rule statistics
        """
        all_rules = self.get_all_rules()

        # Count by domain
        domain_counts = {}
        for rule in all_rules:
            domain = rule.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Count by type
        type_counts = {}
        for rule in all_rules:
            rule_type = rule.get("type", "unknown")
            type_counts[rule_type] = type_counts.get(rule_type, 0) + 1

        # Average confidence
        confidences = [rule.get("confidence", 0) for rule in all_rules]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "total_rules": len(all_rules),
            "rules_by_domain": domain_counts,
            "rules_by_type": type_counts,
            "average_confidence": avg_confidence,
            "highest_priority_rules": len(self.get_rules_by_priority(1))
        }
