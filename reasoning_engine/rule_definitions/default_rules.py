from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class DefaultRules:
    """
    Default reasoning rules that provide fallback logic when domain-specific rules don't apply.
    These rules handle general reasoning patterns and edge cases.
    """

    def __init__(self):
        """Initialize default rules"""
        self.general_rules = self._initialize_general_rules()
        self.confidence_rules = self._initialize_confidence_rules()
        self.fallback_rules = self._initialize_fallback_rules()
        self.meta_reasoning_rules = self._initialize_meta_reasoning_rules()

        logger.info("DefaultRules initialized")

    def _initialize_general_rules(self) -> List[Dict[str, Any]]:
        """Initialize general reasoning rules"""
        return [
            {
                "id": "default_high_confidence_trust",
                "description": "If confidence is high, then prediction can be trusted",
                "conditions": ["HIGH_CONFIDENCE"],
                "conclusion": "PREDICTION_TRUSTWORTHY",
                "confidence": 0.8,
                "priority": 3,
                "type": "deductive",
                "domain": "general"
            },
            {
                "id": "default_low_confidence_caution",
                "description": "If confidence is very low, then exercise caution",
                "conditions": ["VERY_LOW_CONFIDENCE"],
                "conclusion": "EXERCISE_CAUTION",
                "confidence": 0.9,
                "priority": 2,
                "type": "deductive",
                "domain": "general"
            },
            {
                "id": "default_unknown_classification_analysis",
                "description": "If classification is unknown, then further analysis is needed",
                "conditions": ["UNKNOWN_CLASSIFICATION"],
                "conclusion": "FURTHER_ANALYSIS_NEEDED",
                "confidence": 0.7,
                "priority": 3,
                "type": "deductive",
                "domain": "general"
            },
            {
                "id": "default_complex_structure_detailed_review",
                "description": "If structure is complex, then detailed review is warranted",
                "conditions": ["COMPLEX_STRUCTURE"],
                "conclusion": "DETAILED_REVIEW_WARRANTED",
                "confidence": 0.6,
                "priority": 4,
                "type": "deductive",
                "domain": "general"
            },
            {
                "id": "default_simple_structure_standard_analysis",
                "description": "If structure is simple, then standard analysis is sufficient",
                "conditions": ["SIMPLE_STRUCTURE"],
                "conclusion": "STANDARD_ANALYSIS_SUFFICIENT",
                "confidence": 0.6,
                "priority": 4,
                "type": "deductive",
                "domain": "general"
            }
        ]

    def _initialize_confidence_rules(self) -> List[Dict[str, Any]]:
        """Initialize confidence-based reasoning rules"""
        return [
            {
                "id": "confidence_medium_verify",
                "description": "If confidence is medium, then verification is recommended",
                "conditions": ["MEDIUM_CONFIDENCE"],
                "conclusion": "VERIFICATION_RECOMMENDED",
                "confidence": 0.7,
                "priority": 3,
                "type": "deductive",
                "domain": "confidence"
            },
            {
                "id": "confidence_low_multiple_sources",
                "description": "If confidence is low, then multiple sources should be consulted",
                "conditions": ["LOW_CONFIDENCE"],
                "conclusion": "CONSULT_MULTIPLE_SOURCES",
                "confidence": 0.8,
                "priority": 2,
                "type": "deductive",
                "domain": "confidence"
            },
            {
                "id": "confidence_uncertain_human_oversight",
                "description": "If prediction is uncertain, then human oversight is valuable",
                "conditions": ["UNCERTAIN_PREDICTION"],
                "conclusion": "HUMAN_OVERSIGHT_VALUABLE",
                "confidence": 0.8,
                "priority": 2,
                "type": "deductive",
                "domain": "confidence"
            },
            {
                "id": "confidence_requires_review_documentation",
                "description": "If human review is required, then document the decision process",
                "conditions": ["REQUIRES_HUMAN_REVIEW"],
                "conclusion": "DOCUMENT_DECISION_PROCESS",
                "confidence": 0.9,
                "priority": 2,
                "type": "deductive",
                "domain": "confidence"
            },
            {
                "id": "confidence_mapping_error_investigate",
                "description": "If mapping error occurred, then investigate the cause",
                "conditions": ["MAPPING_ERROR"],
                "conclusion": "INVESTIGATE_MAPPING_ERROR",
                "confidence": 0.9,
                "priority": 1,
                "type": "deductive",
                "domain": "confidence"
            }
        ]

    def _initialize_fallback_rules(self) -> List[Dict[str, Any]]:
        """Initialize fallback rules for edge cases"""
        return [
            {
                "id": "fallback_no_specific_rules_general_assessment",
                "description": "If no specific rules apply, then perform general assessment",
                "conditions": [
                    {
                        "type": "not",
                        "condition": {
                            "type": "or",
                            "conditions": ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]
                        }
                    }
                ],
                "conclusion": "PERFORM_GENERAL_ASSESSMENT",
                "confidence": 0.5,
                "priority": 5,
                "type": "abductive",
                "domain": "fallback"
            },
            {
                "id": "fallback_unknown_entity_research",
                "description": "If entity is unknown, then research is needed",
                "conditions": ["ENTITY_UNKNOWN"],
                "conclusion": "RESEARCH_UNKNOWN_ENTITY",
                "confidence": 0.6,
                "priority": 4,
                "type": "deductive",
                "domain": "fallback"
            },
            {
                "id": "fallback_bright_image_visual_clarity",
                "description": "If image is bright, then visual clarity is good",
                "conditions": ["BRIGHT_IMAGE"],
                "conclusion": "GOOD_VISUAL_CLARITY",
                "confidence": 0.6,
                "priority": 5,
                "type": "deductive",
                "domain": "fallback"
            },
            {
                "id": "fallback_dark_image_limited_visibility",
                "description": "If image is dark, then visibility may be limited",
                "conditions": ["DARK_IMAGE"],
                "conclusion": "LIMITED_VISIBILITY_POSSIBLE",
                "confidence": 0.6,
                "priority": 5,
                "type": "deductive",
                "domain": "fallback"
            },
            {
                "id": "fallback_no_clear_conclusion_report_uncertainty",
                "description": "If no clear conclusion can be drawn, then report uncertainty",
                "conditions": [
                    {
                        "type": "and",
                        "conditions": [
                            "LOW_CONFIDENCE_RESULT",
                            {
                                "type": "not",
                                "condition": {
                                    "type": "contains",
                                    "pattern": "ASSESSMENT"
                                }
                            }
                        ]
                    }
                ],
                "conclusion": "REPORT_ANALYSIS_UNCERTAINTY",
                "confidence": 0.8,
                "priority": 3,
                "type": "abductive",
                "domain": "fallback"
            }
        ]

    def _initialize_meta_reasoning_rules(self) -> List[Dict[str, Any]]:
        """Initialize meta-reasoning rules that reason about the reasoning process itself"""
        return [
            {
                "id": "meta_multiple_conflicting_conclusions",
                "description": "If multiple conflicting conclusions exist, then reconciliation is needed",
                "conditions": [
                    {
                        "type": "count",
                        "pattern": "ASSESSMENT",
                        "threshold": 2
                    }
                ],
                "conclusion": "RECONCILE_CONFLICTING_CONCLUSIONS",
                "confidence": 0.8,
                "priority": 2,
                "type": "meta",
                "domain": "meta_reasoning"
            },
            {
                "id": "meta_insufficient_information_gather_more",
                "description": "If information is insufficient for reliable conclusion, then gather more data",
                "conditions": [
                    {
                        "type": "and",
                        "conditions": [
                            "LOW_CONFIDENCE_RESULT",
                            {
                                "type": "contains",
                                "pattern": "UNKNOWN"
                            }
                        ]
                    }
                ],
                "conclusion": "GATHER_MORE_INFORMATION",
                "confidence": 0.8,
                "priority": 2,
                "type": "meta",
                "domain": "meta_reasoning"
            },
            {
                "id": "meta_high_stakes_decision_validation",
                "description": "If decision involves high stakes, then additional validation is required",
                "conditions": [
                    {
                        "type": "or",
                        "conditions": [
                            "IMMEDIATE_PROTECTIVE_ACTION_REQUIRED",
                            "URGENT_MEDICAL_ATTENTION_REQUIRED",
                            "ACTIVATE_EMERGENCY_PROTOCOL"
                        ]
                    }
                ],
                "conclusion": "ADDITIONAL_VALIDATION_REQUIRED_HIGH_STAKES",
                "confidence": 0.9,
                "priority": 1,
                "type": "meta",
                "domain": "meta_reasoning"
            },
            {
                "id": "meta_reasoning_trace_complete",
                "description": "If reasoning process is complete, then summarize the decision path",
                "conditions": [
                    {
                        "type": "contains",
                        "pattern": "REQUIRED"
                    }
                ],
                "conclusion": "SUMMARIZE_DECISION_PATH",
                "confidence": 0.7,
                "priority": 4,
                "type": "meta",
                "domain": "meta_reasoning"
            },
            {
                "id": "meta_expertise_boundary_acknowledge",
                "description": "If reasoning reaches expertise boundary, then acknowledge limitations",
                "conditions": [
                    {
                        "type": "or",
                        "conditions": [
                            "EXPERT_REVIEW_RECOMMENDED",
                            "HUMAN_REVIEW_REQUIRED",
                            "ORGANIZATIONAL_GUIDANCE_NEEDED"
                        ]
                    }
                ],
                "conclusion": "ACKNOWLEDGE_REASONING_LIMITATIONS",
                "confidence": 0.8,
                "priority": 3,
                "type": "meta",
                "domain": "meta_reasoning"
            }
        ]

    def get_rules_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get rules for a specific category

        Args:
            category: Category name (general, confidence, fallback, meta_reasoning)

        Returns:
            List of rules for the specified category
        """
        category_mapping = {
            "general": self.general_rules,
            "confidence": self.confidence_rules,
            "fallback": self.fallback_rules,
            "meta_reasoning": self.meta_reasoning_rules
        }

        return category_mapping.get(category, [])

    def get_all_rules(self) -> List[Dict[str, Any]]:
        """
        Get all default rules

        Returns:
            List of all default rules across all categories
        """
        all_rules = []
        all_rules.extend(self.general_rules)
        all_rules.extend(self.confidence_rules)
        all_rules.extend(self.fallback_rules)
        all_rules.extend(self.meta_reasoning_rules)

        return all_rules

    def get_emergency_rules(self) -> List[Dict[str, Any]]:
        """
        Get rules with highest priority for emergency situations

        Returns:
            List of emergency/high-priority rules
        """
        all_rules = self.get_all_rules()
        return [rule for rule in all_rules if rule.get("priority", 0) <= 2]

    def get_rules_by_reasoning_type(self, reasoning_type: str) -> List[Dict[str, Any]]:
        """
        Get rules by reasoning type

        Args:
            reasoning_type: Type of reasoning (deductive, inductive, abductive, meta)

        Returns:
            List of rules of the specified type
        """
        all_rules = self.get_all_rules()
        return [rule for rule in all_rules if rule.get("type") == reasoning_type]

    def get_applicable_fallback_rules(self, working_memory: set) -> List[Dict[str, Any]]:
        """
        Get fallback rules that might be applicable given current working memory

        Args:
            working_memory: Current set of facts in working memory

        Returns:
            List of potentially applicable fallback rules
        """
        applicable_rules = []

        # Check if we have any risk assessments
        has_risk_assessment = any("RISK" in fact for fact in working_memory)
        has_confidence_info = any("CONFIDENCE" in fact for fact in working_memory)
        has_assessment = any("ASSESSMENT" in fact for fact in working_memory)

        # If no risk assessment, include general fallback rules
        if not has_risk_assessment:
            applicable_rules.extend(self.fallback_rules)

        # If no confidence info, include confidence rules
        if not has_confidence_info:
            applicable_rules.extend(self.confidence_rules)

        # Always include meta-reasoning rules for complex situations
        applicable_rules.extend(self.meta_reasoning_rules)

        return applicable_rules

    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the default rule set

        Returns:
            Dictionary with rule statistics
        """
        all_rules = self.get_all_rules()

        # Count by category
        category_counts = {
            "general": len(self.general_rules),
            "confidence": len(self.confidence_rules),
            "fallback": len(self.fallback_rules),
            "meta_reasoning": len(self.meta_reasoning_rules)
        }

        # Count by reasoning type
        type_counts = {}
        for rule in all_rules:
            rule_type = rule.get("type", "unknown")
            type_counts[rule_type] = type_counts.get(rule_type, 0) + 1

        # Count by priority
        priority_counts = {}
        for rule in all_rules:
            priority = rule.get("priority", 0)
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        # Average confidence
        confidences = [rule.get("confidence", 0) for rule in all_rules]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "total_rules": len(all_rules),
            "rules_by_category": category_counts,
            "rules_by_type": type_counts,
            "rules_by_priority": priority_counts,
            "average_confidence": avg_confidence,
            "emergency_rules": len(self.get_emergency_rules())
        }
