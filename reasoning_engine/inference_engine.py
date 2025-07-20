from typing import Dict, List, Any, Tuple, Set, Optional
import logging
from dataclasses import dataclass
from .rule_definitions.domain_rules import DomainRules
from .rule_definitions.default_rules import DefaultRules

logger = logging.getLogger(__name__)

@dataclass
class InferenceStep:
    """Represents a single step in the reasoning chain"""
    rule_id: str
    rule_description: str
    premises: List[str]
    conclusion: str
    confidence: float
    reasoning_type: str  # "deductive", "inductive", "abductive"

@dataclass
class ReasoningChain:
    """Represents a complete reasoning chain"""
    steps: List[InferenceStep]
    final_conclusion: str
    overall_confidence: float
    chain_type: str

class InferenceEngine:
    """
    Symbolic reasoning engine that applies logical rules to symbols.
    Provides explainable reasoning chains for AI decisions.
    """

    def __init__(self):
        """Initialize the inference engine"""
        self.domain_rules = DomainRules()
        self.default_rules = DefaultRules()
        self.working_memory = set()
        self.reasoning_trace = []
        self.max_inference_depth = 10
        self.confidence_threshold = 0.3

        logger.info("InferenceEngine initialized")

    def infer(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Main inference method that applies reasoning rules to symbols

        Args:
            symbols: List of symbols to reason about

        Returns:
            List of reasoning chain steps as dictionaries
        """
        try:
            # Clear previous state
            self.working_memory = set(symbols)
            self.reasoning_trace = []

            # Apply inference rules iteratively
            reasoning_chains = self._forward_chain_inference()

            # Convert to dictionary format for API response
            chain_dicts = []
            for chain in reasoning_chains:
                chain_dict = {
                    "chain_id": len(chain_dicts),
                    "final_conclusion": chain.final_conclusion,
                    "overall_confidence": chain.overall_confidence,
                    "chain_type": chain.chain_type,
                    "steps": [
                        {
                            "step_id": i,
                            "rule_id": step.rule_id,
                            "rule_description": step.rule_description,
                            "premises": step.premises,
                            "conclusion": step.conclusion,
                            "confidence": step.confidence,
                            "reasoning_type": step.reasoning_type
                        }
                        for i, step in enumerate(chain.steps)
                    ]
                }
                chain_dicts.append(chain_dict)

            logger.info(f"Generated {len(chain_dicts)} reasoning chains")
            return chain_dicts

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            return []

    def _forward_chain_inference(self) -> List[ReasoningChain]:
        """
        Apply forward chaining inference to derive new conclusions

        Returns:
            List of reasoning chains
        """
        chains = []
        iteration = 0

        while iteration < self.max_inference_depth:
            iteration += 1
            new_facts_derived = False

            # Try to apply each rule
            applicable_rules = self._find_applicable_rules()

            for rule in applicable_rules:
                # Apply the rule and get the inference step
                inference_step = self._apply_rule(rule)

                if inference_step and inference_step.confidence >= self.confidence_threshold:
                    # Add new fact to working memory
                    if inference_step.conclusion not in self.working_memory:
                        self.working_memory.add(inference_step.conclusion)
                        new_facts_derived = True

                    # Create or extend reasoning chain
                    chain = self._create_reasoning_chain([inference_step])
                    chains.append(chain)

            # If no new facts were derived, stop
            if not new_facts_derived:
                break

        # Merge related chains and optimize
        optimized_chains = self._optimize_reasoning_chains(chains)

        return optimized_chains

    def _find_applicable_rules(self) -> List[Dict[str, Any]]:
        """
        Find all rules that can be applied given current working memory

        Returns:
            List of applicable rules
        """
        applicable_rules = []

        # Get rules from domain-specific and default rule sets
        all_rules = self.domain_rules.get_all_rules() + self.default_rules.get_all_rules()

        for rule in all_rules:
            if self._can_apply_rule(rule):
                applicable_rules.append(rule)

        # Sort by priority and confidence
        applicable_rules.sort(key=lambda r: (r.get("priority", 0), r.get("confidence", 0)),
                            reverse=True)

        return applicable_rules

    def _can_apply_rule(self, rule: Dict[str, Any]) -> bool:
        """
        Check if a rule can be applied given current working memory

        Args:
            rule: Rule to check

        Returns:
            True if rule is applicable
        """
        conditions = rule.get("conditions", [])

        # Check if all conditions are satisfied
        for condition in conditions:
            if isinstance(condition, str):
                # Simple string matching
                if condition not in self.working_memory:
                    return False
            elif isinstance(condition, dict):
                # Complex condition checking
                if not self._evaluate_complex_condition(condition):
                    return False

        return True

    def _evaluate_complex_condition(self, condition: Dict[str, Any]) -> bool:
        """
        Evaluate complex logical conditions

        Args:
            condition: Complex condition to evaluate

        Returns:
            True if condition is satisfied
        """
        condition_type = condition.get("type", "")

        if condition_type == "and":
            # All sub-conditions must be true
            sub_conditions = condition.get("conditions", [])
            return all(self._evaluate_condition(cond) for cond in sub_conditions)

        elif condition_type == "or":
            # At least one sub-condition must be true
            sub_conditions = condition.get("conditions", [])
            return any(self._evaluate_condition(cond) for cond in sub_conditions)

        elif condition_type == "not":
            # Negation
            sub_condition = condition.get("condition", "")
            return not self._evaluate_condition(sub_condition)

        elif condition_type == "contains":
            # Check if working memory contains pattern
            pattern = condition.get("pattern", "")
            return any(pattern in fact for fact in self.working_memory)

        elif condition_type == "count":
            # Check count of matching facts
            pattern = condition.get("pattern", "")
            threshold = condition.get("threshold", 1)
            count = sum(1 for fact in self.working_memory if pattern in fact)
            return count >= threshold

        return False

    def _evaluate_condition(self, condition) -> bool:
        """Helper method to evaluate a single condition"""
        if isinstance(condition, str):
            return condition in self.working_memory
        elif isinstance(condition, dict):
            return self._evaluate_complex_condition(condition)
        return False

    def _apply_rule(self, rule: Dict[str, Any]) -> Optional[InferenceStep]:
        """
        Apply a specific rule and create an inference step

        Args:
            rule: Rule to apply

        Returns:
            InferenceStep if rule was successfully applied
        """
        try:
            rule_id = rule.get("id", "unknown")
            rule_description = rule.get("description", "")
            conclusion_template = rule.get("conclusion", "")
            base_confidence = rule.get("confidence", 0.5)
            reasoning_type = rule.get("type", "deductive")

            # Get premises that triggered this rule
            premises = []
            for condition in rule.get("conditions", []):
                if isinstance(condition, str) and condition in self.working_memory:
                    premises.append(condition)

            # Generate conclusion
            conclusion = self._generate_conclusion(conclusion_template, premises)

            # Calculate confidence based on rule confidence and premise strength
            confidence = self._calculate_inference_confidence(base_confidence, premises)

            return InferenceStep(
                rule_id=rule_id,
                rule_description=rule_description,
                premises=premises,
                conclusion=conclusion,
                confidence=confidence,
                reasoning_type=reasoning_type
            )

        except Exception as e:
            logger.error(f"Failed to apply rule {rule.get('id', 'unknown')}: {str(e)}")
            return None

    def _generate_conclusion(self, template: str, premises: List[str]) -> str:
        """
        Generate conclusion from template and premises

        Args:
            template: Conclusion template
            premises: List of premises

        Returns:
            Generated conclusion
        """
        # Simple template substitution
        # In a more advanced system, this would use more sophisticated NLG

        if "{substance}" in template:
            # Find substance-related premise
            substance = "unknown_substance"
            for premise in premises:
                if "SUBSTANCE" in premise:
                    substance = premise
                    break
            template = template.replace("{substance}", substance)

        if "{risk_level}" in template:
            # Find risk-related premise
            risk_level = "unknown_risk"
            for premise in premises:
                if "RISK" in premise:
                    risk_level = premise
                    break
            template = template.replace("{risk_level}", risk_level)

        if "{confidence}" in template:
            # Find confidence-related premise
            confidence_level = "medium_confidence"
            for premise in premises:
                if "CONFIDENCE" in premise:
                    confidence_level = premise
                    break
            template = template.replace("{confidence}", confidence_level)

        return template

    def _calculate_inference_confidence(self, base_confidence: float,
                                      premises: List[str]) -> float:
        """
        Calculate confidence for an inference step

        Args:
            base_confidence: Base confidence from rule
            premises: List of premises used

        Returns:
            Calculated confidence score
        """
        # Start with base confidence
        confidence = base_confidence

        # Adjust based on number and quality of premises
        if len(premises) > 1:
            # Multiple supporting premises increase confidence
            confidence *= 1.1

        # Adjust based on premise certainty
        high_certainty_premises = sum(1 for p in premises if "HIGH_CONFIDENCE" in p)
        low_certainty_premises = sum(1 for p in premises if "LOW_CONFIDENCE" in p)

        if high_certainty_premises > 0:
            confidence *= 1.2
        if low_certainty_premises > 0:
            confidence *= 0.8

        # Ensure confidence stays in valid range
        return max(0.0, min(1.0, confidence))

    def _create_reasoning_chain(self, steps: List[InferenceStep]) -> ReasoningChain:
        """
        Create a reasoning chain from inference steps

        Args:
            steps: List of inference steps

        Returns:
            Complete reasoning chain
        """
        if not steps:
            return ReasoningChain([], "No conclusion", 0.0, "empty")

        # Final conclusion is the conclusion of the last step
        final_conclusion = steps[-1].conclusion

        # Overall confidence is the minimum confidence in the chain
        overall_confidence = min(step.confidence for step in steps)

        # Determine chain type based on step types
        step_types = set(step.reasoning_type for step in steps)
        if len(step_types) == 1:
            chain_type = list(step_types)[0]
        else:
            chain_type = "mixed"

        return ReasoningChain(
            steps=steps,
            final_conclusion=final_conclusion,
            overall_confidence=overall_confidence,
            chain_type=chain_type
        )

    def _optimize_reasoning_chains(self, chains: List[ReasoningChain]) -> List[ReasoningChain]:
        """
        Optimize and merge related reasoning chains

        Args:
            chains: List of reasoning chains to optimize

        Returns:
            Optimized list of reasoning chains
        """
        if not chains:
            return chains

        # Remove duplicate chains
        unique_chains = []
        seen_conclusions = set()

        for chain in chains:
            if chain.final_conclusion not in seen_conclusions:
                unique_chains.append(chain)
                seen_conclusions.add(chain.final_conclusion)

        # Sort by confidence and relevance
        unique_chains.sort(key=lambda c: c.overall_confidence, reverse=True)

        # Keep only top N chains to avoid overwhelming output
        max_chains = 5
        return unique_chains[:max_chains]

    def get_working_memory(self) -> Set[str]:
        """Get current working memory state"""
        return self.working_memory.copy()

    def add_fact(self, fact: str) -> None:
        """Add a fact to working memory"""
        self.working_memory.add(fact)

    def remove_fact(self, fact: str) -> None:
        """Remove a fact from working memory"""
        self.working_memory.discard(fact)

    def clear_memory(self) -> None:
        """Clear working memory"""
        self.working_memory.clear()
        self.reasoning_trace.clear()

    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get statistics about the inference process"""
        return {
            "working_memory_size": len(self.working_memory),
            "reasoning_steps": len(self.reasoning_trace),
            "max_depth": self.max_inference_depth,
            "confidence_threshold": self.confidence_threshold
        }
