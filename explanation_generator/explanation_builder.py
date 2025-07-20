from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json
from .trace_logger import TraceLogger

logger = logging.getLogger(__name__)

class ExplanationBuilder:
    """
    Builds human-readable explanations from neurosymbolic reasoning chains.
    Converts symbolic reasoning steps into transparent, understandable narratives.
    """

    def __init__(self):
        """Initialize the explanation builder"""
        self.trace_logger = TraceLogger()
        self.explanation_templates = self._initialize_explanation_templates()
        self.symbol_descriptions = self._initialize_symbol_descriptions()
        self.confidence_descriptors = self._initialize_confidence_descriptors()

        logger.info("ExplanationBuilder initialized")

    def _initialize_explanation_templates(self) -> Dict[str, str]:
        """Initialize templates for different types of explanations"""
        return {
            # Primary reasoning templates
            "deductive": "Based on the rule '{rule_description}', since {premises}, we can conclude that {conclusion}.",
            "inductive": "From the pattern observed in {premises}, we can infer that {conclusion}.",
            "abductive": "The best explanation for {premises} is that {conclusion}.",
            "meta": "Considering the reasoning process itself, {conclusion} because {rule_description}.",

            # Confidence-based templates
            "high_confidence": "This conclusion is highly reliable (confidence: {confidence}%).",
            "medium_confidence": "This conclusion is moderately reliable (confidence: {confidence}%).",
            "low_confidence": "This conclusion has limited reliability (confidence: {confidence}%) and should be verified.",

            # Risk assessment templates
            "high_risk": "⚠️ HIGH RISK: {conclusion} - Immediate attention required.",
            "medium_risk": "⚡ MEDIUM RISK: {conclusion} - Monitor and take precautions.",
            "low_risk": "✅ LOW RISK: {conclusion} - Standard precautions sufficient.",

            # Action recommendation templates
            "immediate_action": "🚨 IMMEDIATE ACTION: {conclusion}",
            "consultation": "👨‍⚕️ CONSULTATION: {conclusion}",
            "monitoring": "👁️ MONITORING: {conclusion}",
            "research": "🔍 RESEARCH: {conclusion}",

            # Summary templates
            "chain_summary": "The reasoning process involved {step_count} steps, leading to the conclusion: {final_conclusion}",
            "confidence_summary": "Overall confidence in this analysis: {confidence}% based on {evidence_count} pieces of evidence."
        }

    def _initialize_symbol_descriptions(self) -> Dict[str, str]:
        """Initialize human-readable descriptions for symbols"""
        return {
            # Substance symbols
            "SUBSTANCE_TOXIC": "a toxic substance",
            "SUBSTANCE_SAFE": "a safe substance",
            "SUBSTANCE_UNKNOWN": "a substance with unknown safety profile",

            # Risk symbols
            "HIGH_RISK": "high risk level",
            "MEDIUM_RISK": "medium risk level",
            "LOW_RISK": "low risk level",

            # Medical symbols
            "MEDICAL_CONDITION": "a medical condition",
            "ALLERGY_RISK": "potential allergic reaction risk",
            "SERIOUS_MEDICAL_CONDITION": "a serious medical condition",
            "MEDICAL_EMERGENCY": "a medical emergency",

            # Confidence symbols
            "HIGH_CONFIDENCE": "high confidence in the prediction",
            "MEDIUM_CONFIDENCE": "moderate confidence in the prediction",
            "LOW_CONFIDENCE": "low confidence in the prediction",
            "VERY_LOW_CONFIDENCE": "very low confidence in the prediction",

            # Entity symbols
            "ENTITY_PERSON": "a person",
            "ENTITY_ORGANIZATION": "an organization",

            # Visual symbols
            "COMPLEX_STRUCTURE": "complex visual structure",
            "SIMPLE_STRUCTURE": "simple visual structure",
            "BRIGHT_IMAGE": "bright image",
            "DARK_IMAGE": "dark image",

            # Context symbols
            "MEDICAL_CONTEXT": "medical context",
            "CHEMICAL_STRUCTURE_DETECTED": "chemical structure detected",
            "KNOWN_TOXIC_SUBSTANCE": "known toxic substance",
            "KNOWN_SAFE_SUBSTANCE": "known safe substance"
        }

    def _initialize_confidence_descriptors(self) -> Dict[str, str]:
        """Initialize descriptors for confidence levels"""
        return {
            "very_high": "extremely confident",
            "high": "highly confident",
            "medium_high": "quite confident",
            "medium": "moderately confident",
            "medium_low": "somewhat confident",
            "low": "not very confident",
            "very_low": "very uncertain"
        }

    def build_explanation(self, neural_output: Dict[str, Any],
                         symbols: List[str],
                         reasoning_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a comprehensive explanation from neural output and reasoning chains

        Args:
            neural_output: Original neural network output
            symbols: Extracted symbols
            reasoning_chains: List of reasoning chain dictionaries

        Returns:
            Comprehensive explanation dictionary
        """
        try:
            # Log the reasoning trace
            trace_id = self.trace_logger.log_reasoning_trace(
                neural_output, symbols, reasoning_chains
            )

            # Build different components of the explanation
            neural_explanation = self._explain_neural_output(neural_output)
            symbol_explanation = self._explain_symbol_extraction(symbols, neural_output)
            reasoning_explanation = self._explain_reasoning_chains(reasoning_chains)
            summary = self._build_summary(neural_output, symbols, reasoning_chains)
            recommendations = self._generate_recommendations(reasoning_chains)

            explanation = {
                "trace_id": trace_id,
                "timestamp": datetime.now().isoformat(),
                "neural_analysis": neural_explanation,
                "symbol_analysis": symbol_explanation,
                "reasoning_analysis": reasoning_explanation,
                "summary": summary,
                "recommendations": recommendations,
                "confidence_assessment": self._assess_overall_confidence(reasoning_chains),
                "evidence_strength": self._assess_evidence_strength(neural_output, symbols),
                "explanation_metadata": {
                    "total_reasoning_steps": sum(len(chain["steps"]) for chain in reasoning_chains),
                    "reasoning_chains": len(reasoning_chains),
                    "symbols_extracted": len(symbols),
                    "explanation_type": "neurosymbolic"
                }
            }

            logger.info(f"Built explanation with trace ID: {trace_id}")
            return explanation

        except Exception as e:
            logger.error(f"Failed to build explanation: {str(e)}")
            return self._create_error_explanation(str(e))

    def _explain_neural_output(self, neural_output: Dict[str, Any]) -> Dict[str, Any]:
        """Explain the neural network analysis"""
        input_type = "text" if "text" in neural_output else "image"
        label = neural_output.get("label", "unknown")
        confidence = neural_output.get("confidence", 0.0)

        explanation = {
            "description": f"The neural network analyzed the {input_type} input and classified it as '{label}'.",
            "confidence_description": self._describe_confidence(confidence),
            "model_info": neural_output.get("model_used", "unknown model"),
            "raw_output": {
                "label": label,
                "confidence": f"{confidence:.1%}"
            }
        }

        # Add entities if present
        if "entities" in neural_output:
            entities = neural_output["entities"]
            explanation["entities_found"] = {
                "count": len(entities),
                "description": f"The system identified {len(entities)} entities in the input.",
                "entities": [
                    {
                        "text": entity.get("word", ""),
                        "type": entity.get("label", entity.get("entity_group", "")),
                        "confidence": f"{entity.get('score', 0):.1%}"
                    }
                    for entity in entities
                ]
            }

        # Add visual features if present
        if "visual_features" in neural_output:
            visual = neural_output["visual_features"]
            explanation["visual_analysis"] = {
                "description": "Visual features were extracted from the image.",
                "complexity": visual.get("complexity", {}),
                "color_analysis": visual.get("color_analysis", {})
            }

        return explanation

    def _explain_symbol_extraction(self, symbols: List[str], neural_output: Dict[str, Any]) -> Dict[str, Any]:
        """Explain how symbols were extracted from neural output"""
        explanation = {
            "description": f"The neural output was converted into {len(symbols)} symbolic representations for reasoning.",
            "symbols": []
        }

        for symbol in symbols:
            symbol_desc = self.symbol_descriptions.get(symbol, symbol)
            symbol_info = {
                "symbol": symbol,
                "description": symbol_desc,
                "source": self._identify_symbol_source(symbol, neural_output)
            }
            explanation["symbols"].append(symbol_info)

        return explanation

    def _explain_reasoning_chains(self, reasoning_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Explain the symbolic reasoning process"""
        if not reasoning_chains:
            return {
                "description": "No reasoning chains were generated.",
                "chains": []
            }

        explanation = {
            "description": f"The system applied logical reasoning rules to generate {len(reasoning_chains)} reasoning chains.",
            "chains": []
        }

        for i, chain in enumerate(reasoning_chains):
            chain_explanation = {
                "chain_id": i,
                "type": chain.get("chain_type", "unknown"),
                "conclusion": chain.get("final_conclusion", ""),
                "confidence": f"{chain.get('overall_confidence', 0):.1%}",
                "steps": []
            }

            # Explain each reasoning step
            for step in chain.get("steps", []):
                step_explanation = self._explain_reasoning_step(step)
                chain_explanation["steps"].append(step_explanation)

            explanation["chains"].append(chain_explanation)

        return explanation

    def _explain_reasoning_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Explain a single reasoning step"""
        reasoning_type = step.get("reasoning_type", "deductive")
        rule_description = step.get("rule_description", "")
        premises = step.get("premises", [])
        conclusion = step.get("conclusion", "")
        confidence = step.get("confidence", 0.0)

        # Convert premises to human-readable form
        premise_descriptions = [
            self.symbol_descriptions.get(premise, premise)
            for premise in premises
        ]

        # Select appropriate template
        template = self.explanation_templates.get(reasoning_type,
                                                  self.explanation_templates["deductive"])

        # Generate human-readable explanation
        human_explanation = template.format(
            rule_description=rule_description,
            premises=", ".join(premise_descriptions),
            conclusion=self._humanize_conclusion(conclusion),
            confidence=f"{confidence:.0%}"
        )

        return {
            "step_id": step.get("step_id", 0),
            "reasoning_type": reasoning_type,
            "human_explanation": human_explanation,
            "rule_applied": rule_description,
            "premises": premise_descriptions,
            "conclusion": self._humanize_conclusion(conclusion),
            "confidence": f"{confidence:.1%}"
        }

    def _build_summary(self, neural_output: Dict[str, Any],
                      symbols: List[str],
                      reasoning_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a summary of the entire analysis"""
        # Count total steps
        total_steps = sum(len(chain.get("steps", [])) for chain in reasoning_chains)

        # Get primary conclusion
        primary_conclusion = "No definitive conclusion"
        if reasoning_chains:
            highest_confidence_chain = max(reasoning_chains,
                                         key=lambda c: c.get("overall_confidence", 0))
            primary_conclusion = highest_confidence_chain.get("final_conclusion", "")

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(reasoning_chains)

        summary = {
            "primary_conclusion": self._humanize_conclusion(primary_conclusion),
            "confidence_level": self._describe_confidence(overall_confidence),
            "analysis_type": "neurosymbolic",
            "process_description": f"The analysis involved neural pattern recognition followed by {total_steps} logical reasoning steps across {len(reasoning_chains)} reasoning chains.",
            "key_findings": self._extract_key_findings(reasoning_chains),
            "limitations": self._identify_limitations(neural_output, reasoning_chains)
        }

        return summary

    def _generate_recommendations(self, reasoning_chains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on reasoning chains"""
        recommendations = []

        # Extract all conclusions
        all_conclusions = []
        for chain in reasoning_chains:
            for step in chain.get("steps", []):
                conclusion = step.get("conclusion", "")
                if conclusion:
                    all_conclusions.append({
                        "conclusion": conclusion,
                        "confidence": step.get("confidence", 0.0)
                    })

        # Categorize recommendations
        urgent_actions = []
        consultations = []
        monitoring = []
        research = []

        for conclusion_info in all_conclusions:
            conclusion = conclusion_info["conclusion"]
            confidence = conclusion_info["confidence"]

            if any(keyword in conclusion for keyword in ["IMMEDIATE", "URGENT", "EMERGENCY"]):
                urgent_actions.append({
                    "action": self._humanize_conclusion(conclusion),
                    "priority": "urgent",
                    "confidence": f"{confidence:.1%}"
                })
            elif any(keyword in conclusion for keyword in ["CONSULTATION", "REVIEW", "GUIDANCE"]):
                consultations.append({
                    "action": self._humanize_conclusion(conclusion),
                    "priority": "high",
                    "confidence": f"{confidence:.1%}"
                })
            elif any(keyword in conclusion for keyword in ["MONITORING", "WATCH", "OBSERVE"]):
                monitoring.append({
                    "action": self._humanize_conclusion(conclusion),
                    "priority": "medium",
                    "confidence": f"{confidence:.1%}"
                })
            elif any(keyword in conclusion for keyword in ["RESEARCH", "ANALYSIS", "INVESTIGATION"]):
                research.append({
                    "action": self._humanize_conclusion(conclusion),
                    "priority": "low",
                    "confidence": f"{confidence:.1%}"
                })

        # Add categorized recommendations
        if urgent_actions:
            recommendations.extend(urgent_actions)
        if consultations:
            recommendations.extend(consultations)
        if monitoring:
            recommendations.extend(monitoring)
        if research:
            recommendations.extend(research)

        # If no specific recommendations, add general ones
        if not recommendations:
            recommendations.append({
                "action": "Continue standard monitoring and follow established protocols",
                "priority": "low",
                "confidence": "50%"
            })

        return recommendations[:5]  # Limit to top 5 recommendations

    def _describe_confidence(self, confidence: float) -> str:
        """Convert numerical confidence to descriptive text"""
        if confidence >= 0.9:
            return "Very high confidence - the analysis is very reliable"
        elif confidence >= 0.8:
            return "High confidence - the analysis is reliable"
        elif confidence >= 0.7:
            return "Good confidence - the analysis is generally reliable"
        elif confidence >= 0.6:
            return "Moderate confidence - the analysis should be verified"
        elif confidence >= 0.5:
            return "Low confidence - additional analysis recommended"
        else:
            return "Very low confidence - results are uncertain and require verification"

    def _humanize_conclusion(self, conclusion: str) -> str:
        """Convert symbolic conclusions to human-readable form"""
        # Simple replacement mapping
        replacements = {
            "_": " ",
            "REQUIRED": "is required",
            "NEEDED": "is needed",
            "RECOMMENDED": "is recommended",
            "ADVISED": "is advised",
            "WARRANTED": "is warranted",
            "ASSESSMENT": "assessment",
            "ANALYSIS": "analysis",
            "REVIEW": "review",
            "CONSULTATION": "consultation",
            "MONITORING": "monitoring",
            "CAUTION": "caution",
            "ATTENTION": "attention",
            "ACTION": "action",
            "PROTECTIVE": "protective"
        }

        humanized = conclusion
        for old, new in replacements.items():
            humanized = humanized.replace(old, new)

        # Capitalize first letter and make it a proper sentence
        humanized = humanized.lower().capitalize()

        if not humanized.endswith('.'):
            humanized += '.'

        return humanized

    def _identify_symbol_source(self, symbol: str, neural_output: Dict[str, Any]) -> str:
        """Identify where a symbol came from"""
        if "CONFIDENCE" in symbol:
            return "confidence assessment"
        elif "SUBSTANCE" in symbol:
            return "chemical classification"
        elif "MEDICAL" in symbol:
            return "medical analysis"
        elif "RISK" in symbol:
            return "risk assessment"
        elif "VISUAL" in symbol or "IMAGE" in symbol:
            return "visual analysis"
        elif "ENTITY" in symbol:
            return "entity recognition"
        else:
            return "pattern recognition"

    def _assess_overall_confidence(self, reasoning_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall confidence in the analysis"""
        if not reasoning_chains:
            return {"level": "none", "description": "No reasoning performed"}

        confidences = [chain.get("overall_confidence", 0) for chain in reasoning_chains]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)

        return {
            "average": f"{avg_confidence:.1%}",
            "range": f"{min_confidence:.1%} - {max_confidence:.1%}",
            "description": self._describe_confidence(avg_confidence),
            "variability": "High" if (max_confidence - min_confidence) > 0.3 else "Low"
        }

    def _assess_evidence_strength(self, neural_output: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
        """Assess the strength of evidence"""
        neural_confidence = neural_output.get("confidence", 0.0)
        entity_count = len(neural_output.get("entities", []))
        symbol_count = len(symbols)

        evidence_score = (neural_confidence * 0.5) + (min(entity_count/5, 1) * 0.3) + (min(symbol_count/10, 1) * 0.2)

        strength = "Strong" if evidence_score > 0.7 else "Moderate" if evidence_score > 0.4 else "Weak"

        return {
            "strength": strength,
            "score": f"{evidence_score:.1%}",
            "factors": {
                "neural_confidence": f"{neural_confidence:.1%}",
                "entities_found": entity_count,
                "symbols_extracted": symbol_count
            }
        }

    def _extract_key_findings(self, reasoning_chains: List[Dict[str, Any]]) -> List[str]:
        """Extract key findings from reasoning chains"""
        findings = []

        for chain in reasoning_chains:
            conclusion = chain.get("final_conclusion", "")
            confidence = chain.get("overall_confidence", 0)

            if confidence > 0.7 and conclusion:
                findings.append(self._humanize_conclusion(conclusion))

        return findings[:3]  # Return top 3 findings

    def _identify_limitations(self, neural_output: Dict[str, Any],
                            reasoning_chains: List[Dict[str, Any]]) -> List[str]:
        """Identify limitations of the analysis"""
        limitations = []

        # Check neural confidence
        if neural_output.get("confidence", 0) < 0.7:
            limitations.append("Neural network prediction has moderate to low confidence")

        # Check reasoning confidence
        if reasoning_chains:
            avg_reasoning_confidence = sum(c.get("overall_confidence", 0) for c in reasoning_chains) / len(reasoning_chains)
            if avg_reasoning_confidence < 0.6:
                limitations.append("Symbolic reasoning has limited confidence")

        # Check for conflicting conclusions
        conclusions = [c.get("final_conclusion", "") for c in reasoning_chains]
        if len(set(conclusions)) > len(conclusions) * 0.7:  # Many different conclusions
            limitations.append("Multiple conflicting conclusions suggest uncertainty")

        # Check for uncertainty indicators
        all_text = str(neural_output) + str(reasoning_chains)
        if "UNCERTAIN" in all_text or "UNKNOWN" in all_text:
            limitations.append("Analysis contains uncertain or unknown elements")

        return limitations

    def _calculate_overall_confidence(self, reasoning_chains: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence across all reasoning chains"""
        if not reasoning_chains:
            return 0.0

        confidences = [chain.get("overall_confidence", 0) for chain in reasoning_chains]
        return sum(confidences) / len(confidences)

    def _create_error_explanation(self, error_message: str) -> Dict[str, Any]:
        """Create an explanation when the main process fails"""
        return {
            "trace_id": "error",
            "timestamp": datetime.now().isoformat(),
            "error": True,
            "message": "An error occurred during explanation generation",
            "details": error_message,
            "summary": {
                "primary_conclusion": "Analysis could not be completed due to an error",
                "confidence_level": "No confidence assessment available"
            }
        }
