from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response models
class ReasoningRequest(BaseModel):
    symbols: List[str]
    max_depth: int = 10
    confidence_threshold: float = 0.3
    rule_domains: Optional[List[str]] = None

class RuleApplicationRequest(BaseModel):
    symbols: List[str]
    rule_id: str
    validate_application: bool = True

class ReasoningResponse(BaseModel):
    success: bool
    reasoning_chains: List[Dict[str, Any]]
    working_memory: List[str]
    rules_applied: List[str]
    processing_stats: Dict[str, Any]

class RuleInfo(BaseModel):
    rule_id: str
    description: str
    domain: str
    confidence: float
    priority: int
    reasoning_type: str

class RulesResponse(BaseModel):
    rules: List[RuleInfo]
    total_count: int
    domains: List[str]

@router.post("/apply", response_model=ReasoningResponse)
async def apply_reasoning(request: ReasoningRequest):
    """
    Apply symbolic reasoning to a set of symbols

    Args:
        request: Reasoning request with symbols and parameters

    Returns:
        Reasoning response with chains and statistics
    """
    try:
        from reasoning_engine.inference_engine import InferenceEngine

        # Initialize inference engine
        inference_engine = InferenceEngine()

        # Configure engine parameters
        inference_engine.max_inference_depth = request.max_depth
        inference_engine.confidence_threshold = request.confidence_threshold

        # Clear and set working memory
        inference_engine.clear_memory()
        for symbol in request.symbols:
            inference_engine.add_fact(symbol)

        # Apply reasoning
        logger.info(f"Applying reasoning to {len(request.symbols)} symbols")
        reasoning_chains = inference_engine.infer(request.symbols)

        # Get final working memory state
        working_memory = list(inference_engine.get_working_memory())

        # Extract rules that were applied
        rules_applied = []
        for chain in reasoning_chains:
            for step in chain.get("steps", []):
                rule_id = step.get("rule_id", "")
                if rule_id and rule_id not in rules_applied:
                    rules_applied.append(rule_id)

        # Generate processing statistics
        stats = inference_engine.get_inference_statistics()
        stats.update({
            "input_symbols": len(request.symbols),
            "output_chains": len(reasoning_chains),
            "total_steps": sum(len(chain.get("steps", [])) for chain in reasoning_chains),
            "unique_rules_applied": len(rules_applied)
        })

        return ReasoningResponse(
            success=True,
            reasoning_chains=reasoning_chains,
            working_memory=working_memory,
            rules_applied=rules_applied,
            processing_stats=stats
        )

    except Exception as e:
        logger.error(f"Reasoning application failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate-symbols")
async def validate_symbols(symbols: List[str]):
    """
    Validate a set of symbols for reasoning compatibility

    Args:
        symbols: List of symbols to validate

    Returns:
        Validation results
    """
    try:
        from symbol_extractor.symbol_mapper import SymbolMapper

        symbol_mapper = SymbolMapper()

        validation_results = {
            "valid_symbols": [],
            "invalid_symbols": [],
            "warnings": [],
            "suggestions": []
        }

        for symbol in symbols:
            # Check if symbol is recognized
            symbol_props = symbol_mapper.get_symbol_properties(symbol)

            if symbol_props.get("type") != "unknown":
                validation_results["valid_symbols"].append({
                    "symbol": symbol,
                    "properties": symbol_props
                })
            else:
                validation_results["invalid_symbols"].append({
                    "symbol": symbol,
                    "reason": "Symbol not recognized in registry"
                })

        # Check for potential conflicts
        has_safe = any("SAFE" in s for s in symbols)
        has_toxic = any("TOXIC" in s for s in symbols)

        if has_safe and has_toxic:
            validation_results["warnings"].append(
                "Conflicting safety symbols detected (both SAFE and TOXIC)"
            )

        # Provide suggestions
        if len(validation_results["valid_symbols"]) == 0:
            validation_results["suggestions"].append(
                "No valid symbols found. Check symbol naming conventions."
            )
        elif len(validation_results["invalid_symbols"]) > 0:
            validation_results["suggestions"].append(
                "Some symbols are not recognized. Verify symbol names or extend symbol registry."
            )

        return validation_results

    except Exception as e:
        logger.error(f"Symbol validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rules", response_model=RulesResponse)
async def get_available_rules(domain: Optional[str] = None, priority_min: Optional[int] = None):
    """
    Get available reasoning rules, optionally filtered by domain or priority

    Args:
        domain: Optional domain filter
        priority_min: Optional minimum priority filter

    Returns:
        List of available rules
    """
    try:
        from reasoning_engine.rule_definitions.domain_rules import DomainRules
        from reasoning_engine.rule_definitions.default_rules import DefaultRules

        domain_rules = DomainRules()
        default_rules = DefaultRules()

        # Get all rules
        all_rules = domain_rules.get_all_rules() + default_rules.get_all_rules()

        # Apply filters
        filtered_rules = all_rules

        if domain:
            filtered_rules = [r for r in filtered_rules if r.get("domain") == domain]

        if priority_min is not None:
            filtered_rules = [r for r in filtered_rules if r.get("priority", 0) >= priority_min]

        # Convert to response format
        rule_infos = []
        for rule in filtered_rules:
            rule_info = RuleInfo(
                rule_id=rule.get("id", ""),
                description=rule.get("description", ""),
                domain=rule.get("domain", ""),
                confidence=rule.get("confidence", 0.0),
                priority=rule.get("priority", 0),
                reasoning_type=rule.get("type", "")
            )
            rule_infos.append(rule_info)

        # Get unique domains
        domains = list(set(rule.get("domain", "") for rule in all_rules))
        domains = [d for d in domains if d]  # Remove empty strings

        return RulesResponse(
            rules=rule_infos,
            total_count=len(rule_infos),
            domains=domains
        )

    except Exception as e:
        logger.error(f"Failed to get rules: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/apply-rule", response_model=Dict[str, Any])
async def apply_single_rule(request: RuleApplicationRequest):
    """
    Apply a single reasoning rule to symbols

    Args:
        request: Rule application request

    Returns:
        Rule application result
    """
    try:
        from reasoning_engine.rule_definitions.domain_rules import DomainRules
        from reasoning_engine.rule_definitions.default_rules import DefaultRules
        from reasoning_engine.inference_engine import InferenceEngine

        # Get the specific rule
        domain_rules = DomainRules()
        default_rules = DefaultRules()

        rule = domain_rules.get_rule_by_id(request.rule_id)
        if not rule:
            rule = default_rules.get_rule_by_id(request.rule_id)

        if not rule:
            raise HTTPException(status_code=404, detail=f"Rule not found: {request.rule_id}")

        # Initialize inference engine
        inference_engine = InferenceEngine()

        # Set up working memory
        inference_engine.clear_memory()
        for symbol in request.symbols:
            inference_engine.add_fact(symbol)

        # Check if rule can be applied
        can_apply = inference_engine._can_apply_rule(rule)

        if not can_apply:
            return {
                "success": False,
                "rule_id": request.rule_id,
                "reason": "Rule conditions not satisfied by provided symbols",
                "required_conditions": rule.get("conditions", []),
                "provided_symbols": request.symbols
            }

        # Apply the rule
        inference_step = inference_engine._apply_rule(rule)

        if inference_step:
            return {
                "success": True,
                "rule_id": request.rule_id,
                "inference_step": {
                    "rule_description": inference_step.rule_description,
                    "premises": inference_step.premises,
                    "conclusion": inference_step.conclusion,
                    "confidence": inference_step.confidence,
                    "reasoning_type": inference_step.reasoning_type
                },
                "new_fact": inference_step.conclusion
            }
        else:
            return {
                "success": False,
                "rule_id": request.rule_id,
                "reason": "Rule application failed during execution"
            }

    except Exception as e:
        logger.error(f"Single rule application failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trace/{trace_id}")
async def get_reasoning_trace(trace_id: str):
    """
    Get detailed reasoning trace by ID

    Args:
        trace_id: Trace identifier

    Returns:
        Detailed reasoning trace
    """
    try:
        from explanation_generator.trace_logger import TraceLogger

        trace_logger = TraceLogger()
        trace = trace_logger.get_trace(trace_id)

        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")

        return trace

    except Exception as e:
        logger.error(f"Failed to get reasoning trace: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_reasoning_statistics():
    """
    Get statistics about the reasoning system

    Returns:
        System statistics
    """
    try:
        from reasoning_engine.rule_definitions.domain_rules import DomainRules
        from reasoning_engine.rule_definitions.default_rules import DefaultRules
        from explanation_generator.trace_logger import TraceLogger

        domain_rules = DomainRules()
        default_rules = DefaultRules()
        trace_logger = TraceLogger()

        # Get rule statistics
        domain_stats = domain_rules.get_domain_statistics()
        default_stats = default_rules.get_rule_statistics()
        trace_stats = trace_logger.get_trace_statistics()

        return {
            "domain_rules": domain_stats,
            "default_rules": default_stats,
            "trace_statistics": trace_stats,
            "system_info": {
                "total_rules": domain_stats["total_rules"] + default_stats["total_rules"],
                "active_traces": trace_stats.get("memory_traces", 0),
                "system_status": "operational"
            }
        }

    except Exception as e:
        logger.error(f"Failed to get reasoning statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/debug/working-memory")
async def debug_working_memory(symbols: List[str]):
    """
    Debug endpoint to inspect working memory state

    Args:
        symbols: Symbols to add to working memory

    Returns:
        Working memory analysis
    """
    try:
        from reasoning_engine.inference_engine import InferenceEngine

        inference_engine = InferenceEngine()

        # Set up working memory
        inference_engine.clear_memory()
        for symbol in symbols:
            inference_engine.add_fact(symbol)

        # Find applicable rules
        applicable_rules = inference_engine._find_applicable_rules()

        return {
            "working_memory": list(inference_engine.get_working_memory()),
            "applicable_rules": [
                {
                    "rule_id": rule.get("id", ""),
                    "description": rule.get("description", ""),
                    "confidence": rule.get("confidence", 0.0),
                    "conditions": rule.get("conditions", [])
                }
                for rule in applicable_rules[:10]  # Limit to first 10
            ],
            "total_applicable_rules": len(applicable_rules),
            "memory_size": len(inference_engine.get_working_memory())
        }

    except Exception as e:
        logger.error(f"Working memory debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
