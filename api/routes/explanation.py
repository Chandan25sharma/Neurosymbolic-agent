from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response models
class ExplanationRequest(BaseModel):
    neural_output: Dict[str, Any]
    symbols: List[str]
    reasoning_chains: List[Dict[str, Any]]
    format: str = "detailed"  # "detailed", "summary", "simple"

class TraceSearchRequest(BaseModel):
    confidence_min: Optional[float] = None
    domain: Optional[str] = None
    reasoning_type: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    limit: int = 10

class ExplanationResponse(BaseModel):
    explanation: Dict[str, Any]
    trace_id: str
    format_used: str
    generation_time: float

class TraceListResponse(BaseModel):
    traces: List[Dict[str, Any]]
    total_count: int
    search_criteria: Dict[str, Any]

@router.post("/generate", response_model=ExplanationResponse)
async def generate_explanation(request: ExplanationRequest):
    """
    Generate a human-readable explanation from reasoning components

    Args:
        request: Explanation generation request

    Returns:
        Generated explanation with trace ID
    """
    try:
        import time
        start_time = time.time()

        from explanation_generator.explanation_builder import ExplanationBuilder

        explanation_builder = ExplanationBuilder()

        # Generate explanation based on format
        logger.info(f"Generating {request.format} explanation")

        if request.format == "detailed":
            explanation = explanation_builder.build_explanation(
                request.neural_output, request.symbols, request.reasoning_chains
            )
        elif request.format == "summary":
            explanation = explanation_builder.build_explanation(
                request.neural_output, request.symbols, request.reasoning_chains
            )
            # Simplify for summary format
            explanation = {
                "trace_id": explanation.get("trace_id"),
                "summary": explanation.get("summary"),
                "recommendations": explanation.get("recommendations", [])[:3],
                "confidence_assessment": explanation.get("confidence_assessment")
            }
        elif request.format == "simple":
            explanation = explanation_builder.build_explanation(
                request.neural_output, request.symbols, request.reasoning_chains
            )
            # Simplify for basic format
            explanation = {
                "trace_id": explanation.get("trace_id"),
                "conclusion": explanation.get("summary", {}).get("primary_conclusion", ""),
                "confidence": explanation.get("confidence_assessment", {}).get("description", ""),
                "key_recommendations": explanation.get("recommendations", [])[:2]
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")

        processing_time = time.time() - start_time

        return ExplanationResponse(
            explanation=explanation,
            trace_id=explanation.get("trace_id", "unknown"),
            format_used=request.format,
            generation_time=processing_time
        )

    except Exception as e:
        logger.error(f"Explanation generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trace/{trace_id}")
async def get_trace_explanation(trace_id: str, format: str = Query("detailed")):
    """
    Get explanation for a specific trace

    Args:
        trace_id: Trace identifier
        format: Explanation format (detailed, summary, simple)

    Returns:
        Trace explanation
    """
    try:
        from explanation_generator.trace_logger import TraceLogger
        from explanation_generator.explanation_builder import ExplanationBuilder

        trace_logger = TraceLogger()
        explanation_builder = ExplanationBuilder()

        # Get the trace
        trace = trace_logger.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")

        # Extract components from trace
        neural_output = trace.get("neural_phase", {}).get("model_output", {})
        symbols = trace.get("symbol_extraction_phase", {}).get("symbols_extracted", [])
        reasoning_chains = trace.get("reasoning_phase", {}).get("reasoning_chains", [])

        # Generate explanation in requested format
        if format == "detailed":
            explanation = explanation_builder.build_explanation(neural_output, symbols, reasoning_chains)
        elif format == "summary":
            full_explanation = explanation_builder.build_explanation(neural_output, symbols, reasoning_chains)
            explanation = {
                "trace_id": trace_id,
                "summary": full_explanation.get("summary"),
                "key_findings": full_explanation.get("summary", {}).get("key_findings", []),
                "confidence": full_explanation.get("confidence_assessment"),
                "trace_info": {
                    "timestamp": trace.get("timestamp"),
                    "complexity": trace.get("analysis_metadata", {}).get("complexity_score"),
                    "domains": trace.get("analysis_metadata", {}).get("domain_coverage", [])
                }
            }
        elif format == "simple":
            explanation = {
                "trace_id": trace_id,
                "conclusion": "Analysis completed",
                "confidence": trace.get("analysis_metadata", {}).get("confidence_distribution", {}).get("mean", 0),
                "timestamp": trace.get("timestamp")
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

        return explanation

    except Exception as e:
        logger.error(f"Failed to get trace explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=TraceListResponse)
async def search_traces(request: TraceSearchRequest):
    """
    Search for traces based on criteria

    Args:
        request: Search criteria

    Returns:
        List of matching traces
    """
    try:
        from explanation_generator.trace_logger import TraceLogger

        trace_logger = TraceLogger()

        # Build search criteria
        search_criteria = {}
        if request.confidence_min is not None:
            search_criteria["confidence_min"] = request.confidence_min
        if request.domain:
            search_criteria["domain"] = request.domain
        if request.reasoning_type:
            search_criteria["reasoning_type"] = request.reasoning_type

        # Perform search
        matching_traces = trace_logger.search_traces(**search_criteria)

        # Apply date filters if specified
        if request.date_from or request.date_to:
            filtered_traces = []
            for trace in matching_traces:
                trace_date = trace.get("timestamp", "")

                # Simple date filtering (in production, would use proper date parsing)
                include_trace = True
                if request.date_from and trace_date < request.date_from:
                    include_trace = False
                if request.date_to and trace_date > request.date_to:
                    include_trace = False

                if include_trace:
                    filtered_traces.append(trace)

            matching_traces = filtered_traces

        # Apply limit
        limited_traces = matching_traces[:request.limit]

        # Create summary for each trace
        trace_summaries = []
        for trace in limited_traces:
            summary = {
                "trace_id": trace.get("trace_id"),
                "timestamp": trace.get("timestamp"),
                "input_type": trace.get("neural_phase", {}).get("input_type"),
                "confidence": trace.get("neural_phase", {}).get("confidence_score"),
                "complexity": trace.get("analysis_metadata", {}).get("complexity_score"),
                "domains": trace.get("analysis_metadata", {}).get("domain_coverage", []),
                "reasoning_chains": trace.get("reasoning_phase", {}).get("chain_count", 0)
            }
            trace_summaries.append(summary)

        return TraceListResponse(
            traces=trace_summaries,
            total_count=len(matching_traces),
            search_criteria=search_criteria
        )

    except Exception as e:
        logger.error(f"Trace search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/traces/recent")
async def get_recent_traces(limit: int = Query(10, ge=1, le=50)):
    """
    Get recent traces

    Args:
        limit: Maximum number of traces to return

    Returns:
        List of recent traces
    """
    try:
        from explanation_generator.trace_logger import TraceLogger

        trace_logger = TraceLogger()
        recent_traces = trace_logger.get_recent_traces(limit)

        # Create summaries
        trace_summaries = []
        for trace in recent_traces:
            summary = {
                "trace_id": trace.get("trace_id"),
                "timestamp": trace.get("timestamp"),
                "input_type": trace.get("neural_phase", {}).get("input_type"),
                "confidence": trace.get("neural_phase", {}).get("confidence_score"),
                "conclusion": "Analysis completed",  # Simplified
                "domains": trace.get("analysis_metadata", {}).get("domain_coverage", [])
            }
            trace_summaries.append(summary)

        return {
            "traces": trace_summaries,
            "count": len(trace_summaries),
            "limit_applied": limit
        }

    except Exception as e:
        logger.error(f"Failed to get recent traces: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export/{trace_id}")
async def export_trace(trace_id: str, format: str = Query("json")):
    """
    Export a trace in various formats

    Args:
        trace_id: Trace identifier
        format: Export format (json, markdown, csv)

    Returns:
        Exported trace content
    """
    try:
        from explanation_generator.trace_logger import TraceLogger

        trace_logger = TraceLogger()
        exported_content = trace_logger.export_trace(trace_id, format)

        if exported_content is None:
            raise HTTPException(status_code=404, detail=f"Trace not found or export failed: {trace_id}")

        # Determine content type
        content_types = {
            "json": "application/json",
            "markdown": "text/markdown",
            "csv": "text/csv"
        }

        return {
            "trace_id": trace_id,
            "format": format,
            "content": exported_content,
            "content_type": content_types.get(format, "text/plain")
        }

    except Exception as e:
        logger.error(f"Trace export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_explanation_statistics():
    """
    Get statistics about explanations and traces

    Returns:
        Explanation system statistics
    """
    try:
        from explanation_generator.trace_logger import TraceLogger

        trace_logger = TraceLogger()
        stats = trace_logger.get_trace_statistics()

        # Add additional explanation-specific statistics
        explanation_stats = {
            "trace_statistics": stats,
            "system_info": {
                "explanation_formats_supported": ["detailed", "summary", "simple"],
                "export_formats_supported": ["json", "markdown", "csv"],
                "search_capabilities": [
                    "confidence_threshold", "domain_filter",
                    "reasoning_type", "date_range"
                ]
            },
            "recent_activity": {
                "traces_in_memory": stats.get("memory_traces", 0),
                "traces_on_disk": stats.get("disk_traces", 0),
                "average_confidence": stats.get("recent_analysis", {}).get("average_confidence", 0)
            }
        }

        return explanation_stats

    except Exception as e:
        logger.error(f"Failed to get explanation statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/trace/{trace_id}")
async def delete_trace(trace_id: str):
    """
    Delete a specific trace

    Args:
        trace_id: Trace identifier

    Returns:
        Deletion confirmation
    """
    try:
        from explanation_generator.trace_logger import TraceLogger
        import os
        from pathlib import Path

        trace_logger = TraceLogger()

        # Remove from memory if present
        if trace_id in trace_logger.session_traces:
            del trace_logger.session_traces[trace_id]

        # Remove from disk if present
        trace_file = trace_logger.log_directory / f"trace_{trace_id}.json"
        if trace_file.exists():
            os.remove(trace_file)
            file_deleted = True
        else:
            file_deleted = False

        return {
            "trace_id": trace_id,
            "deleted": True,
            "memory_removed": trace_id in trace_logger.session_traces,
            "file_removed": file_deleted
        }

    except Exception as e:
        logger.error(f"Failed to delete trace {trace_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare")
async def compare_traces(trace_ids: List[str]):
    """
    Compare multiple traces

    Args:
        trace_ids: List of trace identifiers to compare

    Returns:
        Comparison analysis
    """
    try:
        from explanation_generator.trace_logger import TraceLogger

        if len(trace_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 traces required for comparison")

        if len(trace_ids) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 traces can be compared at once")

        trace_logger = TraceLogger()

        # Get all traces
        traces = []
        for trace_id in trace_ids:
            trace = trace_logger.get_trace(trace_id)
            if trace:
                traces.append(trace)
            else:
                raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")

        # Perform comparison analysis
        comparison = {
            "trace_ids": trace_ids,
            "comparison_timestamp": datetime.now().isoformat(),
            "confidence_comparison": [
                {
                    "trace_id": trace.get("trace_id"),
                    "confidence": trace.get("neural_phase", {}).get("confidence_score", 0)
                }
                for trace in traces
            ],
            "complexity_comparison": [
                {
                    "trace_id": trace.get("trace_id"),
                    "complexity": trace.get("analysis_metadata", {}).get("complexity_score", 0)
                }
                for trace in traces
            ],
            "domain_analysis": {
                "unique_domains": list(set(
                    domain
                    for trace in traces
                    for domain in trace.get("analysis_metadata", {}).get("domain_coverage", [])
                )),
                "domain_overlap": len(set(
                    domain
                    for trace in traces
                    for domain in trace.get("analysis_metadata", {}).get("domain_coverage", [])
                )) < sum(len(trace.get("analysis_metadata", {}).get("domain_coverage", [])) for trace in traces)
            },
            "reasoning_comparison": [
                {
                    "trace_id": trace.get("trace_id"),
                    "reasoning_chains": trace.get("reasoning_phase", {}).get("chain_count", 0),
                    "total_steps": trace.get("reasoning_phase", {}).get("total_steps", 0)
                }
                for trace in traces
            ]
        }

        return comparison

    except Exception as e:
        logger.error(f"Trace comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
