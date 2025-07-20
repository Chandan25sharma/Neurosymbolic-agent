from typing import Dict, List, Any, Optional
import logging
import json
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class TraceLogger:
    """
    Logs and stores reasoning traces for debugging, analysis, and explanation purposes.
    Maintains a record of the complete neurosymbolic reasoning process.
    """

    def __init__(self, log_directory: str = "data/traces"):
        """
        Initialize the trace logger

        Args:
            log_directory: Directory to store trace logs
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        # In-memory trace storage for current session
        self.session_traces = {}
        self.max_memory_traces = 100  # Limit memory usage

        logger.info(f"TraceLogger initialized with directory: {log_directory}")

    def log_reasoning_trace(self, neural_output: Dict[str, Any],
                           symbols: List[str],
                           reasoning_chains: List[Dict[str, Any]]) -> str:
        """
        Log a complete reasoning trace

        Args:
            neural_output: Original neural network output
            symbols: Extracted symbols
            reasoning_chains: List of reasoning chains

        Returns:
            Unique trace ID
        """
        try:
            # Generate unique trace ID
            trace_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now()

            # Create comprehensive trace record
            trace_record = {
                "trace_id": trace_id,
                "timestamp": timestamp.isoformat(),
                "session_info": {
                    "timestamp": timestamp.isoformat(),
                    "trace_type": "neurosymbolic_reasoning"
                },
                "neural_phase": {
                    "input_type": self._determine_input_type(neural_output),
                    "model_output": neural_output,
                    "processing_time": None,  # Could be added if timing is tracked
                    "confidence_score": neural_output.get("confidence", 0.0)
                },
                "symbol_extraction_phase": {
                    "symbols_extracted": symbols,
                    "symbol_count": len(symbols),
                    "extraction_method": "grounded_mapping",
                    "grounding_applied": True
                },
                "reasoning_phase": {
                    "reasoning_chains": reasoning_chains,
                    "chain_count": len(reasoning_chains),
                    "total_steps": sum(len(chain.get("steps", [])) for chain in reasoning_chains),
                    "reasoning_types": list(set(chain.get("chain_type", "") for chain in reasoning_chains))
                },
                "analysis_metadata": {
                    "complexity_score": self._calculate_complexity_score(symbols, reasoning_chains),
                    "confidence_distribution": self._analyze_confidence_distribution(reasoning_chains),
                    "domain_coverage": self._analyze_domain_coverage(symbols, reasoning_chains),
                    "rule_usage": self._analyze_rule_usage(reasoning_chains)
                }
            }

            # Store in memory
            self._store_in_memory(trace_id, trace_record)

            # Persist to disk
            self._persist_trace(trace_id, trace_record)

            logger.info(f"Logged reasoning trace: {trace_id}")
            return trace_id

        except Exception as e:
            logger.error(f"Failed to log reasoning trace: {str(e)}")
            return "log_error"

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a trace by ID

        Args:
            trace_id: Trace identifier

        Returns:
            Trace record or None if not found
        """
        try:
            # Check memory first
            if trace_id in self.session_traces:
                return self.session_traces[trace_id]

            # Check disk
            trace_file = self.log_directory / f"trace_{trace_id}.json"
            if trace_file.exists():
                with open(trace_file, 'r') as f:
                    return json.load(f)

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve trace {trace_id}: {str(e)}")
            return None

    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent traces

        Args:
            limit: Maximum number of traces to return

        Returns:
            List of recent trace records
        """
        try:
            # Get from memory first
            memory_traces = list(self.session_traces.values())
            memory_traces.sort(key=lambda x: x["timestamp"], reverse=True)

            if len(memory_traces) >= limit:
                return memory_traces[:limit]

            # Supplement with disk traces if needed
            remaining = limit - len(memory_traces)
            disk_traces = self._load_recent_disk_traces(remaining)

            all_traces = memory_traces + disk_traces
            return all_traces[:limit]

        except Exception as e:
            logger.error(f"Failed to get recent traces: {str(e)}")
            return []

    def search_traces(self, **criteria) -> List[Dict[str, Any]]:
        """
        Search traces based on criteria

        Args:
            **criteria: Search criteria (e.g., confidence_min, domain, reasoning_type)

        Returns:
            List of matching trace records
        """
        try:
            matching_traces = []

            # Search memory traces
            for trace in self.session_traces.values():
                if self._matches_criteria(trace, criteria):
                    matching_traces.append(trace)

            # Search disk traces (simplified - in production, would use proper indexing)
            disk_traces = self._search_disk_traces(criteria)
            matching_traces.extend(disk_traces)

            # Sort by timestamp (most recent first)
            matching_traces.sort(key=lambda x: x["timestamp"], reverse=True)

            return matching_traces

        except Exception as e:
            logger.error(f"Failed to search traces: {str(e)}")
            return []

    def export_trace(self, trace_id: str, format: str = "json") -> Optional[str]:
        """
        Export a trace in the specified format

        Args:
            trace_id: Trace identifier
            format: Export format ("json", "markdown", "csv")

        Returns:
            Exported trace content or None if error
        """
        try:
            trace = self.get_trace(trace_id)
            if not trace:
                return None

            if format == "json":
                return json.dumps(trace, indent=2)
            elif format == "markdown":
                return self._export_markdown(trace)
            elif format == "csv":
                return self._export_csv(trace)
            else:
                logger.warning(f"Unsupported export format: {format}")
                return None

        except Exception as e:
            logger.error(f"Failed to export trace {trace_id}: {str(e)}")
            return None

    def get_trace_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about logged traces

        Returns:
            Dictionary with trace statistics
        """
        try:
            total_memory = len(self.session_traces)
            total_disk = len(list(self.log_directory.glob("trace_*.json")))

            # Analyze recent traces for patterns
            recent_traces = self.get_recent_traces(50)

            if recent_traces:
                # Confidence statistics
                confidences = [
                    trace["neural_phase"]["confidence_score"]
                    for trace in recent_traces
                ]
                avg_confidence = sum(confidences) / len(confidences)

                # Complexity statistics
                complexities = [
                    trace["analysis_metadata"]["complexity_score"]
                    for trace in recent_traces
                ]
                avg_complexity = sum(complexities) / len(complexities)

                # Domain coverage
                all_domains = []
                for trace in recent_traces:
                    domains = trace["analysis_metadata"]["domain_coverage"]
                    all_domains.extend(domains)

                domain_frequency = {}
                for domain in all_domains:
                    domain_frequency[domain] = domain_frequency.get(domain, 0) + 1

            else:
                avg_confidence = 0
                avg_complexity = 0
                domain_frequency = {}

            return {
                "total_traces": total_memory + total_disk,
                "memory_traces": total_memory,
                "disk_traces": total_disk,
                "recent_analysis": {
                    "traces_analyzed": len(recent_traces),
                    "average_confidence": avg_confidence,
                    "average_complexity": avg_complexity,
                    "domain_frequency": domain_frequency
                },
                "storage_info": {
                    "log_directory": str(self.log_directory),
                    "memory_limit": self.max_memory_traces
                }
            }

        except Exception as e:
            logger.error(f"Failed to get trace statistics: {str(e)}")
            return {}

    def _determine_input_type(self, neural_output: Dict[str, Any]) -> str:
        """Determine the type of input that was processed"""
        if "text" in neural_output:
            return "text"
        elif "visual_features" in neural_output:
            return "image"
        else:
            return "unknown"

    def _calculate_complexity_score(self, symbols: List[str],
                                   reasoning_chains: List[Dict[str, Any]]) -> float:
        """Calculate a complexity score for the reasoning process"""
        # Base complexity from number of symbols
        symbol_complexity = min(len(symbols) / 10, 1.0)

        # Chain complexity
        total_steps = sum(len(chain.get("steps", [])) for chain in reasoning_chains)
        chain_complexity = min(total_steps / 20, 1.0)

        # Type diversity complexity
        chain_types = set(chain.get("chain_type", "") for chain in reasoning_chains)
        type_complexity = min(len(chain_types) / 4, 1.0)

        # Weighted average
        complexity = (symbol_complexity * 0.3 + chain_complexity * 0.5 + type_complexity * 0.2)

        return round(complexity, 3)

    def _analyze_confidence_distribution(self, reasoning_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of confidence scores"""
        if not reasoning_chains:
            return {"mean": 0, "min": 0, "max": 0, "distribution": {}}

        confidences = [chain.get("overall_confidence", 0) for chain in reasoning_chains]

        # Calculate distribution buckets
        distribution = {"high": 0, "medium": 0, "low": 0}
        for conf in confidences:
            if conf >= 0.8:
                distribution["high"] += 1
            elif conf >= 0.5:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return {
            "mean": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "distribution": distribution
        }

    def _analyze_domain_coverage(self, symbols: List[str],
                                reasoning_chains: List[Dict[str, Any]]) -> List[str]:
        """Analyze which domains are covered in the reasoning"""
        domains = set()

        # Extract domains from symbols
        for symbol in symbols:
            if "MEDICAL" in symbol:
                domains.add("medical")
            elif "SUBSTANCE" in symbol or "CHEMICAL" in symbol:
                domains.add("chemical")
            elif "RISK" in symbol:
                domains.add("risk")
            elif "CONFIDENCE" in symbol:
                domains.add("confidence")

        # Extract domains from reasoning chains (would need rule domain info)
        # For now, just return symbol-based domains

        return list(domains)

    def _analyze_rule_usage(self, reasoning_chains: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze which rules were used in reasoning"""
        rule_usage = {}

        for chain in reasoning_chains:
            for step in chain.get("steps", []):
                rule_id = step.get("rule_id", "unknown")
                rule_usage[rule_id] = rule_usage.get(rule_id, 0) + 1

        return rule_usage

    def _store_in_memory(self, trace_id: str, trace_record: Dict[str, Any]) -> None:
        """Store trace in memory with size management"""
        self.session_traces[trace_id] = trace_record

        # Manage memory usage
        if len(self.session_traces) > self.max_memory_traces:
            # Remove oldest traces
            oldest_traces = sorted(
                self.session_traces.items(),
                key=lambda x: x[1]["timestamp"]
            )

            to_remove = len(self.session_traces) - self.max_memory_traces + 10
            for trace_id_to_remove, _ in oldest_traces[:to_remove]:
                del self.session_traces[trace_id_to_remove]

    def _persist_trace(self, trace_id: str, trace_record: Dict[str, Any]) -> None:
        """Persist trace to disk"""
        try:
            trace_file = self.log_directory / f"trace_{trace_id}.json"
            with open(trace_file, 'w') as f:
                json.dump(trace_record, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist trace {trace_id}: {str(e)}")

    def _load_recent_disk_traces(self, limit: int) -> List[Dict[str, Any]]:
        """Load recent traces from disk"""
        try:
            trace_files = list(self.log_directory.glob("trace_*.json"))

            # Sort by modification time
            trace_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            traces = []
            for trace_file in trace_files[:limit]:
                try:
                    with open(trace_file, 'r') as f:
                        trace = json.load(f)
                        traces.append(trace)
                except Exception as e:
                    logger.warning(f"Failed to load trace file {trace_file}: {str(e)}")

            return traces

        except Exception as e:
            logger.error(f"Failed to load disk traces: {str(e)}")
            return []

    def _matches_criteria(self, trace: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if a trace matches search criteria"""
        try:
            # Confidence criteria
            if "confidence_min" in criteria:
                neural_conf = trace["neural_phase"]["confidence_score"]
                if neural_conf < criteria["confidence_min"]:
                    return False

            # Domain criteria
            if "domain" in criteria:
                domains = trace["analysis_metadata"]["domain_coverage"]
                if criteria["domain"] not in domains:
                    return False

            # Reasoning type criteria
            if "reasoning_type" in criteria:
                types = trace["reasoning_phase"]["reasoning_types"]
                if criteria["reasoning_type"] not in types:
                    return False

            return True

        except Exception:
            return False

    def _search_disk_traces(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search disk traces (simplified implementation)"""
        # In a production system, this would use proper indexing
        # For now, just return empty list
        return []

    def _export_markdown(self, trace: Dict[str, Any]) -> str:
        """Export trace as Markdown"""
        md = f"""# Reasoning Trace: {trace['trace_id']}

**Timestamp:** {trace['timestamp']}

## Neural Analysis
- **Input Type:** {trace['neural_phase']['input_type']}
- **Confidence:** {trace['neural_phase']['confidence_score']:.1%}

## Symbol Extraction
- **Symbols Extracted:** {trace['symbol_extraction_phase']['symbol_count']}
- **Symbols:** {', '.join(trace['symbol_extraction_phase']['symbols_extracted'])}

## Reasoning Process
- **Chains Generated:** {trace['reasoning_phase']['chain_count']}
- **Total Steps:** {trace['reasoning_phase']['total_steps']}
- **Reasoning Types:** {', '.join(trace['reasoning_phase']['reasoning_types'])}

## Analysis Metadata
- **Complexity Score:** {trace['analysis_metadata']['complexity_score']}
- **Domains Covered:** {', '.join(trace['analysis_metadata']['domain_coverage'])}
"""
        return md

    def _export_csv(self, trace: Dict[str, Any]) -> str:
        """Export trace as CSV (simplified)"""
        # This would be more complex in a real implementation
        csv_data = f"trace_id,timestamp,confidence,complexity,domains\n"
        csv_data += f"{trace['trace_id']},{trace['timestamp']},{trace['neural_phase']['confidence_score']},{trace['analysis_metadata']['complexity_score']},\"{','.join(trace['analysis_metadata']['domain_coverage'])}\"\n"
        return csv_data
