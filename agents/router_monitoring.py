"""
Router Monitoring & Diagnostics
=================================

Utilities for monitoring router health, tracking model performance,
and debugging routing decisions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from agents.routing_strategy import get_router

logger = logging.getLogger(__name__)


class RouterMonitor:
    """Monitor and log router performance metrics."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize monitor with optional log directory."""
        self.router = get_router()
        self.log_dir = log_dir or Path("./router_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.stats_file = self.log_dir / "router_stats.jsonl"
    
    def log_decision(
        self,
        agent_type: Optional[str],
        selected_model: str,
        budget_priority: str,
        request_tokens: int,
        complexity: float,
    ):
        """Log a routing decision."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "selected_model": selected_model,
            "budget_priority": budget_priority,
            "request_tokens": request_tokens,
            "complexity": complexity,
        }
        
        try:
            with open(self.stats_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to log routing decision: {e}")
    
    def get_summary(self) -> dict:
        """Get summary of router performance."""
        status = self.router.get_router_status()
        
        # Calculate aggregate stats
        total_calls = sum(
            m["calls"] for m in status["provider_metrics"].values()
        )
        total_errors = sum(
            m["errors"] for m in status["provider_metrics"].values()
        )
        
        return {
            "total_calls": total_calls,
            "total_errors": total_errors,
            "error_rate": total_errors / total_calls if total_calls > 0 else 0,
            "models_available": status["models_available"],
            "provider_metrics": status["provider_metrics"],
            "model_profiles": status["model_profiles"],
        }
    
    def print_summary(self):
        """Print formatted summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("OmniRouter Performance Summary")
        print("=" * 70)
        print(f"Total calls: {summary['total_calls']}")
        print(f"Total errors: {summary['total_errors']}")
        print(f"Error rate: {summary['error_rate']:.2%}")
        print(f"Available models: {summary['models_available']}")
        
        print("\nPer-Provider Metrics:")
        print("-" * 70)
        for provider, metrics in summary["provider_metrics"].items():
            print(f"\n{provider.upper()}:")
            print(f"  Available: {metrics['is_available']}")
            print(f"  Calls: {metrics['calls']}")
            print(f"  Errors: {metrics['errors']}")
            if metrics['calls'] > 0:
                print(f"  Avg Latency: {metrics['avg_latency']:.1f}ms")
        
        print("\n" + "=" * 70)


class RouterDebugger:
    """Debug routing decisions and model selection."""
    
    @staticmethod
    def explain_selection(
        agent_type: str,
        budget_priority: str,
        show_all_scores: bool = True,
    ):
        """Explain why a model was selected for these criteria."""
        from agents.routing_strategy import RequestAnalyzer
        
        router = get_router()
        
        # Create a sample request
        sample_messages = [
            {
                "role": "system",
                "content": f"You are a {agent_type}."
            },
            {
                "role": "user",
                "content": "Sample request for analysis.",
            }
        ]
        
        # Analyze
        metadata = RequestAnalyzer.analyze(
            sample_messages,
            agent_type=agent_type,
            budget_priority=budget_priority,
        )
        
        print("\n" + "=" * 70)
        print("Router Selection Explanation")
        print("=" * 70)
        print(f"Agent Type: {agent_type}")
        print(f"Budget Priority: {budget_priority}")
        print(f"\nRequest Analysis:")
        print(f"  Estimated tokens: {metadata.estimated_tokens}")
        print(f"  Complexity score: {metadata.complexity_score:.2f}")
        print(f"  Requires accuracy: {metadata.requires_accuracy}")
        print(f"  Is streaming: {metadata.is_streaming}")
        
        # Get selection
        profile, alias = router.select_model(
            sample_messages,
            agent_type=agent_type,
            budget_priority=budget_priority,
        )
        
        print(f"\nSelected Model:")
        print(f"  Model ID: {profile.model_id}")
        print(f"  Alias: {alias}")
        print(f"  Provider: {profile.provider}")
        print(f"  Tier: {profile.tier.value}")
        print(f"  Max tokens: {profile.max_tokens}")
        print(f"  Latency: {profile.latency_ms:.0f}ms")
        print(f"  Best for: {', '.join(profile.best_for)}")
        
        # Show all scores if requested
        if show_all_scores:
            print(f"\nAll Model Scores:")
            print("-" * 70)
            
            scores = {}
            for model_key, prof in router.models.items():
                if not router.metrics[prof.provider]["is_available"]:
                    continue
                
                if metadata.estimated_tokens > prof.max_tokens:
                    scores[model_key] = ("SKIP", f"tokens too high ({metadata.estimated_tokens}/{prof.max_tokens})")
                    continue
                
                score = router._score_model(
                    prof,
                    metadata,
                    agent_type=agent_type,
                    budget_priority=budget_priority,
                )
                scores[model_key] = (f"{score:.2f}", prof.model_id)
            
            for model_key, (score, model_id) in sorted(
                scores.items(),
                key=lambda x: float(x[1][0]) if x[1][0] != "SKIP" else -999,
                reverse=True
            ):
                indicator = "→" if model_id == profile.model_id else " "
                print(f"{indicator} {model_key:20s} : {str(score):>6s}  ({model_id})")
        
        print("\n" + "=" * 70)
    
    @staticmethod
    def compare_models():
        """Display comparison of all available models."""
        router = get_router()
        
        print("\n" + "=" * 100)
        print("Available LLM Models Comparison")
        print("=" * 100)
        print(f"{'Model':<20} {'Provider':<10} {'Tier':<15} {'Input Cost':<12} {'Output Cost':<12} {'Latency':<10}")
        print("-" * 100)
        
        for model_key, profile in router.models.items():
            available = "✓" if router.metrics[profile.provider]["is_available"] else "✗"
            
            print(
                f"{model_key:<20} "
                f"{profile.provider:<10} "
                f"{profile.tier.value:<15} "
                f"${profile.input_cost_per_mtok:.3f}/MTok  "
                f"${profile.output_cost_per_mtok:.3f}/MTok  "
                f"{profile.latency_ms:.0f}ms"
            )
            print(f"  Best for: {', '.join(profile.best_for)}")
        
        print("\n" + "=" * 100)


def setup_router_monitoring(enable_logging: bool = True):
    """Setup router monitoring (call from main.py startup)."""
    if enable_logging:
        monitor = RouterMonitor()
        logger.info("OmniRouter monitoring enabled")
        return monitor
    return None


if __name__ == "__main__":
    # Example: Run debugging
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "explain":
            agent = sys.argv[2] if len(sys.argv) > 2 else "extractor_agent"
            priority = sys.argv[3] if len(sys.argv) > 3 else "quality"
            RouterDebugger.explain_selection(agent, priority)
        elif sys.argv[1] == "compare":
            RouterDebugger.compare_models()
        elif sys.argv[1] == "summary":
            monitor = RouterMonitor()
            monitor.print_summary()
    else:
        print("Router Monitoring & Diagnostics")
        print("=" * 70)
        print("\nUsage:")
        print("  python -m agents.router_monitoring explain [agent_type] [priority]")
        print("  python -m agents.router_monitoring compare")
        print("  python -m agents.router_monitoring summary")
        print("\nExamples:")
        print("  python -m agents.router_monitoring explain extractor_agent cost")
        print("  python -m agents.router_monitoring explain clinical_auditor_workflow quality")
        print("  python -m agents.router_monitoring compare")
        print("\nPriorities: 'cost', 'speed', 'quality'")
