"""
Example: Using OmniRouter in Your Agents
=========================================

This script demonstrates practical patterns for using the router.
Run: python examples/router_usage_example.py
"""

import asyncio
from typing import Optional


def example_1_simple_extraction():
    """Pattern 1: Simple cost-optimized extraction (no agent code changes)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simple Cost-Optimized Extraction")
    print("=" * 70)
    
    from agents.routing_strategy import get_optimal_model
    from agents.llm import gateway_kwargs
    from langchain_openai import ChatOpenAI
    
    # Simulated patient lab data
    lab_report = """
    Patient ID: P1019
    Date: 2024-01-15
    
    Lab Results:
    - Hemoglobin: 14.2 g/dL (normal)
    - White Blood Cell: 7.3 K/uL (normal)
    - Platelets: 210 K/uL (normal)
    """
    
    # Get optimal model for extraction (will choose cheap model by default)
    messages = [
        {"role": "system", "content": "Extract structured lab values"},
        {"role": "user", "content": lab_report}
    ]
    
    model_id, model_alias = get_optimal_model(
        messages=messages,
        agent_type="extractor_agent",
        budget_priority="cost",
    )
    
    print(f"Selected model: {model_id}")
    print(f"  Using alias: {model_alias} (for LiteLLM gateway)")
    print(f"  Estimated cost: Very low (cheap model)")
    
    # Use with ChatOpenAI (same as before)
    # llm = ChatOpenAI(model=model_alias, **gateway_kwargs())
    # result = llm.invoke(f"Extract lab values from: {lab_report}")


def example_2_quality_validation():
    """Pattern 2: Quality-focused validation for critical decisions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Quality-Focused Clinical Validation")
    print("=" * 70)
    
    from agents.routing_strategy import get_optimal_model
    from langchain_openai import ChatOpenAI
    from agents.llm import gateway_kwargs
    
    # Critical clinical data that needs careful validation
    clinical_note = """
    DIAGNOSIS: Acute Myocardial Infarction
    CLINICAL FINDINGS: ST elevation in leads II, III, aVF
    TROPONIN: 2.4 ng/mL (elevated - normal <0.04)
    RISK ASSESSMENT: Critical - immediate intervention required
    """
    
    messages = [
        {"role": "system", "content": "Validate critical clinical diagnosis"},
        {"role": "user", "content": clinical_note}
    ]
    
    # For critical clinical validation, use best available model
    model_id, model_alias = get_optimal_model(
        messages=messages,
        agent_type="ehr_validation_agent",
        budget_priority="quality",
    )
    
    print(f"Selected model: {model_id}")
    print(f"  Using alias: {model_alias}")
    print(f"  Reasoning: Clinical validation → requires accuracy → GPT-4o")
    
    # Use with ChatOpenAI
    # llm = ChatOpenAI(model=model_alias, **gateway_kwargs())


def example_3_factory_function():
    """Pattern 3: Using the factory function (minimal code changes)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Factory Function for Agent Creation")
    print("=" * 70)
    
    from agents.router_integration import create_router_aware_llm
    
    # Before: llm = ChatOpenAI(model="gpt-4o-mini", **gateway_kwargs())
    
    # After (same interface, smart routing under the hood):
    llm = create_router_aware_llm(
        agent_type="extractor_agent",
        budget_priority="cost",
        temperature=0.5,
    )
    
    print(f"Created LLM: {llm.model}")
    print("  ✓ Model automatically selected based on agent type")
    print("  ✓ Cost-optimized routing enabled")
    print("  ✓ Drop-in replacement for ChatOpenAI(model='...')")


def example_4_router_aware_agent():
    """Pattern 4: RouterAwareAgent base class for advanced control."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: RouterAwareAgent - Per-Request Routing")
    print("=" * 70)
    
    from agents.router_integration import RouterAwareAgent
    
    class LabExtractor(RouterAwareAgent):
        def __init__(self):
            super().__init__(agent_type="extractor_agent")
        
        def extract_labs(self, lab_text: str, is_critical: bool = False) -> dict:
            """Extract labs, using better model for critical cases."""
            
            # Simple labs = cheap model
            budget = "quality" if is_critical else "cost"
            
            llm = self.get_llm_for_task(
                task_description="Extract lab values from report",
                task_content=lab_text,
                budget_priority=budget,
            )
            
            # In real code: use llm to extract
            return {"model_used": llm.model, "critical": is_critical}
    
    # Usage
    extractor = LabExtractor()
    
    simple_labs = "WBC: 7.3, Hgb: 14.2, Plt: 210"
    critical_labs = "Troponin: 2.4 (ELEVATED), K: 2.1 (CRITICAL), pH: 7.25"
    
    simple_result = extractor.extract_labs(simple_labs, is_critical=False)
    critical_result = extractor.extract_labs(critical_labs, is_critical=True)
    
    print(f"Simple labs model: {simple_result['model_used']} (cheap)")
    print(f"Critical labs model: {critical_result['model_used']} (best)")
    print("  ✓ Same agent, different models based on task difficulty")


def example_5_monitoring():
    """Pattern 5: Monitor router performance."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Router Performance Monitoring")
    print("=" * 70)
    
    from agents.router_monitoring import RouterMonitor, RouterDebugger
    
    # Create monitor
    monitor = RouterMonitor()
    
    # Get performance summary
    summary = monitor.get_summary()
    
    print(f"Total router calls: {summary['total_calls']}")
    print(f"Error rate: {summary['error_rate']:.2%}")
    print(f"Available models: {summary['models_available']}")
    
    print("\nPer-provider metrics:")
    for provider, metrics in summary['provider_metrics'].items():
        if metrics['is_available']:
            print(f"  {provider}: {metrics['calls']} calls, "
                  f"{metrics['avg_latency']:.0f}ms avg latency")
    
    # Get detailed explanation
    print("\nRouter decision explanation:")
    RouterDebugger.explain_selection(
        agent_type="extractor_agent",
        budget_priority="cost",
        show_all_scores=False,
    )


def example_6_force_model():
    """Pattern 6: Force specific model (testing/debugging)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Force Specific Model (for testing)")
    print("=" * 70)
    
    from agents.routing_strategy import get_optimal_model
    
    messages = [{"role": "user", "content": "test"}]
    
    # Normal routing
    _, alias_auto = get_optimal_model(
        messages=messages,
        agent_type="extractor_agent",
        budget_priority="cost",
    )
    print(f"Auto-selected: {alias_auto}")
    
    # Force a specific model
    _, alias_forced = get_optimal_model(
        messages=messages,
        agent_type="extractor_agent",
        force_model="gpt-4o",
    )
    print(f"Forced override: {alias_forced}")
    print("  ✓ Useful for A/B testing, debugging, or temporary overrides")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("OmniRouter Usage Examples")
    print("=" * 70)
    
    example_1_simple_extraction()
    example_2_quality_validation()
    example_3_factory_function()
    example_4_router_aware_agent()
    example_5_monitoring()
    example_6_force_model()
    
    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Choose a pattern that fits your use case")
    print("  2. Apply to one agent and test")
    print("  3. Monitor performance: python -m agents.router_monitoring summary")
    print("  4. Scale to other agents")
    print("  5. Adjust model profiles if needed")


if __name__ == "__main__":
    main()
