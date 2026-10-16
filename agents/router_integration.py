"""
Router Integration Examples — How to Use OmniRouter in Agents
==============================================================

This module shows how to integrate the OmniRouter into your existing
agents to enable intelligent LLM selection without changing core logic.
"""

from typing import Optional, Literal, TYPE_CHECKING
from langchain_openai import ChatOpenAI
from agents.llm import gateway_kwargs
from agents.routing_strategy import get_optimal_model

if TYPE_CHECKING:
    from langchain_openai import OpenAIEmbeddings


# =============================================================================
# Pattern 1: Automatic Model Selection (Recommended)
# =============================================================================
# Use this for simple agent tasks that benefit from automatic routing.

def create_router_aware_llm(
    agent_type: str,
    budget_priority: Literal["cost", "speed", "quality"] = "quality",
    temperature: float = 0.7,
    **kwargs
) -> ChatOpenAI:
    """
    Factory function to create ChatOpenAI with router-selected model.
    
    Usage:
        # In your agent
        llm = create_router_aware_llm(
            agent_type="extractor_agent",
            budget_priority="cost",  # Use cheap models for extraction
        )
        
        response = llm.invoke("Extract patient data...")
    
    Args:
        agent_type: Name of the calling agent (used to find best model)
        budget_priority: "cost" (cheap models) | "speed" (fast models) | "quality" (best models)
        temperature: LLM temperature (0-1)
        **kwargs: Additional ChatOpenAI kwargs (top_p, etc.)
    
    Returns:
        ChatOpenAI instance with optimally selected model
    """
    
    # Note: At this point we don't have messages yet, so we use agent_type
    # to select a reasonable default model. For request-specific routing,
    # see Pattern 2 below.
    
    # Map agent type to likely model tier
    tier_hints = {
        "extractor_agent": "cost",
        "ehr_validation_agent": "quality",
        "clinical_auditor_workflow": "quality",
        "completeness_agent": "quality",
        "reporting_agent": "quality",
        "translation_normalization_agent": "cost",
        "monitor_agent": "cost",
    }
    
    final_priority = budget_priority
    if agent_type in tier_hints and budget_priority == "quality":
        # Allow agent-specific defaults
        priority_hint = tier_hints.get(agent_type, "quality")
        if priority_hint != "quality":
            final_priority = priority_hint
    
    # Create a dummy message to analyze (since we don't have real messages yet)
    dummy_messages = [{"role": "user", "content": f"Agent task: {agent_type}"}]
    
    _, model_alias = get_optimal_model(
        messages=dummy_messages,
        agent_type=agent_type,
        budget_priority=final_priority,
    )
    
    return ChatOpenAI(
        model=model_alias,
        temperature=temperature,
        **gateway_kwargs(),
        **kwargs,
    )


# =============================================================================
# Pattern 2: Per-Request Routing (Advanced)
# =============================================================================
# Use this when you want to route different requests differently
# (e.g., simple extraction vs. complex validation in same agent).

class RouterAwareAgent:
    """
    Base class for agents that use per-request routing.
    
    Example:
        class MyExtractor(RouterAwareAgent):
            def __init__(self):
                super().__init__(agent_type="extractor_agent")
            
            async def extract(self, patient_data: str) -> dict:
                # Automatically selects model based on data complexity
                llm = self.get_llm_for_task(
                    task_description="Extract lab values and vital signs",
                    task_content=patient_data,
                )
                
                result = llm.invoke(f"Extract from: {patient_data}")
                return result
    """
    
    def __init__(self, agent_type: str):
        """Initialize router-aware agent."""
        self.agent_type = agent_type
        self.gateway_kwargs = gateway_kwargs()
    
    def get_llm_for_task(
        self,
        task_description: str = "",
        task_content: str = "",
        budget_priority: Literal["cost", "speed", "quality"] = "quality",
        temperature: float = 0.7,
    ) -> ChatOpenAI:
        """
        Get LLM with model selected based on actual task.
        
        Args:
            task_description: What the task is (helps with complexity analysis)
            task_content: The actual content/data to process
            budget_priority: Optimization preference
            temperature: LLM temperature
        
        Returns:
            ChatOpenAI with optimally selected model
        """
        
        # Build message list for router analysis
        messages = [
            {"role": "system", "content": task_description},
            {"role": "user", "content": task_content or task_description},
        ]
        
        _, model_alias = get_optimal_model(
            messages=messages,
            agent_type=self.agent_type,
            budget_priority=budget_priority,
        )
        
        return ChatOpenAI(
            model=model_alias,
            temperature=temperature,
            **self.gateway_kwargs,
        )


# =============================================================================
# Pattern 3: Cost-Conscious Extraction (Real Example)
# =============================================================================

class CostOptimizedExtractor(RouterAwareAgent):
    """
    Example extractor that uses cheap models by default,
    but upgrades to better models for complex/critical tasks.
    """
    
    def __init__(self):
        super().__init__(agent_type="extractor_agent")
    
    def extract_patient_labs(self, labs_text: str) -> dict:
        """Extract lab values using cheap model."""
        llm = self.get_llm_for_task(
            task_description="Extract structured lab values from medical reports",
            task_content=labs_text,
            budget_priority="cost",  # Use cheapest model that works
        )
        
        # Use with your extraction prompt...
        return {}
    
    def validate_extracted_data(self, extracted_data: dict, original_text: str) -> bool:
        """Validate extraction using more capable model."""
        llm = self.get_llm_for_task(
            task_description="Validate that extracted data matches original source",
            task_content=original_text,
            budget_priority="quality",  # Use best model for accuracy
        )
        
        # Use with your validation prompt...
        return True


# =============================================================================
# Pattern 4: Performance-Optimized Agent (Real Example)
# =============================================================================

class PerformanceOptimizedAgent(RouterAwareAgent):
    """
    Example agent that prioritizes response speed for user-facing tasks.
    """
    
    def __init__(self):
        super().__init__(agent_type="qa_assistant")
    
    def answer_user_question(self, question: str, context: str) -> str:
        """Answer quickly, optimize for latency."""
        llm = self.get_llm_for_task(
            task_description="Answer a user question based on clinical context",
            task_content=context + "\nQuestion: " + question,
            budget_priority="speed",  # Fast response preferred
        )
        
        # Use with your QA prompt...
        return ""


# =============================================================================
# Embeddings Helper
# =============================================================================

def create_embeddings(
    model: str = "text-embedding-3-small",
) -> "OpenAIEmbeddings":
    """
    Create OpenAI embeddings with gateway routing.
    
    Note: Embeddings always use OpenAI (text-embedding-3-small) because:
    - Gemini embeddings have API format incompatibilities
    - OpenAI embeddings are reliable for RAG/vector search
    - Cost is minimal compared to chat completions
    
    Usage:
        from agents.router_integration import create_embeddings
        
        embeddings = create_embeddings()
        
        # Use with FAISS or other vector DBs:
        vector_store = FAISS.from_documents(
            docs,
            embeddings,
        )
    
    Returns:
        OpenAIEmbeddings instance routed through gateway
    """
    from langchain_openai import OpenAIEmbeddings
    
    return OpenAIEmbeddings(
        model=model,
        **gateway_kwargs(),
    )


# =============================================================================
# Example Usage
# =============================================================================

"""
To integrate OmniRouter into an existing agent:

1. SIMPLE INTEGRATION (no code changes to agent logic):
   
   Before:
       llm = ChatOpenAI(model="gpt-4o-mini", **gateway_kwargs())
   
   After:
       llm = create_router_aware_llm(agent_type="my_agent")
   
   ✓ Done! Model now auto-selects based on agent type.

2. ADVANCED INTEGRATION (per-request routing):
   
   Before:
       class MyAgent:
           def __init__(self):
               self.llm = ChatOpenAI(model="gpt-4o-mini", **gateway_kwargs())
   
   After:
       class MyAgent(RouterAwareAgent):
           def __init__(self):
               super().__init__(agent_type="my_agent")
           
           def run_task(self, data):
               llm = self.get_llm_for_task(
                   task_description="Describe what you're doing",
                   task_content=data,
               )
               # Use llm instead of self.llm

3. BUDGET-AWARE INTEGRATION:
   
   # In main.py or config:
   COST_MODE = os.getenv("LLM_COST_MODE", "quality")  # or "cost" or "speed"
   
   # Then pass to agents:
   llm = create_router_aware_llm(
       agent_type="extractor_agent",
       budget_priority=COST_MODE,
   )

4. FORCE SPECIFIC MODEL (for testing/debugging):
   
   model_id, model_alias = get_optimal_model(
       messages=[{"role": "user", "content": "test"}],
       agent_type="test_agent",
       force_model="gpt-4o",  # Override router decision
   )

5. MONITOR ROUTER PERFORMANCE:
   
   from agents.routing_strategy import get_router
   
   router = get_router()
   status = router.get_router_status()
   print(f"Provider metrics: {status['provider_metrics']}")
"""

if __name__ == "__main__":
    # Example usage
    print("OmniRouter Integration Examples")
    print("=" * 60)
    
    # Example 1: Simple factory function
    print("\n1. Simple agent with automatic routing:")
    print("   llm = create_router_aware_llm('extractor_agent', budget_priority='cost')")
    
    # Example 2: Router-aware agent class
    print("\n2. Advanced per-request routing:")
    extractor = CostOptimizedExtractor()
    print(f"   Created: {extractor.__class__.__name__}")
    
    # Example 3: Check available models
    from agents.routing_strategy import get_router
    router = get_router()
    status = router.get_router_status()
    print(f"\n3. Available models: {status['models_available']}")
    for provider, metrics in status['provider_metrics'].items():
        print(f"   {provider}: available={metrics['is_available']}")
