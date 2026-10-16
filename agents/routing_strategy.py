"""
Advanced LLM Routing Strategy — Option 1 Extension
====================================================
Extends LiteLLM with intelligent provider selection based on:
  - Request complexity and token requirements
  - Cost optimization per agent type
  - Latency thresholds and performance metrics
  - Prompt characteristics and model capabilities

This layer sits between agents and the LiteLLM gateway,
enabling smart routing decisions without changing existing agent code.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Literal, Optional
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# Model Capabilities & Cost Data
# =============================================================================

class ModelTier(Enum):
    """Model performance and cost tiers."""
    ULTRA_CHEAP = "ultra_cheap"      # Ollama local, ~$0/1M tokens
    CHEAP = "cheap"                  # Gemini Flash, ~$0.075/1M input
    BALANCED = "balanced"            # GPT-4o mini, ~$0.15/1M input
    POWERFUL = "powerful"            # GPT-4o, ~$2.50/1M input
    TURBO = "turbo"                  # GPT-4 Turbo, ~$3/1M input


@dataclass
class ModelProfile:
    """Capabilities and costs for a specific LLM."""
    model_id: str                     # e.g., "openai/gpt-4o-mini"
    model_alias: str                  # e.g., "gpt-5.4-mini"
    provider: str                     # "openai", "gemini", "ollama"
    tier: ModelTier
    
    input_cost_per_mtok: float        # Cost per million input tokens
    output_cost_per_mtok: float       # Cost per million output tokens
    
    max_tokens: int                   # Max context window
    latency_ms: float                 # Typical latency in milliseconds
    
    best_for: list[str]              # ["clinical_review", "extraction", "summarization"]
    requires_api_key: bool = True


# Define available models
MODELS: dict[str, ModelProfile] = {
    "omnirouter-fast": ModelProfile(
        model_id="vertex/gemini-2.0-flash",
        model_alias="vertex/gemini-2.0-flash",
        provider="omnirouter",
        tier=ModelTier.CHEAP,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
        max_tokens=1000000,
        latency_ms=200.0,
        best_for=["extraction", "summarization", "routing", "qa_assistant"],
        requires_api_key=True,
    ),
    "omnirouter-balanced": ModelProfile(
        model_id="vertex/gemini-2.5-flash",
        model_alias="vertex/gemini-2.5-flash",
        provider="omnirouter",
        tier=ModelTier.BALANCED,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
        max_tokens=1000000,
        latency_ms=350.0,
        best_for=["clinical_review", "validation", "ehr_validation_agent", "reporting_agent"],
        requires_api_key=True,
    ),
    "omnirouter-pro": ModelProfile(
        model_id="vertex/gemini-3.1-pro-preview",
        model_alias="vertex/gemini-3.1-pro-preview",
        provider="omnirouter",
        tier=ModelTier.POWERFUL,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
        max_tokens=2000000,
        latency_ms=800.0,
        best_for=["clinical_review", "complex_reasoning", "reporting_agent"],
        requires_api_key=True,
    ),
    "ultra-cheap": ModelProfile(
        model_id="ollama/llama3",
        model_alias="gpt-5.4-mini",
        provider="ollama",
        tier=ModelTier.ULTRA_CHEAP,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
        max_tokens=8192,
        latency_ms=800.0,
        best_for=["extraction", "summarization", "dev_testing"],
        requires_api_key=False,
    ),
    "gemini-flash": ModelProfile(
        model_id="gemini/gemini-3.6-flash",
        model_alias="gpt-5.4-mini",
        provider="gemini",
        tier=ModelTier.CHEAP,
        input_cost_per_mtok=0.075,
        output_cost_per_mtok=0.3,
        max_tokens=200000,
        latency_ms=400.0,
        best_for=["extraction", "summarization", "routing"],
        requires_api_key=True,
    ),
    "gpt-4o-mini": ModelProfile(
        model_id="openai/gpt-4o-mini",
        model_alias="gpt-5.4-mini",
        provider="openai",
        tier=ModelTier.BALANCED,
        input_cost_per_mtok=0.15,
        output_cost_per_mtok=0.6,
        max_tokens=128000,
        latency_ms=300.0,
        best_for=["extraction", "summarization", "validation"],
        requires_api_key=True,
    ),
    "gpt-4o": ModelProfile(
        model_id="openai/gpt-4o",
        model_alias="gpt-5.4-turbo",
        provider="openai",
        tier=ModelTier.POWERFUL,
        input_cost_per_mtok=2.50,
        output_cost_per_mtok=10.0,
        max_tokens=128000,
        latency_ms=250.0,
        best_for=["clinical_review", "reporting", "complex_reasoning"],
        requires_api_key=True,
    ),
}


# =============================================================================
# Request Analyzer
# =============================================================================

@dataclass
class RequestMetadata:
    """Analyzed characteristics of an LLM request."""
    estimated_tokens: int            # Estimated total tokens needed
    complexity_score: float           # 0.0 (simple) to 1.0 (complex)
    requires_accuracy: bool           # Clinical decisions need high accuracy
    is_streaming: bool                # Stream response?
    budget_priority: Literal["cost", "speed", "quality"] = "quality"


class RequestAnalyzer:
    """Analyzes incoming requests to determine optimal routing."""
    
    TOKENS_PER_WORD = 1.3             # Rough estimate
    COMPLEX_KEYWORDS = [
        "diagnose", "clinical", "audit", "validate", "comprehensive",
        "review", "decision", "recommend", "critical", "urgent",
    ]
    
    @staticmethod
    def analyze(
        messages: list[dict],
        agent_type: Optional[str] = None,
        budget_priority: Literal["cost", "speed", "quality"] = "quality",
    ) -> RequestMetadata:
        """Analyze request characteristics to guide routing."""
        
        # Estimate tokens
        total_text = "\n".join(m.get("content", "") for m in messages)
        word_count = len(total_text.split())
        estimated_tokens = int(word_count * RequestAnalyzer.TOKENS_PER_WORD)
        
        # Calculate complexity
        lower_text = total_text.lower()
        complex_keyword_count = sum(
            1 for keyword in RequestAnalyzer.COMPLEX_KEYWORDS
            if keyword in lower_text
        )
        complexity_score = min(1.0, complex_keyword_count * 0.15 + (estimated_tokens / 10000) * 0.2)
        
        # Determine if accuracy is critical
        requires_accuracy = (
            agent_type in ["ehr_validation_agent", "clinical_auditor_workflow"]
            or any(kw in lower_text for kw in ["critical", "clinical", "diagnose"])
        )
        
        return RequestMetadata(
            estimated_tokens=estimated_tokens,
            complexity_score=complexity_score,
            requires_accuracy=requires_accuracy,
            is_streaming=False,  # Can be overridden
            budget_priority=budget_priority,
        )


# =============================================================================
# Omni Router: Intelligent Provider Selection
# =============================================================================

class OmniRouter:
    """
    Intelligent router that selects the best LLM provider based on:
      - Request complexity and token needs
      - Agent type and accuracy requirements
      - Cost vs. speed trade-offs
      - Provider availability and health
    """
    
    def __init__(self):
        """Initialize router with model profiles and performance metrics."""
        self.models = MODELS
        self.metrics = {}  # Track provider performance
        self._init_provider_metrics()
        
        logger.info(
            f"OmniRouter initialized with {len(self.models)} model profiles"
        )
    
    def _init_provider_metrics(self):
        """Initialize per-provider performance tracking."""
        for model_key, profile in self.models.items():
            provider = profile.provider
            if provider not in self.metrics:
                self.metrics[provider] = {
                    "calls": 0,
                    "errors": 0,
                    "total_latency": 0.0,
                    "avg_latency": 0.0,
                    "is_available": self._check_provider_available(provider),
                }
    
    def _check_provider_available(self, provider: str) -> bool:
        """Check if a provider is configured and available.
        
        Returns:
            True if provider has valid configuration and can be used
            False if provider has no key, invalid key, or broken key
        """
        if provider == "omnirouter":
            key = os.getenv("OMNIROUTER_API_KEY", "") or os.getenv("LLM_API_KEY", "")
            base = os.getenv("OMNIROUTER_BASE_URL", "") or os.getenv("LLM_BASE_URL", "")
            return bool(key or base)
        elif provider == "openai":
            key = os.getenv("OPENAI_API_KEY", "")
            # Must be a real OpenAI key (sk-...), not Conduit (sk-cdt-...)
            return bool(key and key.startswith("sk-") and not key.startswith("sk-cdt-"))
        elif provider == "gemini":
            key = os.getenv("GEMINI_API_KEY", "")
            # Gemini keys start with AQ.
            return bool(key and key.startswith("AQ."))
        elif provider == "ollama":
            # Ollama is always available (local/offline)
            return True
        return False
    
    def select_model(
        self,
        messages: list[dict],
        agent_type: Optional[str] = None,
        budget_priority: Literal["cost", "speed", "quality"] = "quality",
        force_model: Optional[str] = None,
    ) -> tuple[ModelProfile, str]:
        """
        Select the best model for a request.
        
        Args:
            messages: Chat messages (list of {"role": "...", "content": "..."})
            agent_type: Name of the calling agent (e.g., "clinical_auditor_workflow")
            budget_priority: Optimization target: "cost", "speed", or "quality"
            force_model: Override router decision (for testing/debugging)
        
        Returns:
            (ModelProfile, model_alias): Selected model and its alias for LiteLLM
        """
        
        # Force override if specified
        if force_model:
            if force_model in self.models:
                profile = self.models[force_model]
                logger.info(f"[ROUTER] Forced model selection: {profile.model_id}")
                return profile, profile.model_alias
            else:
                logger.warning(f"[ROUTER] Forced model '{force_model}' not found; proceeding with auto-select")
        
        # Analyze request
        metadata = RequestAnalyzer.analyze(
            messages,
            agent_type=agent_type,
            budget_priority=budget_priority,
        )
        
        # Score each available model
        scores: dict[str, tuple[float, ModelProfile]] = {}
        
        for model_key, profile in self.models.items():
            # Skip unavailable providers
            if not self.metrics[profile.provider]["is_available"]:
                continue
            
            # Skip models that can't handle token requirements
            if metadata.estimated_tokens > profile.max_tokens:
                continue
            
            score = self._score_model(
                profile,
                metadata,
                agent_type=agent_type,
                budget_priority=budget_priority,
            )
            scores[model_key] = (score, profile)
        
        if not scores:
            # Fallback to Ollama if nothing else works
            logger.warning("[ROUTER] No models matched criteria; falling back to Ollama")
            profile = self.models["ultra-cheap"]
            return profile, profile.model_alias
        
        # Select highest-scoring model
        best_key = max(scores.keys(), key=lambda k: scores[k][0])
        score, best_profile = scores[best_key]
        
        logger.info(
            f"[ROUTER] Selected {best_profile.model_id} (score: {score:.2f}, "
            f"tokens: {metadata.estimated_tokens}, complexity: {metadata.complexity_score:.2f})"
        )
        
        return best_profile, best_profile.model_alias
    
    def _score_model(
        self,
        profile: ModelProfile,
        metadata: RequestMetadata,
        agent_type: Optional[str] = None,
        budget_priority: Literal["cost", "speed", "quality"] = "quality",
    ) -> float:
        """
        Score a model for this specific request.
        Higher score = better choice.
        """
        score = 0.0
        
        # 1. Agent type matching
        if agent_type and profile.best_for:
            if agent_type in profile.best_for:
                score += 3.0
            elif any(agent_type.lower().startswith(bf.lower()) for bf in profile.best_for):
                score += 1.5
        
        # 2. Accuracy requirement
        if metadata.requires_accuracy:
            # High-tier models get bonus for accuracy-critical tasks
            if profile.tier in [ModelTier.POWERFUL, ModelTier.TURBO]:
                score += 2.0
            elif profile.tier == ModelTier.BALANCED:
                score += 1.0
        
        # 3. Budget priority
        if budget_priority == "cost":
            cost_score = (
                3.0 * (1.0 - (profile.input_cost_per_mtok / 10.0))  # Normalize to 10 $/MTok max
            )
            score += cost_score
        elif budget_priority == "speed":
            speed_score = 3.0 * (1.0 - (profile.latency_ms / 1000.0))  # Normalize to 1000ms
            score += speed_score
        else:  # quality (default)
            # Higher tier = better quality
            quality_map = {
                ModelTier.ULTRA_CHEAP: 1.0,
                ModelTier.CHEAP: 2.0,
                ModelTier.BALANCED: 2.5,
                ModelTier.POWERFUL: 3.0,
                ModelTier.TURBO: 3.5,
            }
            score += quality_map.get(profile.tier, 2.0)
        
        # 4. Provider health/availability
        provider_metrics = self.metrics[profile.provider]
        if provider_metrics["calls"] > 0:
            error_rate = provider_metrics["errors"] / provider_metrics["calls"]
            score *= (1.0 - error_rate)  # Penalize unreliable providers
        
        return score
    
    def record_call(
        self,
        model_key: str,
        latency_ms: float,
        success: bool = True,
    ):
        """Record metrics for a model call."""
        if model_key not in self.models:
            return
        
        profile = self.models[model_key]
        provider = profile.provider
        metrics = self.metrics[provider]
        
        metrics["calls"] += 1
        if not success:
            metrics["errors"] += 1
        
        metrics["total_latency"] += latency_ms
        metrics["avg_latency"] = metrics["total_latency"] / metrics["calls"]
        
        if metrics["calls"] % 10 == 0:
            logger.debug(
                f"[ROUTER] {provider} metrics: "
                f"{metrics['calls']} calls, "
                f"{metrics['errors']} errors, "
                f"avg latency: {metrics['avg_latency']:.1f}ms"
            )
    
    def get_router_status(self) -> dict:
        """Get current router status for monitoring."""
        return {
            "models_available": len([m for m in self.metrics.values() if m["is_available"]]),
            "provider_metrics": self.metrics,
            "model_profiles": {
                k: {
                    "provider": v.provider,
                    "tier": v.tier.value,
                    "best_for": v.best_for,
                }
                for k, v in self.models.items()
            }
        }


# =============================================================================
# Global Router Instance
# =============================================================================

_router_instance: Optional[OmniRouter] = None


def get_router() -> OmniRouter:
    """Get or create the global router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = OmniRouter()
    return _router_instance


def get_optimal_model(
    messages: list[dict],
    agent_type: Optional[str] = None,
    budget_priority: Literal["cost", "speed", "quality"] = "quality",
    force_model: Optional[str] = None,
) -> tuple[str, str]:
    """
    Convenience function to get optimal model for a request.
    
    Returns:
        (model_id, model_alias): Real model ID and alias for the gateway
    
    Example:
        model_id, model_alias = get_optimal_model(
            messages=[{"role": "user", "content": "Extract patient data..."}],
            agent_type="extractor_agent",
            budget_priority="cost",
        )
        # Use model_alias with ChatOpenAI: ChatOpenAI(model=model_alias)
    """
    router = get_router()
    profile, alias = router.select_model(
        messages,
        agent_type=agent_type,
        budget_priority=budget_priority,
        force_model=force_model,
    )
    return profile.model_id, alias


def get_embeddings_model() -> str:
    """
    Get embeddings model from OmniRouter configuration.
    
    Returns:
        Embedding model name configured for OmniRouter (e.g. "gemini/gemini-embedding-001")
    """
    return os.getenv("OMNIROUTER_EMBEDDING_MODEL", "gemini/gemini-embedding-001")
