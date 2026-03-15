"""Shared types for research routing/execution.

These models are intentionally app-agnostic so they can be reused by
other projects (e.g., YouTube summarizer) with different frontend flows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

RouteAction = Literal["chat", "new_image", "image_modify", "enhance", "research", "infographic", "clarify"]
RouteResponseMode = Literal["chat", "analysis", "report"]
KnowledgeBoundary = Literal["llm_ok", "web_preferred", "web_required"]
ResearchProvider = Literal["auto", "brave", "tavily", "both"]
ResearchToolMode = Literal["auto", "safe", "deep", "manual"]
ResearchDepth = Literal["quick", "balanced", "deep"]


@dataclass
class RouteDecision:
    action: RouteAction
    confidence: float
    reasoning: str
    clarification_question: str = ""
    response_mode: RouteResponseMode = "chat"
    export_recommended: bool = False
    knowledge_boundary: KnowledgeBoundary = "llm_ok"


@dataclass
class ResearchPlan:
    objective: str
    queries: list[str]
    provider_mode: ResearchProvider
    tool_mode: ResearchToolMode
    llm_provider: str = ""
    llm_model: str = ""
    entities: list[str] = field(default_factory=list)
    comparison_axes: list[str] = field(default_factory=list)
    freshness_sensitive: bool = False
    requires_source_backed_answer: bool = False
    needs_clarification: bool = False
    clarification_question: str = ""


@dataclass
class ResearchSource:
    name: str
    url: str
    domain: str
    tier: str = "reference"
    providers: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)


@dataclass
class ResearchItem:
    title: str
    url: str
    snippet: str
    published_at: Optional[str] = None


@dataclass
class ResearchBatchResult:
    query: str
    provider: str
    tool: str
    results: list[ResearchItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    latency_ms: int = 0
    rate_limited: bool = False
    retry_count: int = 0


@dataclass
class ResearchRunResult:
    answer: str
    sources: list[ResearchSource]
    status: Literal["ok", "fallback", "error"]
    meta: dict
