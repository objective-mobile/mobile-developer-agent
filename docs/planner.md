# Planner Agent

## Overview
The Planner agent analyses a research request and produces a structured `ResearchPlan` that guides the Researcher agent on what to search for and where.

## Responsibilities
- Decompose a high-level research request into concrete search queries
- Identify relevant sources to check (web, knowledge base, specific URLs)
- Define the expected output format for the research

## Tools Used
| Tool | Purpose |
|------|---------|
| `web_search` | Explore the topic to inform the plan |
| `knowledge_search` | Check what's already available locally |

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| `request` | `str` | High-level research topic or question |

## Output
```python
ResearchPlan(
    goal: str,                  # restated research objective
    search_queries: list[str],  # specific queries to execute
    sources_to_check: list[str],# e.g. ["knowledge_base", "web", "https://..."]
    output_format: str,         # expected format of the final research report
)
```

## Entry Point
```python
from agents.planner import run_planner

plan: ResearchPlan = run_planner("Research Android DataStore best practices")
```

## Prompt Source
The system prompt is loaded from `config.PLANNER_PROMPT`.

## Fallback Behaviour
If the LLM response cannot be parsed as JSON, the planner returns a minimal plan using the original request as the single search query.

## Retry Behaviour
Automatically retries up to 3 times on OpenAI rate limit errors (HTTP 429).
