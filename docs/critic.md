# Critic Agent

## Overview
The Critic agent reviews research findings and produces a structured verdict (`CritiqueResult`) indicating whether the research is ready to proceed or needs revision.

## Responsibilities
- Evaluate research findings for freshness, completeness, and structure
- Identify gaps and weaknesses in the research
- Provide specific revision requests when the research is insufficient
- Return an APPROVE or REVISE verdict

## Tools Used
| Tool | Purpose |
|------|---------|
| `web_search` | Verify or supplement research findings |
| `read_url` | Fetch content from specific URLs for deeper validation |
| `knowledge_search` | Cross-reference against the local knowledge base |

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| `findings` | `str` | Raw research findings text to evaluate |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `verdict` | `str` | `"APPROVE"` or `"REVISE"` |
| `is_fresh` | `bool` | Whether the research is up-to-date |
| `is_complete` | `bool` | Whether the research covers the topic sufficiently |
| `is_well_structured` | `bool` | Whether the findings are well organised |
| `strengths` | `list[str]` | Positive aspects of the research |
| `gaps` | `list[str]` | Missing or weak areas |
| `revision_requests` | `list[str]` | Specific actions needed before approval |

## Entry Point
```python
from agents.critic import run_critic

critique: CritiqueResult = run_critic(findings="...")
```

## Prompt Source
The system prompt is loaded from `config.CRITIC_PROMPT`.

## Retry Behaviour
Automatically retries up to 3 times on OpenAI rate limit errors (HTTP 429).
