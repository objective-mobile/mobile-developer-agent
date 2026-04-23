# Business Analyst Agent

## Overview
The BA agent is a LangGraph ReAct agent that takes a user story and produces a structured feature specification (`SpecOutput`) for Android application development.

## Responsibilities
- Research the domain using web search, local knowledge base, and Context7 API docs
- Generate a structured spec with requirements, acceptance criteria, and complexity estimate
- Revise the spec when HITL (Human-in-the-Loop) feedback is provided

## Tools Used
| Tool | Purpose |
|------|---------|
| `web_search` | Find relevant context, best practices, and similar implementations |
| `knowledge_search` | Search the local knowledge base for internal documentation |
| `context7_search` | Fetch up-to-date Android library/framework API docs |

## Input (from LangGraph state)
| Field | Type | Description |
|-------|------|-------------|
| `user_story` | `str` | The feature request to analyse |
| `hitl_feedback` | `str` (optional) | Reviewer feedback for revision rounds |

## Output (to LangGraph state)
| Field | Type | Description |
|-------|------|-------------|
| `spec` | `SpecOutput` | Structured specification |
| `messages` | `list` | Accumulated agent messages |

## SpecOutput Schema
```python
SpecOutput(
    title: str,                    # concise feature title
    requirements: list[str],       # testable functional/non-functional requirements
    acceptance_criteria: list[str],# verifiable acceptance criteria
    estimated_complexity: str,     # "simple" | "medium" | "complex"
)
```

## Entry Point
```python
from agents.ba import ba_node

# Used as a LangGraph node
graph.add_node("ba", ba_node)
```

## Retry Behaviour
Automatically retries up to 3 times on OpenAI rate limit errors (HTTP 429), with configurable wait time parsed from the error response.
