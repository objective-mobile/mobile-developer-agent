# Researcher Agent

## Overview
The Researcher agent executes a research request by searching the web, reading URLs, and querying the local knowledge base. It returns a free-text findings report.

## Responsibilities
- Execute web searches based on the research request or plan
- Fetch and read content from specific URLs
- Query the local knowledge base for existing documentation
- Synthesise findings into a coherent report

## Tools Used
| Tool | Purpose |
|------|---------|
| `web_search` | Search the web for relevant information |
| `read_url` | Fetch and read content from a specific URL |
| `knowledge_search` | Query the local vector knowledge base |

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| `request` | `str` | Research question or topic (may be a formatted `ResearchPlan`) |

## Output
| Type | Description |
|------|-------------|
| `str` | Free-text research findings report |

## Entry Point
```python
from agents.research import run_researcher

findings: str = run_researcher("What are the best practices for Android DataStore?")
```

## Prompt Source
The system prompt is loaded from `config.RESEARCHER_PROMPT`.

## Retry Behaviour
Automatically retries up to 3 times on OpenAI rate limit errors (HTTP 429).
