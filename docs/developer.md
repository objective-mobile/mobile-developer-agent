# Developer Agent

## Overview
The Developer agent is a LangGraph ReAct agent that takes an approved `SpecOutput` and generates a complete, buildable Android Gradle project on disk using Jetpack Compose and Material3.

## Responsibilities
- Research implementation patterns via web search, knowledge base, and Context7
- Generate all required Android project files under `output/{slug}/`
- Address QA review feedback on revision rounds
- Validate generated files using `python_repl`

## Tools Used
| Tool | Purpose |
|------|---------|
| `web_search` | Find implementation patterns for the feature |
| `knowledge_search` | Check the local knowledge base |
| `context7_search` | Fetch Android SDK / Jetpack API docs |
| `fs_write` | Write generated project files to disk |
| `python_repl` | Validate file existence and check for missing imports |

## Input (from LangGraph state)
| Field | Type | Description |
|-------|------|-------------|
| `spec` | `SpecOutput` | Approved feature specification |
| `review` | `ReviewOutput` (optional) | QA feedback for revision rounds |
| `package_name` | `str` (optional) | Reuse existing package name on revisions |
| `app_name` | `str` (optional) | Reuse existing app name on revisions |

## Output (to LangGraph state)
| Field | Type | Description |
|-------|------|-------------|
| `code` | `CodeOutput` | Generated project metadata |
| `messages` | `list` | Accumulated agent messages |

## CodeOutput Schema
```python
CodeOutput(
    source_code: str,       # content of MainActivity.kt
    description: str,       # brief app description
    app_name: str,          # human-readable app name
    package_name: str,      # full Android package (e.g. "com.example.myapp")
    files_created: list[str], # all paths written via fs_write
)
```

## Generated Project Structure
```
output/{slug}/
├── settings.gradle
├── build.gradle
├── gradle.properties
├── gradle/wrapper/gradle-wrapper.properties
└── app/
    ├── build.gradle
    └── src/main/
        ├── AndroidManifest.xml
        ├── res/values/strings.xml
        └── java/com/example/{slug_nodash}/
            └── MainActivity.kt
```

## Key Technology Constraints
| Component | Version |
|-----------|---------|
| Android Gradle Plugin | 8.4.2 |
| Kotlin | 2.0.0 |
| Compose BOM | 2024.09.00 |
| Gradle wrapper | 8.14.4 |
| compileSdk / targetSdk | 34 |
| minSdk | 24 |

- UI: Jetpack Compose + Material3 only (no XML layouts)
- Storage: DataStore Preferences only (no Room, SQLite, or SharedPreferences)

## Entry Point
```python
from agents.developer import developer_node

# Used as a LangGraph node
graph.add_node("developer", developer_node)
```

## Retry Behaviour
Automatically retries up to 3 times on OpenAI rate limit errors (HTTP 429).
