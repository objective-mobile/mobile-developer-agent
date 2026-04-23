# QA Engineer Agent

## Overview
The QA agent is a LangGraph ReAct agent that reviews a generated Android project by reading all files, building the APK with Gradle, and optionally installing/launching it on a connected device via ADB.

## Responsibilities
- Read and validate all generated project files
- Build the APK using Gradle and report build errors
- Install and launch the app on connected ADB devices
- Produce a structured quality verdict with score and actionable feedback

## Tools Used
| Tool | Purpose |
|------|---------|
| `fs_read` | Read generated source and config files |
| `python_repl` | Validate file existence and XML structure |
| `gradle_build` | Trigger a Gradle build and capture output |
| `adb_install_run` | Install and launch the APK on connected devices |

## Input (from LangGraph state)
| Field | Type | Description |
|-------|------|-------------|
| `code` | `CodeOutput` | Generated project metadata including `files_created` |
| `package_name` | `str` (optional) | Override package name from state memory |
| `app_name` | `str` (optional) | Override app name from state memory |

## Output (to LangGraph state)
| Field | Type | Description |
|-------|------|-------------|
| `review` | `ReviewOutput` | Quality verdict |
| `messages` | `list` | Accumulated agent messages |

## ReviewOutput Schema
```python
ReviewOutput(
    verdict: str,           # "APPROVED" | "REVISION_NEEDED"
    issues: list[str],      # critical problems (non-empty when REVISION_NEEDED)
    suggestions: list[str], # non-blocking improvement notes
    score: float,           # 0.0 – 1.0 quality score
)
```

## Scoring Guide
| Score | Condition |
|-------|-----------|
| 1.0 | All files present, build successful, app launched on device |
| 0.8 | All files present, build successful, no devices connected |
| 0.5–0.79 | Build successful but minor code issues |
| 0.0–0.49 | Missing files or build failure |

## Review Process (in order)
1. Read every file listed in `files_created` using `fs_read`
2. Validate file existence and XML structure using `python_repl`
3. Build the APK with `gradle_build(project_root)`
4. If build succeeded, install and launch with `adb_install_run`
5. Produce `ReviewOutput`

## Entry Point
```python
from agents.qa import qa_node

# Used as a LangGraph node
graph.add_node("qa", qa_node)
```

## Retry Behaviour
Automatically retries up to 3 times on OpenAI rate limit errors (HTTP 429).
