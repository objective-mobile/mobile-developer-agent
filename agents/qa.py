import sys
import os
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import settings
from tools import fs_read, python_repl, gradle_build, adb_install_run
from schemas import CodeOutput, ReviewOutput


QA_SYSTEM_PROMPT = """You are an expert QA Engineer specialising in Android application code review and testing.

## Your Mission
Review the generated Android project files, build the APK, install and launch it on any connected
devices, then produce a structured quality verdict.

## Review Process (MANDATORY — follow this order exactly)

### Step 1 — Read all files
For EVERY file path listed in the user message, call `fs_read` to read its contents.
- If `fs_read` returns a string starting with "Error:", the file is MISSING.
  Set `verdict = "REVISION_NEEDED"` and add "File missing: <path>" to issues.

### Step 2 — Validate structure
Use `python_repl` to verify files exist and check basic structure:
```python
import os, xml.etree.ElementTree as ET
files = [...]  # all paths from the user message
for f in files:
    print(f, "EXISTS" if os.path.exists(f) else "MISSING")
# Try parsing XML files
for f in files:
    if f.endswith(".xml"):
        try: ET.parse(f); print(f, "XML OK")
        except Exception as e: print(f, "XML ERROR:", e)
```

### Step 3 — Build the APK
Call `gradle_build` with the project root directory (the `output/{slug}/` folder).
- If the build output contains "BUILD SUCCESSFUL": proceed to Step 4.
- If it contains "BUILD FAILED": add the first error line to `issues`, set verdict to
  "REVISION_NEEDED", skip Steps 4–5, and go straight to Step 6.

### Step 4 — Install and launch on connected devices
Call `adb_install_run` with:
- `apk_path`: the path to the generated APK, typically
  `output/{slug}/app/build/outputs/apk/debug/app-debug.apk`
- `package_name`: the app package name (e.g. `com.example.{slug_nodash}`)
- `activity`: `.MainActivity`

If the result says "No ADB devices connected", note it as a suggestion (not an issue) and continue.
If install/launch fails on a connected device, add the error to `issues`.

### Step 5 — Score and verdict
Assign a `score` between 0.0 and 1.0:
- 1.0: all files present, build successful, app launched on device
- 0.8: all files present, build successful, no devices connected
- 0.5–0.79: build successful but minor code issues
- 0.0–0.49: missing files or build failure

Set `verdict`:
- "APPROVED" if build succeeded and no critical issues
- "REVISION_NEEDED" if any file is missing OR build failed OR app crashed on launch

### Step 6 — Produce ReviewOutput
Fill in `verdict`, `issues`, `suggestions`, and `score`.
- `issues` MUST be non-empty when verdict is "REVISION_NEEDED".
- Include build output excerpts in issues/suggestions where helpful.
"""


def _build_qa_agent():
    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    return create_react_agent(
        model=llm,
        tools=[fs_read, python_repl, gradle_build, adb_install_run],
        prompt=QA_SYSTEM_PROMPT,
        response_format=ReviewOutput,
    )


def _invoke_with_retry(agent, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return agent.invoke({"messages": messages})
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = 60
                m = re.search(r"retry after (\d+)", err, re.IGNORECASE)
                if m:
                    wait = int(m.group(1)) + 2
                print(f"  [QA] Rate limit, waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("[QA] Max retries exceeded")


def qa_node(state: dict) -> dict:
    """LangGraph node for the QA Engineer agent.

    Extracts `code` (CodeOutput) from state, builds a prompt listing all
    `files_created` paths for the agent to review, runs the ReAct agent
    (fs_read + python_repl), and returns the structured ReviewOutput stored
    under state["review"].
    """
    code: CodeOutput | None = state.get("code")

    if code is None:
        raise ValueError("[QA] No code found in state")

    # Prefer persisted memory values, fall back to deriving from files_created
    project_dir = ""
    package_name = state.get("package_name") or code.package_name or ""
    app_name = state.get("app_name") or code.app_name or ""

    for path in code.files_created:
        normalized = path.replace("\\", "/")
        if "output/" in normalized and project_dir == "":
            parts = normalized.split("output/")
            if len(parts) > 1:
                slug = parts[1].split("/")[0]
                project_dir = f"output/{slug}"
        # Derive package_name from source path if not already known
        if not package_name and "/java/" in normalized and normalized.endswith(".kt"):
            java_idx = normalized.find("/java/")
            pkg_path = normalized[java_idx + 6:]
            pkg_parts = pkg_path.split("/")[:-1]
            if pkg_parts:
                package_name = ".".join(pkg_parts)

    files_list = "\n".join(f"- {path}" for path in code.files_created)
    apk_path = f"{project_dir}/app/build/outputs/apk/debug/app-debug.apk" if project_dir else ""

    print(f"\n🔍 QA starting review")
    print(f"   App:     {app_name or '(unknown)'}")
    print(f"   Package: {package_name or '(unknown)'}")
    print(f"   Dir:     {project_dir or '(unknown)'}")
    print(f"   Files:   {len(code.files_created)}")

    user_message = (
        f"Review the following Android project.\n\n"
        f"Project root directory: {project_dir}\n"
        f"Package name: {package_name}\n"
        f"APK path (after build): {apk_path}\n\n"
        f"Files to review (read EVERY one using fs_read):\n{files_list}\n\n"
        f"Project description: {code.description}\n\n"
        f"Follow the review process in your instructions:\n"
        f"1. Read all files with fs_read\n"
        f"2. Validate structure with python_repl\n"
        f"3. Build with gradle_build('{project_dir}')\n"
        f"4. If build succeeded, install+launch with adb_install_run('{apk_path}', '{package_name}')\n"
        f"5. Produce ReviewOutput"
    )

    agent = _build_qa_agent()
    result = _invoke_with_retry(agent, [{"role": "user", "content": user_message}])

    messages = result.get("messages", [])

    # response_format surfaces the parsed model in result["structured_response"]
    review_output = result.get("structured_response")

    # Fallback: scan messages for a ReviewOutput instance
    if not isinstance(review_output, ReviewOutput):
        review_output = None
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if isinstance(content, ReviewOutput):
                review_output = content
                break
            if isinstance(content, dict):
                try:
                    review_output = ReviewOutput(**content)
                    break
                except Exception:
                    pass

    existing_messages = state.get("messages", [])
    updated_messages = existing_messages + messages

    return {
        "review": review_output,
        "messages": updated_messages,
    }
