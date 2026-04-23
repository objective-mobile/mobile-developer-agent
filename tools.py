"""
Shared tools: web_search, read_url, knowledge_search, save_report,
fs_write, fs_read, python_repl, context7_search.
Reused from hw5 with save_report replacing write_report.
"""
import logging
import os
import subprocess
import sys
from typing import Optional

import trafilatura
import httpx
from ddgs import DDGS
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Allow importing retriever from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from config import settings


@tool
def web_search(query: str) -> str:
    """Search the internet for a given query. Returns titles, URLs, and snippets."""
    try:
        results = DDGS().text(query, max_results=settings.max_search_results)
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(
                f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}"
            )
        return f"Found {len(results)} results:\n\n" + "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def read_url(url: str) -> str:
    """Fetch and return the full text content of a web page."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Could not fetch content from {url}"
        text = trafilatura.extract(downloaded)
        if not text:
            return f"Could not extract text from {url}"
        content = text[: settings.max_url_content_length]
        return f"[{len(content)} chars] {content}"
    except httpx.TimeoutException:
        return f"Timeout while fetching {url}"
    except Exception as e:
        return f"Error reading {url}: {str(e)}"


@tool
def knowledge_search(query: str) -> str:
    """Search the LOCAL knowledge base (ingested PDFs). Use this FIRST before web_search."""
    try:
        from retriever import hybrid_search, is_index_ready

        if not is_index_ready():
            return (
                "Knowledge base index not found. "
                "Run `python ingest.py` first to build the index."
            )

        results = hybrid_search(query)
        if not results:
            return "No relevant documents found in the knowledge base."

        parts = [f"Found {len(results)} relevant passages:\n"]
        for i, r in enumerate(results, 1):
            fname = os.path.basename(r["source"])
            parts.append(
                f"[{i}] (score={r['score']}) [{fname} | Page {r['page']}]\n{r['content']}"
            )
        return "\n\n".join(parts)
    except Exception as e:
        return f"Knowledge search error: {str(e)}"


@tool
def save_report(filename: str, content: str) -> str:
    """Save a Markdown research report to the output directory."""
    try:
        os.makedirs(settings.output_dir, exist_ok=True)
        if not filename.endswith(".md"):
            filename += ".md"
        path = os.path.join(settings.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Report saved to {path}"
    except Exception as e:
        return f"Error saving report: {str(e)}"


# ---------------------------------------------------------------------------
# Filesystem tools
# ---------------------------------------------------------------------------

@tool
def fs_write(path: str, content: str) -> str:
    """Write content to a file on disk, creating parent directories as needed."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"
    except OSError as e:
        return f"Error writing {path}: {str(e)}"


@tool
def fs_read(path: str) -> str:
    """Read and return the content of a file from disk."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except OSError as e:
        return f"Error reading {path}: {str(e)}"


# ---------------------------------------------------------------------------
# Python REPL tool
# ---------------------------------------------------------------------------

@tool
def python_repl(code: str) -> str:
    """Execute Python code in a sandboxed subprocess and return stdout/stderr."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out after 30 seconds"
    except Exception as e:
        return f"Error executing code: {str(e)}"


# ---------------------------------------------------------------------------
# Context7 REST API tool
# ---------------------------------------------------------------------------

_CONTEXT7_API_BASE = "https://context7.com/api/v2"


def _context7_headers() -> dict:
    from config import settings
    api_key = settings.context7_api_key
    if not api_key:
        api_key = os.environ.get("CONTEXT7_API_KEY", "")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _resolve_library_id(library: str, query: str) -> str | None:
    """Search Context7 for the best matching library ID."""
    try:
        resp = httpx.get(
            f"{_CONTEXT7_API_BASE}/libs/search",
            headers=_context7_headers(),
            params={"libraryName": library, "query": query},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            return results[0]["id"]
    except Exception as e:
        logger.debug("[context7] library search failed: %s", e)
    return None


@tool
def context7_search(library: str, version: Optional[str] = None) -> str:
    """Query Context7 REST API for live library documentation.

    Resolves the library name to a Context7 library ID, then fetches
    relevant documentation snippets. Falls back to web_search +
    knowledge_search if the API is unavailable or not configured.
    """
    headers = _context7_headers()
    if not headers:
        logger.warning("[context7] No API key configured; falling back to web/knowledge search")
        query = library if not version else f"{library} {version}"
        web_result = web_search.invoke({"query": f"{query} documentation"})
        rag_result = knowledge_search.invoke({"query": query})
        return f"[Context7 fallback]\n\n## Web Search\n{web_result}\n\n## Knowledge Base\n{rag_result}"

    query = f"{library} {version} documentation" if version else f"{library} documentation"

    try:
        # Step 1: resolve library ID
        library_id = _resolve_library_id(library, query)
        if not library_id:
            raise ValueError(f"Library '{library}' not found in Context7")

        # Step 2: fetch documentation context
        resp = httpx.get(
            f"{_CONTEXT7_API_BASE}/context",
            headers=headers,
            params={"libraryId": library_id, "query": query, "type": "txt"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.text or f"[Context7] No documentation found for {library}"

    except Exception as e:
        logger.warning("[context7] Query failed (%s); falling back to web/knowledge search", e)
        web_result = web_search.invoke({"query": f"{query} documentation"})
        rag_result = knowledge_search.invoke({"query": query})
        return f"[Context7 fallback]\n\n## Web Search\n{web_result}\n\n## Knowledge Base\n{rag_result}"


# ---------------------------------------------------------------------------
# Android build & run tools
# ---------------------------------------------------------------------------

@tool
def gradle_build(project_dir: str) -> str:
    """Build an Android project using the Gradle binary from GRADLE_HOME.

    Calls GRADLE_HOME/bin/gradle assembleDebug directly — no wrapper jar needed.
    Uses JAVA_HOME and ANDROID_HOME from settings/.env.
    """
    try:
        from android_pipeline import build_toolchain_env
        env = build_toolchain_env()
    except Exception:
        env = os.environ.copy()

    gradle_home = env.get("GRADLE_HOME", "")
    if not gradle_home:
        return "Error: GRADLE_HOME is not set. Add it to .env."

    gradle_exe = os.path.join(
        gradle_home, "bin",
        "gradle.bat" if os.name == "nt" else "gradle"
    )

    if not os.path.exists(gradle_exe):
        return f"Error: gradle executable not found at {gradle_exe}"

    cmd = [gradle_exe, "assembleDebug", "--info"]

    print(f"\n🔨 Building: {gradle_exe} assembleDebug")
    print(f"   Working dir: {project_dir}")

    try:
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr

        status = "BUILD SUCCESSFUL" if result.returncode == 0 else f"BUILD FAILED (exit {result.returncode})"
        print(f"\n{'✅' if result.returncode == 0 else '❌'} {status}")

        if result.returncode != 0:
            # Extract meaningful error lines — skip internal stack frames
            error_lines = []
            for line in output.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                # Skip internal Gradle stack frames
                if stripped.startswith("at org.gradle.") or stripped.startswith("at com.sun.") or stripped.startswith("at java."):
                    continue
                # Keep lines with actual error info
                if any(kw in stripped for kw in ("error:", "Error:", "FAILED", "Exception", "warning:", "Could not", "Caused by", "> ")):
                    error_lines.append(stripped)
            if error_lines:
                print("\n--- Build errors ---")
                print("\n".join(error_lines[:40]))
            else:
                # Fallback: last 30 lines
                lines = output.strip().splitlines()
                print("\n".join(lines[-30:]))

        return f"{status}\n\n{output}"
    except subprocess.TimeoutExpired:
        return "Error: gradle build timed out after 300 seconds"
    except Exception as e:
        return f"Error running gradle build: {e}"


@tool
def adb_install_run(apk_path: str, package_name: str, activity: str = ".MainActivity") -> str:
    """Install an APK on all connected ADB devices and launch the main activity.

    Args:
        apk_path: Absolute or relative path to the .apk file.
        package_name: Android package name, e.g. "com.example.myapp".
        activity: Activity to launch, e.g. ".MainActivity" (relative) or full name.

    Returns combined output from adb commands. If no devices are connected,
    returns a descriptive message without raising.
    """
    try:
        from android_pipeline import build_toolchain_env
        env = build_toolchain_env()
    except Exception:
        env = os.environ.copy()

    android_home = env.get("ANDROID_HOME", "")
    adb_candidates = ["adb"]
    if android_home:
        adb_candidates.insert(0, os.path.join(android_home, "platform-tools", "adb"))

    adb = "adb"
    for candidate in adb_candidates:
        try:
            subprocess.run([candidate, "version"], capture_output=True, timeout=5)
            adb = candidate
            break
        except Exception:
            continue

    # Check connected devices
    try:
        devices_result = subprocess.run(
            [adb, "devices"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [l.strip() for l in devices_result.stdout.splitlines()
                 if l.strip() and not l.startswith("List of devices")]
        devices = [l.split()[0] for l in lines if "\tdevice" in l]
    except Exception as e:
        return f"Error checking adb devices: {e}"

    if not devices:
        return "No ADB devices connected. Skipping install/run step."

    output_parts = [f"Found {len(devices)} device(s): {', '.join(devices)}"]

    for device in devices:
        # Install APK
        try:
            install = subprocess.run(
                [adb, "-s", device, "install", "-r", apk_path],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            install_out = (install.stdout + "\n" + install.stderr).strip()
            output_parts.append(f"[{device}] install: {install_out}")
        except Exception as e:
            output_parts.append(f"[{device}] install error: {e}")
            continue

        # Launch activity
        full_activity = f"{package_name}/{activity}" if not activity.startswith(package_name) else activity
        try:
            launch = subprocess.run(
                [adb, "-s", device, "shell", "am", "start", "-n", full_activity],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            launch_out = (launch.stdout + "\n" + launch.stderr).strip()
            output_parts.append(f"[{device}] launch: {launch_out}")
        except Exception as e:
            output_parts.append(f"[{device}] launch error: {e}")

    return "\n".join(output_parts)
