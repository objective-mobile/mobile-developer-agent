"""
Android development pipeline — LangGraph StateGraph definition.

Wires BA → HITL Gate → Developer → QA with:
- MemorySaver checkpointer for HITL persistence
- Command API for conditional routing out of QA
- Toolchain env helper for subprocess PATH construction
- Observability setup for Langfuse / LangSmith
"""
import logging
import os
from typing import Optional

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from schemas import CodeOutput, ReviewOutput, SpecOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 7.1 — AndroidPipelineState
# ---------------------------------------------------------------------------

class AndroidPipelineState(TypedDict):
    user_story: str
    spec: Optional[SpecOutput]
    code: Optional[CodeOutput]
    review: Optional[ReviewOutput]
    iteration: int
    messages: list[BaseMessage]
    hitl_feedback: Optional[str]
    app_name: Optional[str]       # persisted from CodeOutput for QA/ADB reuse
    package_name: Optional[str]   # persisted from CodeOutput for QA/ADB reuse


# ---------------------------------------------------------------------------
# 7.2 — build_toolchain_env
# ---------------------------------------------------------------------------

def build_toolchain_env() -> dict[str, str]:
    """Return a copy of os.environ with Android toolchain bin/ dirs prepended to PATH.

    Reads ANDROID_HOME, JAVA_HOME, GRADLE_HOME from pydantic settings (which
    loads .env) and injects them into the returned env dict so subprocesses
    see them even if they weren't exported in the shell.
    """
    from config import settings as _settings

    env = os.environ.copy()

    # Inject values from .env / pydantic settings into the env dict
    setting_map = {
        "ANDROID_HOME": _settings.android_home,
        "JAVA_HOME": _settings.java_home,
        "GRADLE_HOME": _settings.gradle_home,
    }
    for var, value in setting_map.items():
        if value and var not in env:
            env[var] = value

    current_path = env.get("PATH", "")
    prepend_parts: list[str] = []

    for var in ("JAVA_HOME", "ANDROID_HOME"):
        value = env.get(var)
        if value:
            bin_dir = os.path.join(value, "bin")
            prepend_parts.append(bin_dir)
        else:
            logging.warning("[toolchain] %s not set, skipping build step", var)

    # Add Android platform-tools (adb) to PATH
    android_home = env.get("ANDROID_HOME")
    if android_home:
        prepend_parts.append(os.path.join(android_home, "platform-tools"))

    if prepend_parts:
        new_path = os.pathsep.join(prepend_parts + [current_path])
        env["PATH"] = new_path

    return env


# ---------------------------------------------------------------------------
# 7.4 — setup_observability
# ---------------------------------------------------------------------------

def setup_observability() -> list:
    """Return a list of LangChain callbacks for tracing.

    - Detects LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY → Langfuse callback
    - Detects LANGCHAIN_API_KEY → enables LangSmith via LANGCHAIN_TRACING_V2=true
    - If neither present, logs a warning and returns []
    """
    callbacks: list = []

    langfuse_public = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret = os.environ.get("LANGFUSE_SECRET_KEY", "")
    langchain_api_key = os.environ.get("LANGCHAIN_API_KEY", "")

    if langfuse_public and langfuse_secret:
        try:
            from langfuse.callback import CallbackHandler as LangfuseCallback  # type: ignore
            handler = LangfuseCallback(
                public_key=langfuse_public,
                secret_key=langfuse_secret,
            )
            callbacks.append(handler)
            logger.info("[monitor] Langfuse tracing enabled")
        except ImportError:
            logger.warning("[monitor] langfuse package not installed; skipping Langfuse tracing")

    if langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        logger.info("[monitor] LangSmith tracing enabled")

    if not callbacks and not langchain_api_key:
        logging.warning("[monitor] No tracing credentials found, continuing without tracing")

    return callbacks


# ---------------------------------------------------------------------------
# 7.5 — HITL gate node
# ---------------------------------------------------------------------------

def hitl_gate_node(state: AndroidPipelineState) -> Command:
    """Pause execution for human review of the BA spec.

    Calls interrupt() to surface the spec to the user.
    On resume:
      - action == "approve"  → route to developer
      - action == "reject"   → store feedback and route back to ba
    """
    decision = interrupt({"spec": state["spec"]})

    action = decision.get("action", "reject") if isinstance(decision, dict) else "reject"

    if action == "approve":
        return Command(goto="developer")

    # Reject path — capture feedback and loop back to BA
    feedback = decision.get("feedback", "") if isinstance(decision, dict) else str(decision)
    return Command(
        goto="ba",
        update={"hitl_feedback": feedback},
    )


# ---------------------------------------------------------------------------
# 7.6 — QA wrapper node (increments iteration, routes via Command)
# ---------------------------------------------------------------------------

def qa_wrapper_node(state: AndroidPipelineState) -> Command:
    """Wrap the QA node to increment iteration and route via Command API."""
    from agents.qa import qa_node

    result = qa_node(state)
    new_iteration = state.get("iteration", 0) + 1
    review: Optional[ReviewOutput] = result.get("review")

    if review and review.verdict == "REVISION_NEEDED" and new_iteration < 5:
        return Command(
            goto="developer",
            update={**result, "iteration": new_iteration},
        )
    return Command(
        goto=END,
        update={**result, "iteration": new_iteration},
    )


# ---------------------------------------------------------------------------
# 7.6 — BA / Developer node wrappers (plain dict returns)
# ---------------------------------------------------------------------------

def ba_wrapper_node(state: AndroidPipelineState) -> dict:
    from agents.ba import ba_node
    return ba_node(state)


def developer_wrapper_node(state: AndroidPipelineState) -> dict:
    from agents.developer import developer_node
    result = developer_node(state)
    code: Optional[CodeOutput] = result.get("code")
    extra = {}
    if code:
        extra["app_name"] = code.app_name or ""
        extra["package_name"] = code.package_name or ""
    return {**result, **extra}


# ---------------------------------------------------------------------------
# 7.6 — Build and compile the StateGraph
# ---------------------------------------------------------------------------

def build_android_pipeline():
    """Compile and return the Android development StateGraph.

    Graph topology:
        START → ba → hitl_gate (Command-based routing)
        hitl_gate → developer (approve) | ba (reject)
        developer → qa_wrapper
        qa_wrapper → developer (REVISION_NEEDED, iteration < 5) | END
    """
    graph = StateGraph(AndroidPipelineState)

    # Register nodes
    graph.add_node("ba", ba_wrapper_node)
    graph.add_node("hitl_gate", hitl_gate_node)
    graph.add_node("developer", developer_wrapper_node)
    graph.add_node("qa", qa_wrapper_node)

    # Static edges
    graph.add_edge(START, "ba")
    graph.add_edge("ba", "hitl_gate")
    # hitl_gate uses Command for routing (approve → developer, reject → ba)
    # developer always proceeds to qa
    graph.add_edge("developer", "qa")
    # qa uses Command for routing (REVISION_NEEDED + iter<5 → developer, else → END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Compiled pipeline (lazy singleton — imported by main.py)
android_pipeline = None


def get_android_pipeline():
    """Return the compiled pipeline, building it on first call."""
    global android_pipeline
    if android_pipeline is None:
        android_pipeline = build_android_pipeline()
    return android_pipeline
