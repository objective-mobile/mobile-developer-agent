"""
Supervisor agent that orchestrates Plan -> Research -> Critique cycle.
Sub-agents are wrapped as @tool functions.
HITL is implemented via LangGraph interrupt on save_report.
"""
import json
import time
import re
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from config import settings, SUPERVISOR_PROMPT
from tools import save_report as _save_report_tool


@tool
def plan(request: str) -> str:
    """Decompose the user research request into a structured ResearchPlan."""
    from agents.planner import run_planner

    print(f"\n[Supervisor -> Planner]")
    print(f"  plan({request[:80]!r})")

    plan_obj = run_planner(request)
    print(f"  ResearchPlan: goal={plan_obj.goal!r}")
    print(f"  queries={plan_obj.search_queries}")
    return (
        f"Goal: {plan_obj.goal}\n"
        f"Search queries: {json.dumps(plan_obj.search_queries)}\n"
        f"Sources to check: {json.dumps(plan_obj.sources_to_check)}\n"
        f"Output format: {plan_obj.output_format}"
    )


@tool
def research(request: str) -> str:
    """Execute research following a plan. Returns detailed findings as markdown."""
    from agents.research import run_researcher

    print(f"\n[Supervisor -> Researcher]")
    print(f"  research({request[:80]!r}...)")

    content = run_researcher(request)
    print(f"  Research complete ({len(str(content))} chars)")
    return str(content)


@tool
def critique(findings: str) -> str:
    """Critically evaluate research findings. Returns verdict APPROVE or REVISE."""
    from agents.critic import run_critic

    print(f"\n[Supervisor -> Critic]")
    print(f"  critique(findings: {len(findings)} chars)")

    c = run_critic(findings)
    print(f"  CritiqueResult: verdict={c.verdict!r}, fresh={c.is_fresh}, complete={c.is_complete}")
    print(f"  gaps={c.gaps}")
    return (
        f"Verdict: {c.verdict}\n"
        f"Is fresh: {c.is_fresh}\n"
        f"Is complete: {c.is_complete}\n"
        f"Is well structured: {c.is_well_structured}\n"
        f"Strengths: {json.dumps(c.strengths)}\n"
        f"Gaps: {json.dumps(c.gaps)}\n"
        f"Revision requests: {json.dumps(c.revision_requests)}"
    )


@tool
def save_report(filename: str, content: str) -> str:
    """Save the final research report. Requires user approval (HITL)."""
    print(f"\n[Supervisor -> save_report]")
    print(f"  save_report(filename={filename!r}, content: {len(content)} chars)")

    decision = interrupt({
        "tool": "save_report",
        "filename": filename,
        "content_preview": content[:500] + ("..." if len(content) > 500 else ""),
        "full_content": content,
    })

    action = decision.get("action", "reject")

    if action == "approve":
        result = _save_report_tool.invoke({"filename": filename, "content": content})
        print(f"\n  Approved! {result}")
        return result
    elif action == "edit":
        feedback = decision.get("feedback", "")
        return f"EDIT_REQUESTED: {feedback}"
    else:
        reason = decision.get("reason", "User rejected")
        print(f"\n  Rejected: {reason}")
        return f"Report saving was rejected by user: {reason}"


def build_supervisor():
    llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key.get_secret_value())
    checkpointer = MemorySaver()
    supervisor = create_react_agent(
        model=llm,
        tools=[plan, research, critique, save_report],
        prompt=SUPERVISOR_PROMPT,
        checkpointer=checkpointer,
    )
    return supervisor
