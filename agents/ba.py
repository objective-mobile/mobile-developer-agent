import sys
import os
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import settings
from tools import web_search, knowledge_search, context7_search
from schemas import SpecOutput


BA_SYSTEM_PROMPT = """You are an expert Business Analyst specialising in Android application development.

## Your Mission
Given a user story, research the domain thoroughly and produce a structured specification.

## Research Process (MANDATORY — do this BEFORE writing the spec)
1. Call `web_search` to find relevant context, best practices, and similar implementations.
2. Call `knowledge_search` to search the local knowledge base for internal documentation.
3. Call `context7_search` with the relevant Android library or framework to get up-to-date API docs.

## Spec Generation
After completing research, produce a `SpecOutput` with:
- `title`: a concise, descriptive title for the feature
- `requirements`: at least one clear, testable requirement (functional and non-functional)
- `acceptance_criteria`: at least one specific, verifiable acceptance criterion
- `estimated_complexity`: one of "simple", "medium", or "complex"

## Rules
- You MUST call both `web_search` and `knowledge_search` before producing the spec.
- Base your spec on the research findings — do not invent requirements without evidence.
- Be specific and actionable in requirements and acceptance criteria.
"""


def _build_ba_agent():
    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    return create_react_agent(
        model=llm,
        tools=[web_search, knowledge_search, context7_search],
        prompt=BA_SYSTEM_PROMPT,
        response_format=SpecOutput,
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
                print(f"  [BA] Rate limit, waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("[BA] Max retries exceeded")


def ba_node(state: dict) -> dict:
    """LangGraph node for the Business Analyst agent.

    Extracts `user_story` and optional `hitl_feedback` from state,
    runs the ReAct agent (web_search + knowledge_search + context7_search),
    and returns the structured SpecOutput stored under state["spec"].
    """
    user_story = state.get("user_story", "")
    hitl_feedback = state.get("hitl_feedback")

    # Build the user message, incorporating HITL feedback on revision rounds
    if hitl_feedback:
        user_message = (
            f"User Story:\n{user_story}\n\n"
            f"REVISION FEEDBACK from reviewer:\n{hitl_feedback}\n\n"
            "Please revise the specification addressing the feedback above."
        )
    else:
        user_message = f"User Story:\n{user_story}"

    agent = _build_ba_agent()
    result = _invoke_with_retry(agent, [{"role": "user", "content": user_message}])

    messages = result.get("messages", [])

    # response_format surfaces the parsed model in result["structured_response"]
    spec_output = result.get("structured_response")

    # Fallback: scan messages for a SpecOutput instance
    if not isinstance(spec_output, SpecOutput):
        spec_output = None
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if isinstance(content, SpecOutput):
                spec_output = content
                break
            if isinstance(content, dict):
                try:
                    spec_output = SpecOutput(**content)
                    break
                except Exception:
                    pass

    existing_messages = state.get("messages", [])
    updated_messages = existing_messages + messages

    return {
        "spec": spec_output,
        "messages": updated_messages,
    }
