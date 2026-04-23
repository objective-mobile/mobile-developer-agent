"""
Tool correctness tests.
Verifies that agents call the right tools for their tasks.
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import settings
from tools import web_search, knowledge_search, read_url, save_report


tool_metric = ToolCorrectnessMetric(threshold=0.5, model="gpt-4o-mini")


def _collect_tool_calls(agent, query: str) -> list[ToolCall]:
    """Run agent and collect all tool calls made during execution."""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    tool_calls = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                tool_calls.append(ToolCall(name=name, input_parameters=args))
    return tool_calls


def test_planner_uses_search_tools():
    """Planner should call knowledge_search and/or web_search when given a research query."""
    from config import PLANNER_PROMPT

    llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key.get_secret_value())
    agent = create_react_agent(model=llm, tools=[web_search, knowledge_search], prompt=PLANNER_PROMPT)

    query = "What is Retrieval-Augmented Generation?"
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    actual_calls = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                actual_calls.append(ToolCall(name=name, input_parameters=args))

    expected_tools = [
        ToolCall(name="knowledge_search", input_parameters={"query": query}),
    ]

    test_case = LLMTestCase(
        input=query,
        actual_output=result["messages"][-1].content,
        tools_called=actual_calls,
        expected_tools=expected_tools,
    )
    assert_test(test_case, [tool_metric])


def test_researcher_uses_search_tools():
    """Researcher should call knowledge_search and web_search when executing a plan."""
    from config import RESEARCHER_PROMPT

    llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key.get_secret_value())
    agent = create_react_agent(
        model=llm,
        tools=[web_search, read_url, knowledge_search],
        prompt=RESEARCHER_PROMPT,
    )

    plan_str = (
        "Goal: Explain RAG\n"
        "Search queries: ['what is RAG', 'retrieval augmented generation overview']\n"
        "Sources to check: ['knowledge_base', 'web']\n"
        "Output format: structured markdown report"
    )

    result = agent.invoke({"messages": [{"role": "user", "content": plan_str}]})

    actual_calls = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                actual_calls.append(ToolCall(name=name, input_parameters=args))

    expected_tools = [
        ToolCall(name="knowledge_search", input_parameters={"query": "what is RAG"}),
    ]

    test_case = LLMTestCase(
        input=plan_str,
        actual_output=result["messages"][-1].content,
        tools_called=actual_calls,
        expected_tools=expected_tools,
    )
    assert_test(test_case, [tool_metric])


def test_critic_uses_search_tools():
    """Critic should call web_search or knowledge_search to verify claims."""
    from config import CRITIC_PROMPT

    llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key.get_secret_value())
    agent = create_react_agent(
        model=llm,
        tools=[web_search, read_url, knowledge_search],
        prompt=CRITIC_PROMPT,
    )

    findings = (
        "RAG combines retrieval with generation. "
        "Lewis et al. (2020) introduced it. "
        "It reduces hallucinations significantly."
    )

    result = agent.invoke({"messages": [{"role": "user", "content": findings}]})

    actual_calls = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                actual_calls.append(ToolCall(name=name, input_parameters=args))

    expected_tools = [
        ToolCall(name="web_search", input_parameters={"query": "RAG retrieval augmented generation"}),
    ]

    test_case = LLMTestCase(
        input=findings,
        actual_output=result["messages"][-1].content,
        tools_called=actual_calls,
        expected_tools=expected_tools,
    )
    assert_test(test_case, [tool_metric])
