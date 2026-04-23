"""
Component tests for the Planner Agent.
Evaluates plan quality using GEval.
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.planner import run_planner


plan_quality_metric = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that the plan contains specific search queries (not vague single words)",
        "Check that sources_to_check includes relevant sources for the topic (knowledge_base and/or web)",
        "Check that the output_format describes a structured report format",
        "Check that the goal clearly restates what the user wants to research",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)


def _plan_to_str(plan) -> str:
    return (
        f"Goal: {plan.goal}\n"
        f"Search queries: {plan.search_queries}\n"
        f"Sources to check: {plan.sources_to_check}\n"
        f"Output format: {plan.output_format}"
    )


def test_plan_quality_rag():
    """Planner should produce a high-quality plan for a standard RAG query."""
    query = "What is Retrieval-Augmented Generation (RAG)?"
    plan = run_planner(query)
    actual = _plan_to_str(plan)

    test_case = LLMTestCase(input=query, actual_output=actual)
    assert_test(test_case, [plan_quality_metric])


def test_plan_has_queries():
    """Planner should produce multiple specific search queries, not just one vague term."""
    query = "Compare naive RAG vs sentence-window retrieval"
    plan = run_planner(query)

    assert len(plan.search_queries) >= 2, "Plan should have at least 2 search queries"
    for q in plan.search_queries:
        assert len(q) > 5, f"Query too short/vague: {q!r}"

    actual = _plan_to_str(plan)
    test_case = LLMTestCase(input=query, actual_output=actual)
    assert_test(test_case, [plan_quality_metric])


def test_plan_sources():
    """Planner should include knowledge_base and/or web in sources_to_check."""
    query = "Explain how large language models are trained"
    plan = run_planner(query)

    valid_sources = {"knowledge_base", "web"}
    for src in plan.sources_to_check:
        assert src in valid_sources, f"Unexpected source: {src!r}"

    assert len(plan.sources_to_check) >= 1, "At least one source must be specified"
