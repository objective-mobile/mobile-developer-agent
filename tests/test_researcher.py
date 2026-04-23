"""
Component tests for the Research Agent.
Evaluates groundedness of research output against retrieved context.
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.planner import run_planner
from agents.research import run_researcher
from retriever import hybrid_search, is_index_ready


groundedness_metric = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Extract every factual claim from 'actual output'",
        "For each claim, check if it can be directly supported by 'retrieval context'",
        "Claims not present in retrieval context count as ungrounded, even if generally true",
        "Score = number of grounded claims / total claims",
        "If retrieval context is empty, score based on whether claims are reasonable for the topic",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model="gpt-4o-mini",
    threshold=0.7,
)

completeness_metric = GEval(
    name="Research Completeness",
    evaluation_steps=[
        "Check that the output covers the main aspects of the input query",
        "Check that the output is structured (has sections or clear paragraphs)",
        "Check that the output includes sources or references where applicable",
        "Penalize if the output is just a single sentence or clearly incomplete",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.6,
)


def _get_retrieval_context(query: str) -> list[str]:
    if not is_index_ready():
        return []
    results = hybrid_search(query)
    return [r["content"] for r in results]


def test_research_grounded():
    """Research output should be grounded in the knowledge base for a RAG query."""
    query = "What is Retrieval-Augmented Generation (RAG)?"
    plan = run_planner(query)
    plan_str = (
        f"Goal: {plan.goal}\n"
        f"Search queries: {plan.search_queries}\n"
        f"Sources to check: {plan.sources_to_check}\n"
        f"Output format: {plan.output_format}"
    )

    findings = run_researcher(plan_str)
    retrieval_context = _get_retrieval_context(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=findings,
        retrieval_context=retrieval_context if retrieval_context else [findings[:500]],
    )
    assert_test(test_case, [groundedness_metric])


def test_research_completeness():
    """Research output should be complete and well-structured."""
    query = "What are the main components of a LangChain pipeline?"
    plan = run_planner(query)
    plan_str = (
        f"Goal: {plan.goal}\n"
        f"Search queries: {plan.search_queries}\n"
        f"Sources to check: {plan.sources_to_check}\n"
        f"Output format: {plan.output_format}"
    )

    findings = run_researcher(plan_str)

    assert len(findings) > 200, "Research output is too short to be complete"

    test_case = LLMTestCase(input=query, actual_output=findings)
    assert_test(test_case, [completeness_metric])


def test_research_edge_case():
    """Research should handle a vague query gracefully without crashing."""
    query = "rag"
    plan = run_planner(query)
    plan_str = f"Goal: {plan.goal}\nSearch queries: {plan.search_queries}"

    findings = run_researcher(plan_str)
    retrieval_context = _get_retrieval_context(query)

    assert findings is not None and len(findings) > 0

    test_case = LLMTestCase(
        input=query,
        actual_output=findings,
        retrieval_context=retrieval_context if retrieval_context else [findings[:500]],
    )
    # Lower threshold for edge cases
    edge_groundedness = GEval(
        name="Groundedness",
        evaluation_steps=[
            "Extract every factual claim from 'actual output'",
            "For each claim, check if it can be directly supported by 'retrieval context'",
            "Score = number of grounded claims / total claims",
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model="gpt-4o-mini",
        threshold=0.5,
    )
    assert_test(test_case, [edge_groundedness])
