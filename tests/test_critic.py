"""
Component tests for the Critic Agent.
Evaluates critique quality using GEval.
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.critic import run_critic


critique_quality_metric = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "Check that the critique identifies specific issues, not vague complaints",
        "Check that revision_requests are actionable (researcher can act on them)",
        "If verdict is APPROVE, gaps list should be empty or contain only minor items",
        "If verdict is REVISE, there must be at least one revision_request",
        "Check that the critique evaluates freshness, completeness, and structure",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)

verdict_consistency_metric = GEval(
    name="Verdict Consistency",
    evaluation_steps=[
        "If the actual output contains 'APPROVE', check that the critique is mostly positive with few or no gaps",
        "If the actual output contains 'REVISE', check that there are specific revision requests listed",
        "The verdict must be consistent with the identified strengths and gaps",
        "A verdict of APPROVE with many critical gaps is inconsistent — penalize heavily",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)


GOOD_FINDINGS = """
# Retrieval-Augmented Generation: A Comprehensive Overview

## Introduction
Retrieval-Augmented Generation (RAG) combines a retrieval system with a generative language model.
When a query arrives, relevant documents are fetched from a vector store and provided as context.

## How It Works
1. Query is embedded using a dense encoder (e.g., sentence-transformers)
2. Top-k similar chunks are retrieved from FAISS or similar index
3. Retrieved chunks are prepended to the prompt
4. The LLM generates a grounded answer

## Types of RAG
- **Naive RAG**: Fixed-size chunks, simple cosine similarity retrieval
- **Sentence-window RAG**: Small chunks for retrieval, larger window returned as context
- **Parent-child RAG**: Hierarchical chunking for better context coherence

## Sources
- Lewis et al. (2020) "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- LangChain documentation (2024)
- Local knowledge base: retrieval-augmented-generation.pdf

## Summary
RAG is a proven technique to reduce hallucinations and improve factual accuracy in LLM responses.
"""

POOR_FINDINGS = """
RAG is a thing. It uses retrieval. LLMs are used too. There are some papers about it.
It might be good. Sources: internet.
"""


def _critique_to_str(c) -> str:
    return (
        f"Verdict: {c.verdict}\n"
        f"Is fresh: {c.is_fresh}\n"
        f"Is complete: {c.is_complete}\n"
        f"Is well structured: {c.is_well_structured}\n"
        f"Strengths: {c.strengths}\n"
        f"Gaps: {c.gaps}\n"
        f"Revision requests: {c.revision_requests}"
    )


def test_critique_approve():
    """Critic should APPROVE high-quality, well-structured research."""
    critique = run_critic(GOOD_FINDINGS)
    actual = _critique_to_str(critique)

    test_case = LLMTestCase(input=GOOD_FINDINGS, actual_output=actual)
    assert_test(test_case, [critique_quality_metric, verdict_consistency_metric])


def test_critique_revise():
    """Critic should REVISE poor, incomplete research with actionable requests."""
    critique = run_critic(POOR_FINDINGS)
    actual = _critique_to_str(critique)

    test_case = LLMTestCase(input=POOR_FINDINGS, actual_output=actual)
    assert_test(test_case, [critique_quality_metric, verdict_consistency_metric])

    # If verdict is REVISE, there must be revision requests
    if critique.verdict == "REVISE":
        assert len(critique.revision_requests) > 0, (
            "REVISE verdict must include at least one revision_request"
        )


def test_critique_has_structured_output():
    """CritiqueResult must always have all required fields populated."""
    critique = run_critic(GOOD_FINDINGS)

    assert critique.verdict in ("APPROVE", "REVISE")
    assert isinstance(critique.strengths, list)
    assert isinstance(critique.gaps, list)
    assert isinstance(critique.revision_requests, list)
    assert isinstance(critique.is_fresh, bool)
    assert isinstance(critique.is_complete, bool)
    assert isinstance(critique.is_well_structured, bool)
