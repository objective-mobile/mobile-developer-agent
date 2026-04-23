"""
End-to-end evaluation on the golden dataset.
Runs the full Supervisor -> Planner -> Researcher -> Critic pipeline
and evaluates with AnswerRelevancy + Correctness metrics.
"""
import json
import os
import uuid
import pytest

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langgraph.types import Command

from supervisor import build_supervisor


GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")

answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict 'expected output'",
        "Penalize omission of critical details mentioned in 'expected output'",
        "Different wording of the same concept is acceptable",
        "If 'actual output' is a refusal for an out-of-domain query, check that 'expected output' also indicates refusal or out-of-scope",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model="gpt-4o-mini",
    threshold=0.6,
)

domain_relevance = GEval(
    name="Domain Relevance",
    evaluation_steps=[
        "Check that the response stays within the domain of AI, LLMs, RAG, and related topics",
        "For failure_case inputs (out-of-domain, nonsensical), check that the system gracefully declines or redirects",
        "For happy_path inputs, check that the response is substantive and on-topic",
        "Penalize responses that hallucinate or go completely off-topic",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.6,
)


def _run_pipeline(query: str) -> str:
    """Run the full supervisor pipeline and return the final answer."""
    supervisor = build_supervisor()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = supervisor.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )

    # Auto-approve any HITL interrupt
    state = supervisor.get_state(config)
    if state.next:
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                result = supervisor.invoke(
                    Command(resume={"action": "approve"}),
                    config=config,
                )
                break

    msgs = result.get("messages", [])
    for msg in reversed(msgs):
        if hasattr(msg, "type") and msg.type == "ai" and msg.content:
            return msg.content
    return ""


def _load_golden_dataset():
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Happy path tests ---

def test_e2e_happy_path_rag():
    """Full pipeline should produce a relevant answer about RAG."""
    dataset = _load_golden_dataset()
    example = next(e for e in dataset if e["category"] == "happy_path" and "RAG" in e["input"])

    actual = _run_pipeline(example["input"])
    assert actual, "Pipeline returned empty response"

    test_case = LLMTestCase(
        input=example["input"],
        actual_output=actual,
        expected_output=example["expected_output"],
    )
    assert_test(test_case, [answer_relevancy, correctness])


def test_e2e_happy_path_langchain():
    """Full pipeline should produce a relevant answer about LangChain."""
    dataset = _load_golden_dataset()
    example = next(e for e in dataset if "LangChain" in e["input"])

    actual = _run_pipeline(example["input"])
    assert actual

    test_case = LLMTestCase(
        input=example["input"],
        actual_output=actual,
        expected_output=example["expected_output"],
    )
    assert_test(test_case, [answer_relevancy, correctness])


# --- Golden dataset bulk evaluation ---

def test_golden_dataset_bulk():
    """
    Run evaluation on the full golden dataset.
    Collects scores and prints a summary. Does not fail on individual misses —
    the goal is to establish a baseline.
    """
    dataset = _load_golden_dataset()
    results = []

    for example in dataset:
        try:
            actual = _run_pipeline(example["input"])
            if not actual:
                results.append({
                    "input": example["input"],
                    "category": example["category"],
                    "passed": False,
                    "error": "empty response",
                })
                continue

            test_case = LLMTestCase(
                input=example["input"],
                actual_output=actual,
                expected_output=example["expected_output"],
            )

            correctness.measure(test_case)
            answer_relevancy.measure(test_case)
            domain_relevance.measure(test_case)

            passed = (
                correctness.score >= correctness.threshold
                and answer_relevancy.score >= answer_relevancy.threshold
            )

            results.append({
                "input": example["input"][:60],
                "category": example["category"],
                "correctness": round(correctness.score, 3),
                "relevancy": round(answer_relevancy.score, 3),
                "domain": round(domain_relevance.score, 3),
                "passed": passed,
            })
        except Exception as e:
            results.append({
                "input": example["input"][:60],
                "category": example["category"],
                "passed": False,
                "error": str(e)[:100],
            })

    # Print summary
    passed_count = sum(1 for r in results if r.get("passed"))
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Golden Dataset Results: {passed_count}/{total} passed ({100*passed_count//total}%)")
    print(f"{'='*60}")
    for r in results:
        status = "✅" if r.get("passed") else "❌"
        scores = ""
        if "correctness" in r:
            scores = f" | C:{r['correctness']} R:{r['relevancy']} D:{r['domain']}"
        err = f" | ERROR: {r['error']}" if "error" in r else ""
        print(f"  {status} [{r['category']}] {r['input']}{scores}{err}")

    # Save results to output
    os.makedirs("output", exist_ok=True)
    with open("output/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Baseline assertion: at least 50% of happy_path cases should pass
    happy_results = [r for r in results if r.get("category") == "happy_path"]
    happy_passed = sum(1 for r in happy_results if r.get("passed"))
    assert happy_passed >= len(happy_results) // 2, (
        f"Too many happy_path failures: {happy_passed}/{len(happy_results)}"
    )
