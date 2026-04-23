# Homework: Testing a Multi-Agent System (Extension of hw8)

Write automated tests for your multi-agent system from `homework-lesson-8`, using DeepEval and the approaches from Lecture 10.

---

### What Changes Compared to homework-8

| Before (homework-lesson-8) | After (homework-lesson-10)                    |
|-|----------------------------------------------|
| Multi-agent system without tests | Same system + test coverage           |
| Quality checked manually (vibe check) | Automated evals with 0–1 metrics         |
| No golden dataset | 10–15 golden examples for regression testing |
| No CI-ready tests | `deepeval test run` runs all tests       |

---

### What You Need to Implement

#### 1. Golden Dataset (10–15 examples)

Create a golden dataset for testing your system. Each example is an `input` → `expected_output` pair with a category:

| Category | Count | Examples |
|---|-----------|---|
| **Happy path** | 3–5       | Typical research queries the system should answer fully |
| **Edge cases** | 3–5       | Ambiguous queries, very narrow or very broad topics, queries in multiple languages |
| **Failure cases** | 3–5       | Out-of-domain queries, nonsensical queries, queries on forbidden topics |

Save as `tests/golden_dataset.json`:

```json
[
  {
    "input": "Compare naive RAG vs sentence-window retrieval",
    "expected_output": "Naive RAG splits documents into fixed-size chunks...",
    "category": "happy_path"
  }
]
```

You can use Ragas `TestsetGenerator` for initial generation, but **manual review is required** — fix or remove low-quality examples.

#### 2. Component-Level Tests

Test each sub-agent individually.

**Planner Agent — plan structure quality:**

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that the plan contains specific search queries (not vague)",
        "Check that sources_to_check includes relevant sources for the topic",
        "Check that the output_format matches what the user asked for",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-5.4-mini",
    threshold=0.7,
)
```

**Critic Agent — critique quality:**

```python
critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "Check that the critique identifies specific issues, not vague complaints",
        "Check that revision_requests are actionable (researcher can act on them)",
        "If verdict is APPROVE, gaps list should be empty or contain only minor items",
        "If verdict is REVISE, there must be at least one revision_request",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-5.4-mini",
    threshold=0.7,
)
```

**Research Agent — answer groundedness:**

```python
groundedness = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Extract every factual claim from 'actual output'",
        "For each claim, check if it can be directly supported by 'retrieval context'",
        "Claims not present in retrieval context count as ungrounded, even if true",
        "Score = number of grounded claims / total claims",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model="gpt-5.4-mini",
    threshold=0.7,
)
```

#### 3. Tool Correctness Tests

Verify that agents call the correct tools:

```python
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric

# Planner should use web_search and/or knowledge_search for exploration
# Researcher should use web_search, read_url, knowledge_search
# Critic should verify facts via web_search

tool_metric = ToolCorrectnessMetric(threshold=0.5, model="gpt-5.4-mini")
```

Create at least 3 test cases for tool correctness:
- Planner receives a query → should call search tools
- Researcher receives a plan → should use tools according to `sources_to_check`
- Supervisor receives APPROVE from Critic → should call `save_report`

#### 4. End-to-End Test

Test the full pipeline Supervisor → Planner → Researcher → Critic:

```python
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-5.4-mini")

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict 'expected output'",
        "Penalize omission of critical details",
        "Different wording of the same concept is acceptable",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model="gpt-5.4-mini",
    threshold=0.6,
)
```

Run evaluation on the full golden dataset and save the results.

### Project Structure

```
homework-lesson-10/
├── tests/
│   ├── golden_dataset.json       # 15-20 golden examples
│   ├── test_planner.py           # Planner agent tests
│   ├── test_researcher.py        # Research agent tests (groundedness)
│   ├── test_critic.py            # Critic agent tests
│   ├── test_tools.py             # Tool correctness tests
│   └── test_e2e.py               # End-to-end evaluation on golden dataset
├── ... (all files from homework-lesson-8)
└── README.md
```

---

### How to Run Tests

```bash
# Run all tests
deepeval test run tests/

# Run specific test file
deepeval test run tests/test_planner.py

# Run with verbose output
deepeval test run tests/ -v
```

---

### Requirements

1. **Golden Dataset:** 15–20 examples (happy path + edge cases + failure cases), saved as JSON
2. **Component tests:** at least one test per Planner, Researcher, Critic
3. **Tool correctness:** at least 3 test cases
4. **End-to-end:** evaluation on the full golden dataset with at least 2 metrics
5. **Custom metric:** at least 1 GEval metric tailored to your business logic
6. **Thresholds:** justified thresholds (don't start at 0.95 — establish a baseline first, then raise them)
7. **Tests run:** `deepeval test run tests/` passes without errors

---

### Expected Output

```
$ deepeval test run tests/

Running 5 test files...

tests/test_planner.py
  ✅ test_plan_quality (Plan Quality: 0.85, threshold: 0.7)
  ✅ test_plan_has_queries (Plan Quality: 0.90, threshold: 0.7)

tests/test_researcher.py
  ✅ test_research_grounded (Groundedness: 0.78, threshold: 0.7)
  ❌ test_research_edge_case (Groundedness: 0.45, threshold: 0.7)

tests/test_critic.py
  ✅ test_critique_approve (Critique Quality: 0.92, threshold: 0.7)
  ✅ test_critique_revise (Critique Quality: 0.88, threshold: 0.7)

tests/test_tools.py
  ✅ test_planner_tools (Tool Correctness: 1.0, threshold: 0.5)
  ✅ test_researcher_tools (Tool Correctness: 1.0, threshold: 0.5)
  ✅ test_supervisor_save (Tool Correctness: 1.0, threshold: 0.5)

tests/test_e2e.py
  ✅ test_golden_dataset [15/20 passed]
     Correctness: avg 0.74, min 0.42, max 0.95
     Answer Relevancy: avg 0.81, min 0.55, max 0.98
     Citation Presence: avg 0.70, min 0.30, max 1.00

======================================================
Overall: 19/20 passed (95.0% pass rate)
```

> Some tests may fail — that's normal. The goal isn't a 100% pass rate, but having a **baseline** for future improvements. Record the current scores and gradually improve the system.
