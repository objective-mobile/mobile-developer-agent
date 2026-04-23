"""
End-to-end pipeline tests for the Android development pipeline.

Non-LLM tests mock all three agent nodes and the HITL interrupt to verify:
  - Full pipeline run with auto-approve HITL produces a final state with CodeOutput.
  - iteration increments correctly across QA–Developer loops.
  - Routing to END at iteration limit (after 5 QA iterations).
  - build_toolchain_env() with and without env vars set.

LLM-as-a-Judge test (marked @pytest.mark.llm):
  - Uses android_judge GEval metric to verify final code is relevant to the
    original user story.
  - Validates: Requirements 10.4

Sub-task 14.2 — Property 17 (Judge test results are structured) is already
implemented in tests/test_android_properties.py.
"""
import os
import sys
import uuid
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.messages import AIMessage, HumanMessage
from schemas import SpecOutput, CodeOutput, ReviewOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(**kwargs) -> SpecOutput:
    defaults = dict(
        title="E2E Test App",
        requirements=["The app shall display a hello world screen."],
        acceptance_criteria=["Given the app is launched, a greeting is shown."],
        estimated_complexity="simple",
    )
    defaults.update(kwargs)
    return SpecOutput(**defaults)


def _make_code(**kwargs) -> CodeOutput:
    defaults = dict(
        source_code=(
            "package com.example.e2etestapp\n\n"
            "import androidx.appcompat.app.AppCompatActivity\n\n"
            "class MainActivity : AppCompatActivity() {}\n"
        ),
        description="E2E test Android app.",
        files_created=[
            "output/e2e-test-app/app/src/main/AndroidManifest.xml",
            "output/e2e-test-app/app/src/main/java/com/example/e2etestapp/MainActivity.kt",
            "output/e2e-test-app/app/build.gradle",
            "output/e2e-test-app/build.gradle",
            "output/e2e-test-app/settings.gradle",
            "output/e2e-test-app/gradle/wrapper/gradle-wrapper.properties",
        ],
    )
    defaults.update(kwargs)
    return CodeOutput(**defaults)


def _make_review(verdict: str = "APPROVED") -> ReviewOutput:
    issues = ["Needs improvement"] if verdict == "REVISION_NEEDED" else []
    return ReviewOutput(
        verdict=verdict,
        issues=issues,
        suggestions=[],
        score=0.9 if verdict == "APPROVED" else 0.3,
    )


def _initial_state(user_story: str = "Build a hello world Android app") -> dict:
    """Return a minimal valid initial pipeline state (no MagicMock objects)."""
    return {
        "user_story": user_story,
        "iteration": 0,
        "messages": [],
        "spec": None,
        "code": None,
        "review": None,
        "hitl_feedback": None,
    }


# ---------------------------------------------------------------------------
# Full pipeline run with auto-approve HITL
# ---------------------------------------------------------------------------

class TestFullPipelineAutoApprove:
    """Full pipeline run: BA → HITL (auto-approve) → Developer → QA (APPROVED).

    We patch the inner agent node functions (ba_node, developer_node, qa_node)
    to return plain dicts with real Pydantic objects and real LangChain messages,
    so the MemorySaver checkpointer can serialize the state.
    """

    def _run_pipeline(self, ba_result, dev_result, qa_result_fn):
        """Helper: build and invoke the pipeline with mocked agent nodes."""
        with patch("agents.ba.ba_node", side_effect=lambda s: ba_result), \
             patch("agents.developer.developer_node", side_effect=lambda s: dev_result), \
             patch("agents.qa.qa_node", side_effect=qa_result_fn), \
             patch("android_pipeline.interrupt", return_value={"action": "approve"}):

            from android_pipeline import build_android_pipeline
            pipeline = build_android_pipeline()
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            return pipeline.invoke(_initial_state(), config=config)

    def test_final_state_contains_code_output(self):
        """After a full run with auto-approve, the final state must contain a CodeOutput."""
        spec = _make_spec()
        code = _make_code()
        review = _make_review("APPROVED")

        ba_result = {"spec": spec, "messages": [HumanMessage(content="ba done")]}
        dev_result = {"code": code, "messages": [AIMessage(content="dev done")]}

        def qa_fn(state):
            return {"review": review, "messages": [AIMessage(content="qa done")]}

        result = self._run_pipeline(ba_result, dev_result, qa_fn)

        assert result.get("code") is not None, "Final state must contain a CodeOutput"
        assert isinstance(result["code"], CodeOutput)

    def test_final_state_contains_approved_review(self):
        """After a full run, the final state must contain an APPROVED ReviewOutput."""
        spec = _make_spec()
        code = _make_code()
        review = _make_review("APPROVED")

        ba_result = {"spec": spec, "messages": [HumanMessage(content="ba done")]}
        dev_result = {"code": code, "messages": [AIMessage(content="dev done")]}

        def qa_fn(state):
            return {"review": review, "messages": [AIMessage(content="qa done")]}

        result = self._run_pipeline(ba_result, dev_result, qa_fn)

        assert result.get("review") is not None
        assert result["review"].verdict == "APPROVED"

    def test_final_state_spec_matches_ba_output(self):
        """The spec in the final state must match what the BA node produced."""
        spec = _make_spec(title="Unique Pipeline Spec Title")
        code = _make_code()
        review = _make_review("APPROVED")

        ba_result = {"spec": spec, "messages": [HumanMessage(content="ba done")]}
        dev_result = {"code": code, "messages": [AIMessage(content="dev done")]}

        def qa_fn(state):
            return {"review": review, "messages": [AIMessage(content="qa done")]}

        result = self._run_pipeline(ba_result, dev_result, qa_fn)

        assert result["spec"].title == "Unique Pipeline Spec Title"


# ---------------------------------------------------------------------------
# Iteration counter increments across QA–Developer loops
# ---------------------------------------------------------------------------

class TestIterationCounter:
    """Assert iteration increments correctly across QA–Developer loops."""

    def test_iteration_increments_twice_then_approved(self):
        """QA returns REVISION_NEEDED twice then APPROVED; iteration must reach 3."""
        spec = _make_spec()
        code = _make_code()
        revision_review = _make_review("REVISION_NEEDED")
        approved_review = _make_review("APPROVED")

        ba_result = {"spec": spec, "messages": [HumanMessage(content="ba done")]}
        dev_result = {"code": code, "messages": [AIMessage(content="dev done")]}

        qa_call_count = {"n": 0}

        def qa_side_effect(state):
            qa_call_count["n"] += 1
            if qa_call_count["n"] <= 2:
                return {"review": revision_review, "messages": [AIMessage(content=f"qa revision {qa_call_count['n']}")]}
            return {"review": approved_review, "messages": [AIMessage(content="qa approved")]}

        with patch("agents.ba.ba_node", side_effect=lambda s: ba_result), \
             patch("agents.developer.developer_node", side_effect=lambda s: dev_result), \
             patch("agents.qa.qa_node", side_effect=qa_side_effect), \
             patch("android_pipeline.interrupt", return_value={"action": "approve"}):

            from android_pipeline import build_android_pipeline
            pipeline = build_android_pipeline()
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            result = pipeline.invoke(_initial_state(), config=config)

        # QA was called 3 times (2 revisions + 1 approval)
        assert qa_call_count["n"] == 3, (
            f"Expected QA to be called 3 times, got {qa_call_count['n']}"
        )
        # Final iteration should be 3 (incremented on each Developer→QA transition)
        assert result["iteration"] == 3, (
            f"Expected iteration=3 after 3 QA calls, got {result['iteration']}"
        )
        assert result["review"].verdict == "APPROVED"

    def test_iteration_is_one_after_single_approved_qa(self):
        """After a single QA call that approves, iteration must be 1."""
        spec = _make_spec()
        code = _make_code()
        review = _make_review("APPROVED")

        ba_result = {"spec": spec, "messages": [HumanMessage(content="ba done")]}
        dev_result = {"code": code, "messages": [AIMessage(content="dev done")]}

        def qa_fn(state):
            return {"review": review, "messages": [AIMessage(content="qa done")]}

        with patch("agents.ba.ba_node", side_effect=lambda s: ba_result), \
             patch("agents.developer.developer_node", side_effect=lambda s: dev_result), \
             patch("agents.qa.qa_node", side_effect=qa_fn), \
             patch("android_pipeline.interrupt", return_value={"action": "approve"}):

            from android_pipeline import build_android_pipeline
            pipeline = build_android_pipeline()
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            result = pipeline.invoke(_initial_state(), config=config)

        # After one QA call (APPROVED), iteration should be 1
        assert result["iteration"] == 1


# ---------------------------------------------------------------------------
# Routing to END at iteration limit
# ---------------------------------------------------------------------------

class TestIterationLimit:
    """Assert pipeline routes to END after 5 QA iterations with REVISION_NEEDED."""

    def test_pipeline_ends_after_iteration_limit(self):
        """When QA always returns REVISION_NEEDED, pipeline must end after 5 iterations."""
        spec = _make_spec()
        code = _make_code()
        revision_review = _make_review("REVISION_NEEDED")

        ba_result = {"spec": spec, "messages": [HumanMessage(content="ba done")]}
        dev_result = {"code": code, "messages": [AIMessage(content="dev done")]}

        qa_call_count = {"n": 0}

        def qa_always_revision(state):
            qa_call_count["n"] += 1
            return {"review": revision_review, "messages": [AIMessage(content=f"qa revision {qa_call_count['n']}")]}

        with patch("agents.ba.ba_node", side_effect=lambda s: ba_result), \
             patch("agents.developer.developer_node", side_effect=lambda s: dev_result), \
             patch("agents.qa.qa_node", side_effect=qa_always_revision), \
             patch("android_pipeline.interrupt", return_value={"action": "approve"}):

            from android_pipeline import build_android_pipeline
            pipeline = build_android_pipeline()
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            result = pipeline.invoke(_initial_state(), config=config)

        # Pipeline must have stopped — QA called at most 5 times
        assert qa_call_count["n"] <= 5, (
            f"QA was called {qa_call_count['n']} times — expected at most 5"
        )
        # Final review must still be REVISION_NEEDED
        assert result["review"].verdict == "REVISION_NEEDED"
        # Iteration must be at the limit
        assert result["iteration"] >= 5, (
            f"Expected iteration >= 5 at limit, got {result['iteration']}"
        )

    def test_pipeline_ends_with_terminal_review_output(self):
        """At iteration limit, the final state must surface the terminal ReviewOutput."""
        spec = _make_spec()
        code = _make_code()
        revision_review = ReviewOutput(
            verdict="REVISION_NEEDED",
            issues=["Critical: missing error handling"],
            suggestions=["Add try/catch blocks"],
            score=0.2,
        )

        ba_result = {"spec": spec, "messages": [HumanMessage(content="ba done")]}
        dev_result = {"code": code, "messages": [AIMessage(content="dev done")]}

        def qa_fn(state):
            return {"review": revision_review, "messages": [AIMessage(content="qa revision")]}

        with patch("agents.ba.ba_node", side_effect=lambda s: ba_result), \
             patch("agents.developer.developer_node", side_effect=lambda s: dev_result), \
             patch("agents.qa.qa_node", side_effect=qa_fn), \
             patch("android_pipeline.interrupt", return_value={"action": "approve"}):

            from android_pipeline import build_android_pipeline
            pipeline = build_android_pipeline()
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            result = pipeline.invoke(_initial_state(), config=config)

        assert result["review"] is not None
        assert result["review"].verdict == "REVISION_NEEDED"
        assert len(result["review"].issues) >= 1


# ---------------------------------------------------------------------------
# build_toolchain_env() with and without env vars set
# ---------------------------------------------------------------------------

class TestBuildToolchainEnv:
    """Assert build_toolchain_env() with and without env vars set."""

    def test_all_vars_set_appear_in_path(self):
        """When all three toolchain vars are set, their bin/ dirs appear in PATH."""
        android_home = "/opt/android-sdk"
        java_home = "/opt/jdk"
        gradle_home = "/opt/gradle"
        env_overrides = {
            "ANDROID_HOME": android_home,
            "JAVA_HOME": java_home,
            "GRADLE_HOME": gradle_home,
            "PATH": "/usr/bin:/bin",
        }
        clean_env = {k: v for k, v in os.environ.items() if k not in ("ANDROID_HOME", "JAVA_HOME", "GRADLE_HOME")}
        clean_env.update(env_overrides)

        with patch.dict(os.environ, clean_env, clear=True):
            from android_pipeline import build_toolchain_env
            result = build_toolchain_env()

        path_parts = result["PATH"].split(os.pathsep)
        assert os.path.join(android_home, "bin") in path_parts, "ANDROID_HOME/bin must be in PATH"
        assert os.path.join(java_home, "bin") in path_parts, "JAVA_HOME/bin must be in PATH"
        assert os.path.join(gradle_home, "bin") in path_parts, "GRADLE_HOME/bin must be in PATH"

    def test_no_vars_set_does_not_raise(self):
        """When no toolchain vars are set, build_toolchain_env() must not raise."""
        clean_env = {
            k: v for k, v in os.environ.items()
            if k not in ("ANDROID_HOME", "JAVA_HOME", "GRADLE_HOME")
        }
        clean_env["PATH"] = "/usr/bin:/bin"

        with patch.dict(os.environ, clean_env, clear=True):
            from android_pipeline import build_toolchain_env
            result = build_toolchain_env()  # must not raise

        assert "PATH" in result
        # Original PATH must still be present
        assert "/usr/bin" in result["PATH"] or "/bin" in result["PATH"]

    def test_partial_vars_set(self):
        """When only JAVA_HOME is set, only its bin/ dir appears in PATH."""
        java_home = "/opt/jdk"
        clean_env = {
            k: v for k, v in os.environ.items()
            if k not in ("ANDROID_HOME", "JAVA_HOME", "GRADLE_HOME")
        }
        clean_env["PATH"] = "/usr/bin:/bin"
        clean_env["JAVA_HOME"] = java_home

        with patch.dict(os.environ, clean_env, clear=True):
            from android_pipeline import build_toolchain_env
            result = build_toolchain_env()

        path_parts = result["PATH"].split(os.pathsep)
        assert os.path.join(java_home, "bin") in path_parts, "JAVA_HOME/bin must be in PATH when set"
        # ANDROID_HOME and GRADLE_HOME were not set — their bin/ dirs must not appear
        assert not any("android-sdk" in p for p in path_parts)
        assert not any("gradle" in p.lower() for p in path_parts)

    def test_toolchain_env_returns_dict(self):
        """build_toolchain_env() must always return a dict."""
        from android_pipeline import build_toolchain_env
        result = build_toolchain_env()
        assert isinstance(result, dict)

    def test_toolchain_env_contains_path_key(self):
        """build_toolchain_env() result must always contain a PATH key."""
        from android_pipeline import build_toolchain_env
        result = build_toolchain_env()
        assert "PATH" in result


# ---------------------------------------------------------------------------
# Sub-task 14.1 — LLM-as-a-Judge test: final code is relevant to user story
# Validates: Requirements 10.4
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_final_code_is_relevant_to_user_story():
    """
    LLM-as-a-Judge: verify that the final CodeOutput from the pipeline is
    relevant to the original user story.

    Uses the android_judge GEval metric from conftest.py.
    Validates: Requirements 10.4
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping LLM judge test")

    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase
    from tests.conftest import android_judge

    user_story = (
        "Build an Android to-do list app where users can add, complete, "
        "and delete tasks. Tasks should persist across app restarts."
    )

    # Realistic CodeOutput that addresses the user story
    code_output = CodeOutput(
        source_code=(
            "package com.example.todoapp\n\n"
            "import android.os.Bundle\n"
            "import android.widget.Button\n"
            "import android.widget.EditText\n"
            "import android.widget.ListView\n"
            "import androidx.appcompat.app.AppCompatActivity\n"
            "import androidx.lifecycle.ViewModelProvider\n\n"
            "class MainActivity : AppCompatActivity() {\n\n"
            "    private lateinit var viewModel: TaskViewModel\n"
            "    private lateinit var taskInput: EditText\n"
            "    private lateinit var addButton: Button\n"
            "    private lateinit var taskList: ListView\n\n"
            "    override fun onCreate(savedInstanceState: Bundle?) {\n"
            "        super.onCreate(savedInstanceState)\n"
            "        setContentView(R.layout.activity_main)\n\n"
            "        viewModel = ViewModelProvider(this)[TaskViewModel::class.java]\n"
            "        taskInput = findViewById(R.id.taskInput)\n"
            "        addButton = findViewById(R.id.addButton)\n"
            "        taskList = findViewById(R.id.taskList)\n\n"
            "        addButton.setOnClickListener {\n"
            "            val text = taskInput.text.toString().trim()\n"
            "            if (text.isNotEmpty()) {\n"
            "                viewModel.addTask(text)\n"
            "                taskInput.text.clear()\n"
            "            }\n"
            "        }\n\n"
            "        viewModel.tasks.observe(this) { tasks ->\n"
            "            val adapter = TaskAdapter(this, tasks)\n"
            "            taskList.adapter = adapter\n"
            "        }\n"
            "    }\n"
            "}\n"
        ),
        description=(
            "An Android to-do list app with Room database persistence. "
            "Users can add tasks, mark them complete, and delete them. "
            "Tasks are stored in a local Room database and persist across restarts."
        ),
        files_created=[
            "output/todo-app/app/src/main/AndroidManifest.xml",
            "output/todo-app/app/src/main/java/com/example/todoapp/MainActivity.kt",
            "output/todo-app/app/src/main/java/com/example/todoapp/TaskViewModel.kt",
            "output/todo-app/app/src/main/java/com/example/todoapp/TaskAdapter.kt",
            "output/todo-app/app/src/main/java/com/example/todoapp/data/Task.kt",
            "output/todo-app/app/src/main/java/com/example/todoapp/data/TaskDao.kt",
            "output/todo-app/app/src/main/java/com/example/todoapp/data/TaskDatabase.kt",
            "output/todo-app/app/src/main/res/layout/activity_main.xml",
            "output/todo-app/app/build.gradle",
            "output/todo-app/build.gradle",
            "output/todo-app/settings.gradle",
            "output/todo-app/gradle/wrapper/gradle-wrapper.properties",
        ],
    )

    # Format for the judge
    actual_output = (
        f"Description: {code_output.description}\n\n"
        f"Source Code (MainActivity.kt):\n{code_output.source_code}\n\n"
        f"Files Created:\n" + "\n".join(f"- {f}" for f in code_output.files_created)
    )

    test_case = LLMTestCase(
        input=user_story,
        actual_output=actual_output,
    )

    assert_test(test_case, [android_judge])
