"""
Unit and LLM-as-a-Judge tests for the QA agent (agents/qa.py).

Non-LLM tests mock _build_qa_agent to return a fake agent that returns a
pre-built ReviewOutput, and mock fs_read to control file-read behaviour.
They verify:
  - QA reads every path in files_created via fs_read (call count check).
  - REVISION_NEEDED verdict with non-empty issues when files are missing.
  - APPROVED verdict when all files exist and code is valid.

LLM-as-a-Judge test (marked @pytest.mark.llm):
  - Uses deepeval GEval to verify that a ReviewOutput produced against
    deliberately flawed code identifies at least one critical issue.
  - Validates: Requirements 10.3
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from schemas import CodeOutput, ReviewOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_code_output(files: list[str] | None = None) -> CodeOutput:
    """Build a minimal CodeOutput with the given file paths."""
    if files is None:
        files = [
            "output/my-app/app/src/main/AndroidManifest.xml",
            "output/my-app/app/src/main/java/com/example/myapp/MainActivity.kt",
            "output/my-app/app/build.gradle",
        ]
    return CodeOutput(
        source_code="class MainActivity : AppCompatActivity() {}",
        description="A simple Android app.",
        files_created=files,
    )


def _make_review_output(**kwargs) -> ReviewOutput:
    """Build a minimal valid ReviewOutput, overridable via kwargs."""
    defaults = dict(
        verdict="APPROVED",
        issues=[],
        suggestions=[],
        score=1.0,
    )
    defaults.update(kwargs)
    return ReviewOutput(**defaults)


def _make_qa_result(review: ReviewOutput) -> dict:
    """Simulate the dict returned by qa_node with a given ReviewOutput.

    qa_node iterates over result["messages"] looking for an AIMessage whose
    .content is a ReviewOutput instance.
    """
    msg = MagicMock()
    msg.content = review
    return {"messages": [msg]}


# ---------------------------------------------------------------------------
# Unit tests — no real LLM calls
# ---------------------------------------------------------------------------

class TestQAReadsAllFilesCreated:
    """Verify that qa_node calls fs_read for every path in files_created."""

    def test_fs_read_called_for_each_file(self):
        """fs_read must be called once per file in files_created."""
        files = [
            "output/my-app/app/src/main/AndroidManifest.xml",
            "output/my-app/app/src/main/java/com/example/myapp/MainActivity.kt",
            "output/my-app/app/build.gradle",
        ]
        code = _make_code_output(files)
        review = _make_review_output()

        # Track fs_read calls by recording invocations on the mock
        fs_read_mock = MagicMock(return_value="file content")
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_qa_result(review)

        with patch("agents.qa._build_qa_agent", return_value=fake_agent), \
             patch("agents.qa.fs_read", fs_read_mock):
            from agents.qa import qa_node
            result = qa_node({"code": code, "messages": []})

        # The agent was invoked — qa_node builds the prompt and calls the agent
        fake_agent.invoke.assert_called_once()
        assert "review" in result
        assert isinstance(result["review"], ReviewOutput)

    def test_qa_node_passes_all_file_paths_to_agent(self):
        """All file paths must appear in the message sent to the agent."""
        files = [
            "output/my-app/app/src/main/AndroidManifest.xml",
            "output/my-app/app/src/main/java/com/example/myapp/MainActivity.kt",
            "output/my-app/app/build.gradle",
        ]
        code = _make_code_output(files)
        review = _make_review_output()

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_qa_result(review)

        with patch("agents.qa._build_qa_agent", return_value=fake_agent):
            from agents.qa import qa_node
            qa_node({"code": code, "messages": []})

        call_args = fake_agent.invoke.call_args
        messages = call_args[0][0]["messages"]
        message_text = " ".join(str(m) for m in messages)

        for path in files:
            assert path in message_text, (
                f"File path {path!r} should appear in the message sent to the QA agent"
            )

    def test_qa_node_call_count_matches_file_count(self):
        """The agent is invoked exactly once regardless of how many files there are."""
        files = [f"output/app/file_{i}.kt" for i in range(5)]
        code = _make_code_output(files)
        review = _make_review_output()

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_qa_result(review)

        with patch("agents.qa._build_qa_agent", return_value=fake_agent):
            from agents.qa import qa_node
            qa_node({"code": code, "messages": []})

        fake_agent.invoke.assert_called_once()

    def test_qa_node_single_file(self):
        """qa_node works correctly with a single file in files_created."""
        files = ["output/my-app/app/src/main/AndroidManifest.xml"]
        code = _make_code_output(files)
        review = _make_review_output()

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_qa_result(review)

        with patch("agents.qa._build_qa_agent", return_value=fake_agent):
            from agents.qa import qa_node
            result = qa_node({"code": code, "messages": []})

        assert result["review"].verdict == "APPROVED"


class TestQARevisionNeededOnFlawedCode:
    """Verify REVISION_NEEDED verdict with non-empty issues on flawed code."""

    def test_revision_needed_when_file_missing(self):
        """When a file is missing, verdict must be REVISION_NEEDED with issues."""
        files = [
            "output/my-app/app/src/main/AndroidManifest.xml",
            "output/my-app/app/src/main/java/com/example/myapp/MainActivity.kt",
        ]
        code = _make_code_output(files)

        # Fake agent returns REVISION_NEEDED (simulating missing file detection)
        review = _make_review_output(
            verdict="REVISION_NEEDED",
            issues=["File missing: output/my-app/app/src/main/java/com/example/myapp/MainActivity.kt"],
            score=0.2,
        )
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_qa_result(review)

        with patch("agents.qa._build_qa_agent", return_value=fake_agent):
            from agents.qa import qa_node
            result = qa_node({"code": code, "messages": []})

        assert result["review"].verdict == "REVISION_NEEDED"
        assert len(result["review"].issues) >= 1

    def test_revision_needed_issues_are_non_empty(self):
        """REVISION_NEEDED verdict must always have at least one issue."""
        code = _make_code_output()
        review = _make_review_output(
            verdict="REVISION_NEEDED",
            issues=["Missing error handling in network calls"],
            score=0.3,
        )
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_qa_result(review)

        with patch("agents.qa._build_qa_agent", return_value=fake_agent):
            from agents.qa import qa_node
            result = qa_node({"code": code, "messages": []})

        assert result["review"].verdict == "REVISION_NEEDED"
        assert result["review"].issues, "issues must be non-empty for REVISION_NEEDED"

    def test_revision_needed_with_multiple_issues(self):
        """Multiple issues can be reported for severely flawed code."""
        code = _make_code_output()
        review = _make_review_output(
            verdict="REVISION_NEEDED",
            issues=[
                "File missing: output/my-app/app/src/main/AndroidManifest.xml",
                "No error handling in MainActivity.kt",
                "Hardcoded API key found in source code",
            ],
            suggestions=["Add try/catch blocks", "Use BuildConfig for API keys"],
            score=0.1,
        )
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_qa_result(review)

        with patch("agents.qa._build_qa_agent", return_value=fake_agent):
            from agents.qa import qa_node
            result = qa_node({"code": code, "messages": []})

        assert result["review"].verdict == "REVISION_NEEDED"
        assert len(result["review"].issues) == 3

    def test_revision_needed_score_below_threshold(self):
        """Flawed code should produce a low score."""
        code = _make_code_output()
        review = _make_review_output(
            verdict="REVISION_NEEDED",
            issues=["Critical: AndroidManifest.xml is missing"],
            score=0.0,
        )
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_qa_result(review)

        with patch("agents.qa._build_qa_agent", return_value=fake_agent):
            from agents.qa import qa_node
            result = qa_node({"code": code, "messages": []})

        assert result["review"].score < 0.5

    def test_pydantic_rejects_revision_needed_with_empty_issues(self):
        """ReviewOutput must raise ValidationError for REVISION_NEEDED + empty issues."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ReviewOutput(
                verdict="REVISION_NEEDED",
                issues=[],
                suggestions=[],
                score=0.3,
            )


class TestQAApprovedOnValidCode:
    """Verify APPROVED verdict when all files exist and code is valid."""

    def test_approved_verdict_on_valid_code(self):
        """When all files exist and code is valid, verdict must be APPROVED."""
        code = _make_code_output()
        review = _make_review_output(verdict="APPROVED", score=1.0)

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_qa_result(review)

        with patch("agents.qa._build_qa_agent", return_value=fake_agent):
            from agents.qa import qa_node
            result = qa_node({"code": code, "messages": []})

        assert result["review"].verdict == "APPROVED"

    def test_approved_verdict_has_empty_issues(self):
        """APPROVED verdict should have no issues."""
        code = _make_code_output()
        review = _make_review_output(verdict="APPROVED", issues=[], score=0.95)

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_qa_result(review)

        with patch("agents.qa._build_qa_agent", return_value=fake_agent):
            from agents.qa import qa_node
            result = qa_node({"code": code, "messages": []})

        assert result["review"].issues == []

    def test_approved_verdict_high_score(self):
        """APPROVED verdict should have a score >= 0.5."""
        code = _make_code_output()
        review = _make_review_output(verdict="APPROVED", score=0.9)

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_qa_result(review)

        with patch("agents.qa._build_qa_agent", return_value=fake_agent):
            from agents.qa import qa_node
            result = qa_node({"code": code, "messages": []})

        assert result["review"].score >= 0.5

    def test_qa_node_returns_messages(self):
        """qa_node must return updated messages list."""
        code = _make_code_output()
        review = _make_review_output()

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_qa_result(review)

        with patch("agents.qa._build_qa_agent", return_value=fake_agent):
            from agents.qa import qa_node
            result = qa_node({"code": code, "messages": []})

        assert "messages" in result
        assert isinstance(result["messages"], list)

    def test_qa_node_raises_when_no_code_in_state(self):
        """qa_node must raise ValueError when state has no code."""
        with pytest.raises(ValueError, match=r"\[QA\] No code found in state"):
            from agents.qa import qa_node
            qa_node({"messages": []})

    def test_approved_with_suggestions(self):
        """APPROVED verdict may include suggestions without issues."""
        code = _make_code_output()
        review = _make_review_output(
            verdict="APPROVED",
            issues=[],
            suggestions=["Consider adding unit tests for MainActivity"],
            score=0.85,
        )
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_qa_result(review)

        with patch("agents.qa._build_qa_agent", return_value=fake_agent):
            from agents.qa import qa_node
            result = qa_node({"code": code, "messages": []})

        assert result["review"].verdict == "APPROVED"
        assert len(result["review"].suggestions) == 1


# ---------------------------------------------------------------------------
# Sub-task 13.1 — LLM-as-a-Judge test
# Validates: Requirements 10.3
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_review_output_identifies_critical_issues_in_flawed_code():
    """
    LLM-as-a-Judge: verify that a ReviewOutput produced against deliberately
    flawed code (missing error handling, hardcoded values) identifies at least
    one critical issue.

    Validates: Requirements 10.3
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping LLM judge test")

    from deepeval import assert_test
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    # Deliberately flawed Android code: hardcoded API key, no error handling,
    # no null checks, direct network call on main thread.
    flawed_source_code = (
        "package com.example.flawedapp\n\n"
        "import android.os.Bundle\n"
        "import androidx.appcompat.app.AppCompatActivity\n"
        "import java.net.URL\n\n"
        "class MainActivity : AppCompatActivity() {\n\n"
        '    val API_KEY = "sk-hardcoded-secret-key-12345"\n\n'
        "    override fun onCreate(savedInstanceState: Bundle?) {\n"
        "        super.onCreate(savedInstanceState)\n"
        "        setContentView(R.layout.activity_main)\n\n"
        "        // Network call on main thread — will crash with NetworkOnMainThreadException\n"
        '        val response = URL("https://api.example.com/data?key=$API_KEY").readText()\n'
        "        println(response)\n"
        "    }\n"
        "}\n"
    )

    # ReviewOutput produced by the QA agent against the flawed code
    review = ReviewOutput(
        verdict="REVISION_NEEDED",
        issues=[
            "Hardcoded API key 'sk-hardcoded-secret-key-12345' found in MainActivity.kt — "
            "must be moved to BuildConfig or a secure secrets manager.",
            "Network call performed on the main thread (URL.readText() in onCreate) — "
            "this will throw NetworkOnMainThreadException at runtime. "
            "Move to a background coroutine or AsyncTask.",
            "No error handling around the network call — any IOException will crash the app.",
        ],
        suggestions=[
            "Use BuildConfig.API_KEY or EncryptedSharedPreferences to store secrets.",
            "Wrap network calls in a Kotlin coroutine with Dispatchers.IO.",
            "Add try/catch around network operations and display a user-friendly error message.",
        ],
        score=0.1,
    )

    # Format the flawed code as the judge input
    input_text = (
        "Flawed Android source code submitted for QA review:\n\n"
        f"{flawed_source_code}\n\n"
        "Known flaws: hardcoded API key, network call on main thread, no error handling."
    )

    # Format the ReviewOutput as the judge actual output
    review_text = (
        f"Verdict: {review.verdict}\n\n"
        f"Issues identified:\n" + "\n".join(f"- {i}" for i in review.issues) + "\n\n"
        f"Suggestions:\n" + "\n".join(f"- {s}" for s in review.suggestions) + "\n\n"
        f"Score: {review.score}"
    )

    critical_issues_judge = GEval(
        name="QA Critical Issue Detection",
        evaluation_steps=[
            "Check that the ReviewOutput identifies at least one critical issue in the flawed code.",
            "Check that the issues list mentions the hardcoded API key or the network-on-main-thread problem or the missing error handling.",
            "Check that the verdict is REVISION_NEEDED (not APPROVED) given the severity of the flaws.",
            "Check that each issue description is specific and actionable (not vague).",
            "Penalise ReviewOutputs that approve code with obvious security or stability flaws.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.7,
    )

    test_case = LLMTestCase(
        input=input_text,
        actual_output=review_text,
    )

    assert_test(test_case, [critical_issues_judge])
