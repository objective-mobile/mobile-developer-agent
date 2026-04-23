"""
Unit and LLM-as-a-Judge tests for the Developer agent (agents/developer.py).

Non-LLM tests use tmp_path fixture and mock _build_developer_agent to return
a fake agent that returns a pre-built CodeOutput. They verify:
  - Required Android files appear in files_created (AndroidManifest.xml,
    app/build.gradle, at least one .kt/.java source file).
  - All files_created paths exist on disk after execution.
  - gradle-wrapper.properties contains a valid Gradle distribution URL.
  - Single-Activity output for estimated_complexity == "simple".

LLM-as-a-Judge test (marked @pytest.mark.llm):
  - Uses deepeval GEval to verify that a CodeOutput covers all requirements
    listed in the input SpecOutput.
  - Validates: Requirements 10.2
"""
import os
import re
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from schemas import SpecOutput, CodeOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRADLE_URL_PATTERN = re.compile(
    r"https://services\.gradle\.org/distributions/gradle-[^/\s]+-bin\.zip"
)

ACTIVITY_EXTENDS_PATTERN = re.compile(
    r"class\s+\w+\s*:\s*(?:AppCompatActivity|Activity)\s*\(\s*\)",
    re.MULTILINE,
)


def _make_spec(**kwargs) -> SpecOutput:
    defaults = dict(
        title="My Todo App",
        requirements=["The app shall display a list of tasks."],
        acceptance_criteria=["Given the app is launched, items are shown."],
        estimated_complexity="simple",
    )
    defaults.update(kwargs)
    return SpecOutput(**defaults)


def _make_code_output(tmp_path: Path, slug: str = "my-todo-app") -> CodeOutput:
    """Build a realistic CodeOutput and write all files to tmp_path."""
    gradle_wrapper_content = (
        "distributionBase=GRADLE_USER_HOME\n"
        "distributionPath=wrapper/dists\n"
        "distributionUrl=https://services.gradle.org/distributions/gradle-8.7-bin.zip\n"
        "zipStoreBase=GRADLE_USER_HOME\n"
        "zipStorePath=wrapper/dists\n"
    )

    main_activity_content = (
        "package com.example.mytodoapp\n\n"
        "import android.os.Bundle\n"
        "import androidx.appcompat.app.AppCompatActivity\n\n"
        "class MainActivity : AppCompatActivity() {\n"
        "    override fun onCreate(savedInstanceState: Bundle?) {\n"
        "        super.onCreate(savedInstanceState)\n"
        "        setContentView(R.layout.activity_main)\n"
        "    }\n"
        "}\n"
    )

    files = {
        f"output/{slug}/build.gradle": "// root build.gradle\nbuildscript {}\n",
        f"output/{slug}/settings.gradle": f'rootProject.name = "{slug}"\ninclude(":app")\n',
        f"output/{slug}/gradle/wrapper/gradle-wrapper.properties": gradle_wrapper_content,
        f"output/{slug}/app/build.gradle": (
            "plugins {\n    id 'com.android.application'\n    id 'kotlin-android'\n}\n"
        ),
        f"output/{slug}/app/src/main/AndroidManifest.xml": (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<manifest xmlns:android="http://schemas.android.com/apk/res/android">\n'
            "    <application>\n"
            '        <activity android:name=".MainActivity" android:exported="true" />\n'
            "    </application>\n"
            "</manifest>\n"
        ),
        f"output/{slug}/app/src/main/java/com/example/mytodoapp/MainActivity.kt": main_activity_content,
        f"output/{slug}/app/src/main/res/layout/activity_main.xml": (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"\n'
            '    android:layout_width="match_parent"\n'
            '    android:layout_height="match_parent" />\n'
        ),
    }

    # Write all files to tmp_path
    for rel_path, content in files.items():
        full_path = tmp_path / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    # files_created uses absolute paths (as written to tmp_path)
    files_created = [str(tmp_path / rel_path) for rel_path in files]

    return CodeOutput(
        source_code=main_activity_content,
        description="A simple Android to-do list app.",
        files_created=files_created,
    )


def _make_agent_result(code_output: CodeOutput) -> dict:
    """Simulate the dict returned by developer_node with a given CodeOutput."""
    msg = MagicMock()
    msg.content = code_output
    return {"messages": [msg]}


# ---------------------------------------------------------------------------
# Unit tests — no real LLM calls
# ---------------------------------------------------------------------------

class TestDeveloperRequiredFiles:
    """Verify required Android files appear in files_created."""

    def test_android_manifest_in_files_created(self, tmp_path):
        code = _make_code_output(tmp_path)
        has_manifest = any("AndroidManifest.xml" in p for p in code.files_created)
        assert has_manifest, "files_created must contain AndroidManifest.xml"

    def test_app_build_gradle_in_files_created(self, tmp_path):
        code = _make_code_output(tmp_path)
        has_app_gradle = any("app/build.gradle" in p or "app\\build.gradle" in p for p in code.files_created)
        assert has_app_gradle, "files_created must contain app/build.gradle"

    def test_kotlin_or_java_source_in_files_created(self, tmp_path):
        code = _make_code_output(tmp_path)
        has_source = any(p.endswith(".kt") or p.endswith(".java") for p in code.files_created)
        assert has_source, "files_created must contain at least one .kt or .java source file"

    def test_developer_node_returns_required_files(self, tmp_path):
        """developer_node result must contain a CodeOutput with required Android files."""
        spec = _make_spec()
        code = _make_code_output(tmp_path)

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_agent_result(code)

        with patch("agents.developer._build_developer_agent", return_value=fake_agent):
            from agents.developer import developer_node
            result = developer_node({"spec": spec, "messages": []})

        assert "code" in result
        assert isinstance(result["code"], CodeOutput)

        files = result["code"].files_created
        assert any("AndroidManifest.xml" in p for p in files)
        assert any("app/build.gradle" in p or "app\\build.gradle" in p for p in files)
        assert any(p.endswith(".kt") or p.endswith(".java") for p in files)


class TestDeveloperFilesExistOnDisk:
    """Verify all files_created paths exist on disk after execution."""

    def test_all_files_created_exist_on_disk(self, tmp_path):
        code = _make_code_output(tmp_path)
        for path in code.files_created:
            assert os.path.exists(path), f"File listed in files_created does not exist: {path}"

    def test_developer_node_files_exist_on_disk(self, tmp_path):
        """After developer_node runs, every path in files_created must exist."""
        spec = _make_spec()
        code = _make_code_output(tmp_path)

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_agent_result(code)

        with patch("agents.developer._build_developer_agent", return_value=fake_agent):
            from agents.developer import developer_node
            result = developer_node({"spec": spec, "messages": []})

        for path in result["code"].files_created:
            assert os.path.exists(path), f"File does not exist on disk: {path}"

    def test_files_created_is_non_empty(self, tmp_path):
        code = _make_code_output(tmp_path)
        assert len(code.files_created) > 0, "files_created must not be empty"


class TestGradleWrapperProperties:
    """Verify gradle-wrapper.properties contains a valid Gradle distribution URL."""

    def _get_gradle_wrapper_content(self, tmp_path: Path) -> str:
        code = _make_code_output(tmp_path)
        # Find the gradle-wrapper.properties file
        wrapper_path = next(
            (p for p in code.files_created if "gradle-wrapper.properties" in p),
            None,
        )
        assert wrapper_path is not None, "gradle-wrapper.properties not found in files_created"
        return Path(wrapper_path).read_text(encoding="utf-8")

    def test_gradle_wrapper_properties_exists(self, tmp_path):
        code = _make_code_output(tmp_path)
        has_wrapper = any("gradle-wrapper.properties" in p for p in code.files_created)
        assert has_wrapper, "files_created must contain gradle-wrapper.properties"

    def test_gradle_wrapper_distribution_url_present(self, tmp_path):
        content = self._get_gradle_wrapper_content(tmp_path)
        assert "distributionUrl=" in content, "gradle-wrapper.properties must contain distributionUrl"

    def test_gradle_wrapper_distribution_url_matches_pattern(self, tmp_path):
        content = self._get_gradle_wrapper_content(tmp_path)
        url_value = None
        for line in content.splitlines():
            if line.startswith("distributionUrl="):
                url_value = line.split("=", 1)[1].strip()
                break
        assert url_value is not None, "distributionUrl key not found in gradle-wrapper.properties"
        match = GRADLE_URL_PATTERN.fullmatch(url_value)
        assert match is not None, (
            f"distributionUrl does not match expected pattern "
            f"'https://services.gradle.org/distributions/gradle-.*-bin.zip': {url_value!r}"
        )

    def test_developer_node_gradle_wrapper_url(self, tmp_path):
        """developer_node output must include a valid gradle-wrapper.properties URL."""
        spec = _make_spec()
        code = _make_code_output(tmp_path)

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_agent_result(code)

        with patch("agents.developer._build_developer_agent", return_value=fake_agent):
            from agents.developer import developer_node
            result = developer_node({"spec": spec, "messages": []})

        wrapper_path = next(
            (p for p in result["code"].files_created if "gradle-wrapper.properties" in p),
            None,
        )
        assert wrapper_path is not None, "gradle-wrapper.properties not in files_created"
        content = Path(wrapper_path).read_text(encoding="utf-8")
        url_value = None
        for line in content.splitlines():
            if line.startswith("distributionUrl="):
                url_value = line.split("=", 1)[1].strip()
                break
        assert url_value is not None
        assert GRADLE_URL_PATTERN.fullmatch(url_value), (
            f"Invalid distributionUrl: {url_value!r}"
        )


class TestSingleActivityForSimpleComplexity:
    """Verify single-Activity output for estimated_complexity == 'simple'."""

    def test_simple_complexity_has_exactly_one_activity(self, tmp_path):
        """For simple complexity, source_code must contain exactly one Activity class."""
        spec = _make_spec(estimated_complexity="simple")
        code = _make_code_output(tmp_path)

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_agent_result(code)

        with patch("agents.developer._build_developer_agent", return_value=fake_agent):
            from agents.developer import developer_node
            result = developer_node({"spec": spec, "messages": []})

        source = result["code"].source_code
        activity_matches = ACTIVITY_EXTENDS_PATTERN.findall(source)
        assert len(activity_matches) == 1, (
            f"Expected exactly 1 Activity class for simple complexity, "
            f"found {len(activity_matches)}: {activity_matches}"
        )

    def test_simple_complexity_source_extends_appcompat_or_activity(self, tmp_path):
        """The single Activity must extend AppCompatActivity or Activity."""
        code = _make_code_output(tmp_path)
        source = code.source_code
        assert ACTIVITY_EXTENDS_PATTERN.search(source), (
            "source_code must contain a class extending AppCompatActivity or Activity"
        )

    def test_source_code_contains_main_activity_class(self, tmp_path):
        code = _make_code_output(tmp_path)
        assert "MainActivity" in code.source_code, (
            "source_code should contain MainActivity class"
        )

    def test_developer_node_passes_spec_to_agent(self, tmp_path):
        """developer_node must forward the spec in the agent invocation."""
        spec = _make_spec(title="Weather App", estimated_complexity="simple")
        code = _make_code_output(tmp_path)

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_agent_result(code)

        with patch("agents.developer._build_developer_agent", return_value=fake_agent):
            from agents.developer import developer_node
            developer_node({"spec": spec, "messages": []})

        call_args = fake_agent.invoke.call_args
        messages = call_args[0][0]["messages"]
        assert any("Weather App" in str(m) for m in messages), (
            "Spec title should appear in the messages passed to the agent"
        )

    def test_developer_node_includes_qa_review_feedback(self, tmp_path):
        """When review feedback is present, it must be included in the agent prompt."""
        from schemas import ReviewOutput
        spec = _make_spec()
        code = _make_code_output(tmp_path)
        review = ReviewOutput(
            verdict="REVISION_NEEDED",
            issues=["Missing error handling in network calls"],
            suggestions=["Add try/catch around Retrofit calls"],
            score=0.4,
        )

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = _make_agent_result(code)

        with patch("agents.developer._build_developer_agent", return_value=fake_agent):
            from agents.developer import developer_node
            developer_node({"spec": spec, "review": review, "messages": []})

        call_args = fake_agent.invoke.call_args
        messages = call_args[0][0]["messages"]
        assert any("Missing error handling" in str(m) for m in messages), (
            "QA review issues should appear in the messages passed to the agent"
        )


# ---------------------------------------------------------------------------
# Sub-task 12.1 — LLM-as-a-Judge test
# Validates: Requirements 10.2
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_code_output_covers_all_spec_requirements():
    """
    LLM-as-a-Judge: verify that a generated CodeOutput covers all requirements
    listed in the input SpecOutput.

    Validates: Requirements 10.2
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping LLM judge test")

    from deepeval import assert_test
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    spec = SpecOutput(
        title="Android Login Screen",
        requirements=[
            "The app shall provide a login screen with email and password fields.",
            "The app shall validate that the email field is not empty before submission.",
            "The app shall display an error message when login fails.",
            "The app shall navigate to the home screen on successful login.",
        ],
        acceptance_criteria=[
            "Given the user enters valid credentials, when they tap Login, then they are navigated to HomeActivity.",
            "Given the user leaves the email field empty, when they tap Login, then an error 'Email is required' is shown.",
            "Given the server returns an error, when the user taps Login, then 'Login failed. Please try again.' is displayed.",
        ],
        estimated_complexity="simple",
    )

    # Realistic CodeOutput that addresses the spec requirements
    code_output = CodeOutput(
        source_code=(
            "package com.example.loginapp\n\n"
            "import android.content.Intent\n"
            "import android.os.Bundle\n"
            "import android.widget.Button\n"
            "import android.widget.EditText\n"
            "import android.widget.Toast\n"
            "import androidx.appcompat.app.AppCompatActivity\n\n"
            "class MainActivity : AppCompatActivity() {\n\n"
            "    private lateinit var emailField: EditText\n"
            "    private lateinit var passwordField: EditText\n"
            "    private lateinit var loginButton: Button\n\n"
            "    override fun onCreate(savedInstanceState: Bundle?) {\n"
            "        super.onCreate(savedInstanceState)\n"
            "        setContentView(R.layout.activity_main)\n\n"
            "        emailField = findViewById(R.id.emailField)\n"
            "        passwordField = findViewById(R.id.passwordField)\n"
            "        loginButton = findViewById(R.id.loginButton)\n\n"
            "        loginButton.setOnClickListener {\n"
            "            val email = emailField.text.toString()\n"
            "            val password = passwordField.text.toString()\n\n"
            "            if (email.isEmpty()) {\n"
            "                emailField.error = \"Email is required\"\n"
            "                return@setOnClickListener\n"
            "            }\n\n"
            "            performLogin(email, password)\n"
            "        }\n"
            "    }\n\n"
            "    private fun performLogin(email: String, password: String) {\n"
            "        // Simulate login — replace with real API call\n"
            "        if (email == \"user@example.com\" && password == \"password\") {\n"
            "            startActivity(Intent(this, HomeActivity::class.java))\n"
            "            finish()\n"
            "        } else {\n"
            "            Toast.makeText(this, \"Login failed. Please try again.\", Toast.LENGTH_SHORT).show()\n"
            "        }\n"
            "    }\n"
            "}\n"
        ),
        description=(
            "A simple Android login screen with email/password fields, "
            "input validation, error display, and navigation to HomeActivity on success."
        ),
        files_created=[
            "output/android-login-screen/app/src/main/AndroidManifest.xml",
            "output/android-login-screen/app/src/main/java/com/example/loginapp/MainActivity.kt",
            "output/android-login-screen/app/src/main/res/layout/activity_main.xml",
            "output/android-login-screen/app/build.gradle",
            "output/android-login-screen/build.gradle",
            "output/android-login-screen/settings.gradle",
            "output/android-login-screen/gradle/wrapper/gradle-wrapper.properties",
        ],
    )

    # Format spec requirements as the judge input
    spec_text = (
        f"Spec Title: {spec.title}\n\n"
        f"Requirements:\n" + "\n".join(f"- {r}" for r in spec.requirements) + "\n\n"
        f"Acceptance Criteria:\n" + "\n".join(f"- {ac}" for ac in spec.acceptance_criteria)
    )

    # Format code output as the judge actual output
    code_text = (
        f"Description: {code_output.description}\n\n"
        f"Source Code:\n{code_output.source_code}\n\n"
        f"Files Created:\n" + "\n".join(f"- {f}" for f in code_output.files_created)
    )

    coverage_judge = GEval(
        name="CodeOutput Requirements Coverage",
        evaluation_steps=[
            "Check that the generated code addresses each requirement listed in the spec.",
            "Check that the code implements the behaviour described in each acceptance criterion.",
            "Check that the code is valid Android/Kotlin syntax with proper class structure.",
            "Penalise implementations that ignore or skip any stated requirement.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model="gpt-4o-mini",
        threshold=0.7,
    )

    test_case = LLMTestCase(
        input=spec_text,
        actual_output=code_text,
    )

    assert_test(test_case, [coverage_judge])
