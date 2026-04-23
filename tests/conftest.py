"""Shared fixtures and helpers for all test modules."""
import sys
import os

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

android_judge = GEval(
    name="AndroidCodeRelevance",
    evaluation_steps=[
        "Check that the generated code addresses the requirements in the user story",
        "Check that the code is valid Android/Kotlin/Java syntax",
        "Check that acceptance criteria from the spec are reflected in the implementation",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)
