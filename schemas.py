from typing import Literal
from pydantic import BaseModel, Field, field_validator


class ResearchPlan(BaseModel):
    goal: str = Field(description="What we are trying to answer")
    search_queries: list[str] = Field(description="Specific queries to execute")
    sources_to_check: list[str] = Field(description="'knowledge_base', 'web', or both")
    output_format: str = Field(description="What the final report should look like")


class CritiqueResult(BaseModel):
    verdict: Literal["APPROVE", "REVISE"]
    is_fresh: bool = Field(description="Is the data up-to-date and based on recent sources?")
    is_complete: bool = Field(description="Does the research fully cover the user's original request?")
    is_well_structured: bool = Field(description="Are findings logically organized and ready for a report?")
    strengths: list[str] = Field(description="What is good about the research")
    gaps: list[str] = Field(description="What is missing, outdated, or poorly structured")
    revision_requests: list[str] = Field(description="Specific things to fix if verdict is REVISE")


class SpecOutput(BaseModel):
    title: str
    requirements: list[str]
    acceptance_criteria: list[str]
    estimated_complexity: Literal["simple", "medium", "complex"]


class CodeOutput(BaseModel):
    source_code: str
    description: str
    files_created: list[str]
    app_name: str = Field(default="", description="Human-readable app name")
    package_name: str = Field(default="", description="Android package name e.g. com.example.myapp")


class ReviewOutput(BaseModel):
    verdict: Literal["APPROVED", "REVISION_NEEDED"]
    issues: list[str]
    suggestions: list[str]
    score: float = Field(ge=0.0, le=1.0)

    @field_validator("issues")
    @classmethod
    def issues_required_on_revision(cls, v, info):
        if info.data.get("verdict") == "REVISION_NEEDED" and not v:
            raise ValueError("issues must be non-empty when verdict is REVISION_NEEDED")
        return v
