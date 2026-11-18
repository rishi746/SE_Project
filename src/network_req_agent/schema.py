from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RequirementCategory = Literal[
    "business_requirements",
    "functional_requirements",
    "security_requirements",
    "privacy_requirements",
    "performance_requirements",
    "availability_reliability_requirements",
    "audit_logging_requirements",
    "integration_requirements",
    "operational_maintenance_requirements",
    "constraints_assumptions",
]


class RequirementRecord(BaseModel):
    requirement_id: str
    statement: str
    category: RequirementCategory
    priority: Literal["Must", "Should", "Could"]
    risk_level: Literal["Low", "Medium", "High", "Critical"]
    acceptance_criteria: list[str] = Field(min_length=1)
    source_refs: list[str] = Field(min_length=1)


class SftOutput(BaseModel):
    project_type: str
    domain: Literal["networking"]
    requirements: list[RequirementRecord] = Field(min_length=8)
    ambiguities: list[str]
    clarification_questions: list[str] = Field(min_length=3)
    traceability_notes: list[str] = Field(min_length=1)
    human_approval_required: list[str] = Field(min_length=1)


class SftExample(BaseModel):
    id: str
    instruction: str
    input: str
    output: SftOutput
