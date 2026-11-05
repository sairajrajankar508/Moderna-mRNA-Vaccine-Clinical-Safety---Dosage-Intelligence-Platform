"""Module 4a: the structured output schema every report must satisfy (FR-4.1).

The model is strict:
* ``extra="forbid"`` means a response with any additional field is rejected,
  so the report has exactly the five required fields.
* Every field is required, including ``Disclaimer`` (the PRD template gave it
  a default, which would let a response without a disclaimer validate).
* ``strict=True`` turns off type coercion: ``"true"`` (a string) is not
  accepted as a bool and ``"0.1"`` (a string) is not accepted as a float.
  A mistyped field raises ``pydantic.ValidationError`` instead of being
  silently converted.

Units: the field name fixed by the PRD is ``Recommended_Dose_mg`` (milligrams),
while the dataset and the guideline use micrograms. 1 mg = 1000 mcg.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

REQUIRED_DISCLAIMER = (
    "Student engineering project output - not real clinical guidance. "
    "Do not use for medical decisions."
)

MCG_PER_MG = 1000.0


class ClinicalSafetyReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    Eligible_For_Vaccine: bool = Field(
        ..., description="Whether the patient is eligible given risk and guidelines"
    )
    Recommended_Dose_mg: float = Field(
        ..., ge=0.0,
        description="Recommended dose in MILLIGRAMS (1 mg = 1000 micrograms); must respect the "
                    "retrieved guideline bounds; 0.0 when not eligible",
    )
    Reasoning_Steps: List[str] = Field(
        ..., min_length=1, description="Ordered, human-readable reasoning steps behind the decision"
    )
    FDA_Safety_Warnings_Cited: List[str] = Field(
        ..., description="Exact guideline sentences, copied verbatim from the excerpts, that the recommendation relies on"
    )
    Disclaimer: str = Field(
        ..., description=f'Must always be present; use exactly: "{REQUIRED_DISCLAIMER}"'
    )

    @field_validator("Reasoning_Steps", "FDA_Safety_Warnings_Cited")
    @classmethod
    def no_blank_items(cls, items: List[str]) -> List[str]:
        if any(not item.strip() for item in items):
            raise ValueError("list items must be non-empty strings")
        return items

    @field_validator("Disclaimer")
    @classmethod
    def disclaimer_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Disclaimer must not be empty")
        return value

    @property
    def recommended_dose_mcg(self) -> float:
        return round(self.Recommended_Dose_mg * MCG_PER_MG, 3)
