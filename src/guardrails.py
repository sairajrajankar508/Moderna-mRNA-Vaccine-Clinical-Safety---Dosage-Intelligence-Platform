"""Module 4b: deterministic safety checks applied after the LLM (FR-4.4).

The LLM writes the first draft of the report. These checks then run on every
validated report, whatever produced it, and they can only make the result
MORE conservative: they can switch Eligible_For_Vaccine to false and lower
the dose, never the opposite.

Every rule is taken from a specific clause of mrna_safety_guidelines.txt and
cites that clause's exact sentence. One number is a project assumption, not a
guideline fact: section 2.3 says "meaningfully above the normal reference
range" for the inflammatory biomarker but gives no value, so
ELEVATED_BIOMARKER_MG_L is configurable and reports say it is an assumption.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

from config import GUIDELINES_TXT
from schemas import MCG_PER_MG, REQUIRED_DISCLAIMER, ClinicalSafetyReport

# Project assumption (see module docstring). Override with the env var of the same name.
ELEVATED_BIOMARKER_MG_L = float(os.getenv("ELEVATED_BIOMARKER_MG_L", "10.0"))


@dataclass
class PatientProfile:
    patient_id: str
    age: int
    biomarker_level: float
    pre_existing_conditions: int
    vaccine_dose_mcg: float
    prior_reaction_history: str

    @classmethod
    def from_record(cls, record: dict) -> "PatientProfile":
        return cls(
            patient_id=str(record.get("Patient_ID", "CUSTOM")),
            age=int(record["Age"]),
            biomarker_level=float(record["Biomarker_Level"]),
            pre_existing_conditions=int(record["Pre_Existing_Conditions"]),
            vaccine_dose_mcg=float(record["Vaccine_Dose_mcg"]),
            prior_reaction_history=str(record["Prior_Reaction_History"]),
        )

    def to_record(self) -> dict:
        return {
            "Patient_ID": self.patient_id,
            "Age": self.age,
            "Biomarker_Level": self.biomarker_level,
            "Pre_Existing_Conditions": self.pre_existing_conditions,
            "Vaccine_Dose_mcg": self.vaccine_dose_mcg,
            "Prior_Reaction_History": self.prior_reaction_history,
        }


# ---------------------------------------------------------------------------
# Reading facts out of guideline text
# ---------------------------------------------------------------------------

def guideline_sentences(text: str) -> list[str]:
    """Split guideline text into sentences/bullets, stripping bullet markers."""
    pieces = []
    for line in text.splitlines():
        line = line.strip().lstrip("-").strip()
        if line:
            pieces.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if s.strip())
    return pieces


def find_sentence(texts: list[str], pattern: str) -> str | None:
    """First sentence across texts matching the regex (case-insensitive)."""
    regex = re.compile(pattern, re.IGNORECASE)
    for text in texts:
        for sentence in guideline_sentences(text):
            if regex.search(sentence):
                return sentence
    return None


@lru_cache(maxsize=1)
def full_guideline_text() -> str:
    return GUIDELINES_TXT.read_text(encoding="utf-8")


@dataclass
class DoseRange:
    low_mcg: float
    high_mcg: float
    clause: str  # the part of the sentence that states this range
    sentence: str  # the full guideline sentence, used for citations


@dataclass
class DoseLimits:
    """Dose facts parsed from retrieved guideline text (all micrograms)."""

    ranges: list[DoseRange] = field(default_factory=list)
    review_above_mcg: float | None = None
    review_sentence: str | None = None

    @property
    def max_safe_mcg(self) -> float | None:
        """Highest dose the automated pipeline may recommend without clinician review."""
        if self.review_above_mcg is not None:
            return self.review_above_mcg
        return max(r.high_mcg for r in self.ranges) if self.ranges else None

    @property
    def primary_range(self) -> DoseRange | None:
        return max(self.ranges, key=lambda r: r.high_mcg) if self.ranges else None

    @property
    def lower_reactogenicity_range(self) -> DoseRange | None:
        return next((r for r in self.ranges if re.search(r"lower-reactogenicity|booster", r.clause, re.I)), None)


_RANGE = re.compile(r"between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?) micrograms", re.I)
_ABOVE = re.compile(r"doses? above (\d+(?:\.\d+)?) micrograms", re.I)


def extract_dose_limits(chunks: list[str]) -> DoseLimits:
    """Parse microgram dose ranges and the review threshold from retrieved text."""
    limits = DoseLimits()
    for sentence in (s for chunk in chunks for s in guideline_sentences(chunk)):
        # "X fall between 25 and 100 micrograms, while Y are dosed between 25 and 50" -> two clauses.
        for clause in re.split(r",\s*while\s+", sentence):
            for low, high in _RANGE.findall(clause):
                if not any((r.low_mcg, r.high_mcg) == (float(low), float(high)) for r in limits.ranges):
                    limits.ranges.append(DoseRange(float(low), float(high), clause, sentence))
        match = _ABOVE.search(sentence)
        if match and limits.review_above_mcg is None:
            limits.review_above_mcg = float(match.group(1))
            limits.review_sentence = sentence
    return limits


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    rule_id: str
    clause: str  # guideline section
    quote_pattern: str  # regex locating the exact sentence to cite
    applies: Callable[[PatientProfile, DoseLimits], bool]
    explanation: Callable[[PatientProfile, DoseLimits], str]


RULES = [
    Rule("pediatric_scope", "2.2", r"Ages 5 to 11 years",
         lambda p, _: p.age < 12,
         lambda p, _: f"Patient is {p.age}; paediatric dose tiers and formulations are outside this pipeline's adult scope."),
    Rule("candidate_dose_above_review_threshold", "2.1", r"Doses above \d+ micrograms",
         lambda p, lim: lim.review_above_mcg is not None and p.vaccine_dose_mcg > lim.review_above_mcg,
         lambda p, lim: f"Candidate dose {p.vaccine_dose_mcg:g} mcg exceeds {lim.review_above_mcg:g} mcg, "
                        "so it requires secondary clinician review before administration."),
    Rule("age_65_plus_with_2_conditions", "2.2", r"Ages 65 and older",
         lambda p, _: p.age >= 65 and p.pre_existing_conditions >= 2,
         lambda p, _: f"Age {p.age} with {p.pre_existing_conditions} pre-existing conditions must be flagged "
                      "for secondary clinician review regardless of dose."),
    Rule("three_or_more_conditions", "3.2", r"Three or more relevant pre-existing conditions",
         lambda p, _: p.pre_existing_conditions >= 3,
         lambda p, _: f"{p.pre_existing_conditions} pre-existing conditions (3 or more) is a precaution requiring secondary review."),
    Rule("prior_moderate_reaction", "3.2", r"Prior history of a moderate reaction",
         lambda p, _: p.prior_reaction_history.lower() == "moderate",
         lambda p, _: "A prior moderate reaction to an mRNA product is a precaution requiring secondary review."),
    Rule("elevated_biomarker", "2.3", r"inflammatory biomarker level meaningfully above",
         lambda p, _: p.biomarker_level > ELEVATED_BIOMARKER_MG_L,
         lambda p, _: f"Biomarker {p.biomarker_level:g} mg/L is above the project's configured threshold of "
                      f"{ELEVATED_BIOMARKER_MG_L:g} mg/L (a project assumption; the guideline gives no number), "
                      "so dosing is deferred, the conservative option in section 2.3."),
]


@dataclass
class GuardrailOutcome:
    report: ClinicalSafetyReport
    actions: list[str]  # human-readable list of what the guardrails changed
    rules_triggered: list[str]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip('"').lstrip("-").strip().lower()


def is_grounded(citation: str, chunks: list[str]) -> bool:
    """True if the citation appears verbatim (ignoring whitespace/case) in a retrieved chunk."""
    needle = _normalize(citation)
    return len(needle) >= 20 and any(needle in _normalize(chunk) for chunk in chunks)


def enforce(report: ClinicalSafetyReport, patient: PatientProfile, chunks: list[str]) -> GuardrailOutcome:
    """Apply every safety check to a validated report and return the corrected copy."""
    eligible = report.Eligible_For_Vaccine
    dose_mg = report.Recommended_Dose_mg
    steps = list(report.Reasoning_Steps)
    actions, triggered = [], []

    # 1. Citations must be verbatim guideline text the model was actually shown.
    citations = [c for c in report.FDA_Safety_Warnings_Cited if is_grounded(c, chunks)]
    removed = len(report.FDA_Safety_Warnings_Cited) - len(citations)
    if removed:
        actions.append(f"Removed {removed} citation(s) not found verbatim in the retrieved guideline excerpts.")

    def flag(rule_id: str, message: str, quote: str | None) -> None:
        nonlocal eligible
        triggered.append(rule_id)
        if eligible:
            actions.append(f"Set Eligible_For_Vaccine to false ({rule_id}).")
        eligible = False
        steps.append(f"[Safety check] {message}")
        if quote and not any(_normalize(quote) == _normalize(c) for c in citations):
            citations.append(quote)

    # 2. FR-4.4: the recommended dose must lie inside the guideline's dose range. The bound is read from
    #    the retrieved text; the full guideline is read as well, so a dose clause that retrieval happened
    #    to miss is still enforced (it can only add restrictions, never remove them).
    retrieved_limits = extract_dose_limits(chunks)
    limits = extract_dose_limits(chunks + [full_guideline_text()])
    max_mcg = limits.max_safe_mcg
    dose_mcg = dose_mg * MCG_PER_MG
    source = "retrieved text" if retrieved_limits.max_safe_mcg is not None else "full guideline (it was not in the retrieved excerpts)"
    if max_mcg is None:
        flag("no_dose_guidance_found",
             "Neither the retrieved excerpts nor the guideline state a dose range, so no dose can be confirmed; "
             "defaulting to not eligible.", None)
    elif dose_mcg > max_mcg:
        flag("recommended_dose_above_guideline_max",
             f"Recommended dose {dose_mcg:g} mcg ({dose_mg:g} mg) exceeds the guideline maximum of "
             f"{max_mcg:g} mcg stated in the {source}; rejected.",
             limits.review_sentence or (limits.primary_range.sentence if limits.primary_range else None))
    elif eligible and limits.primary_range and dose_mcg < limits.primary_range.low_mcg:
        low, high = limits.primary_range.low_mcg, limits.primary_range.high_mcg
        flag("recommended_dose_below_guideline_min",
             f"Recommended dose {dose_mcg:g} mcg is below the {low:g}-{high:g} mcg reference range, so it cannot "
             "be confirmed as a valid dose; rejected.", limits.primary_range.sentence)

    # 3. Clause-based secondary-review and deferral rules.
    for rule in RULES:
        if rule.applies(patient, limits):
            quote = find_sentence(chunks, rule.quote_pattern) or find_sentence([full_guideline_text()], rule.quote_pattern)
            flag(rule.rule_id, f"Guideline {rule.clause}: {rule.explanation(patient, limits)}", quote)

    # 4. An eligible decision must cite, verbatim, the guideline sentence that states the dose range.
    #    (Any other grounded sentence, e.g. about storage, does not support a dose recommendation.)
    if eligible and not any(_RANGE.search(c) for c in citations):
        flag("no_dose_range_citation",
             "Eligibility must be supported by a verbatim citation of the guideline's dose range, and none was "
             "given; defaulting to not eligible.", None)

    # 5. A not-eligible report carries no dose.
    if not eligible and dose_mg != 0.0:
        actions.append(f"Recommended_Dose_mg reset from {dose_mg:g} to 0.0 because the patient is not eligible for an automated recommendation.")
        steps.append("Recommended dose set to 0.0 mg: the decision is deferred to secondary clinician review.")
        dose_mg = 0.0

    # 6. The disclaimer is always the required notice.
    if report.Disclaimer != REQUIRED_DISCLAIMER:
        actions.append("Disclaimer replaced with the required safety notice.")

    final = ClinicalSafetyReport(
        Eligible_For_Vaccine=eligible,
        Recommended_Dose_mg=round(float(dose_mg), 4),
        Reasoning_Steps=steps,
        FDA_Safety_Warnings_Cited=citations,
        Disclaimer=REQUIRED_DISCLAIMER,
    )
    return GuardrailOutcome(final, actions, triggered)
