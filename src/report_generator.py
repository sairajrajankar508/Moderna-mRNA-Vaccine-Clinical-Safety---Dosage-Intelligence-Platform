"""Module 4d: combine risk + retrieved guidelines into a validated safety report.

Covers FR-4.2 (grounded prompt), FR-4.3 (validate, retry exactly once),
FR-4.5 (disclaimer always present) and FR-4.6 (.json + .md per report).
FR-4.1 lives in schemas.py and FR-4.4 in guardrails.py.

Flow for one patient
--------------------
  patient record -> Module 1 risk -> Module 3 retrieval (several targeted
  questions) -> prompt -> LLM -> JSON parse + Pydantic validation (retry
  once) -> guardrails.enforce -> save .json and .md
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from config import GENERATION_LOG_PATH, REPORTS_DIR, ensure_output_dirs
from guardrails import ELEVATED_BIOMARKER_MG_L, PatientProfile, enforce
from schemas import REQUIRED_DISCLAIMER, ClinicalSafetyReport

MAX_ATTEMPTS = 2  # first try + exactly one retry

SYSTEM_PROMPT = f"""You are a clinical safety reporting assistant for a student engineering project.

Use ONLY two sources of facts: the Module 1 risk prediction and the guideline excerpts in the user message.
The patient record is the input being assessed, not a source of clinical rules.

Required safety constraints:
1) Never recommend a dose that contradicts the retrieved guideline text.
2) Never invent a clinical metric, statistic, threshold or guideline that is not present in the retrieved excerpts.
3) If the excerpts do not contain enough information to answer confidently, set Eligible_For_Vaccine to false
   (the conservative default) and explain why in Reasoning_Steps instead of guessing.
4) Every report must include the Disclaimer field with exactly this text: "{REQUIRED_DISCLAIMER}"

Output rules:
- Recommended_Dose_mg is in MILLIGRAMS. The guideline uses micrograms. Convert: 1 mg = 1000 micrograms (divide micrograms by 1000).
- If the patient is not eligible, set Recommended_Dose_mg to 0.0.
- FDA_Safety_Warnings_Cited must contain sentences copied word for word from the excerpts. Do not paraphrase them.
- Reasoning_Steps is an ordered list of short sentences that a clinician can follow.
- Reply with one JSON object only: no markdown fences, no text before or after it."""


class ReportGenerationError(RuntimeError):
    """Raised when the LLM response still fails validation after the single retry."""

    def __init__(self, message: str, attempts: int, last_error: Exception, raw_responses: list[str]):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
        self.raw_responses = raw_responses


@dataclass
class GenerationResult:
    report: ClinicalSafetyReport  # final report after guardrails
    draft: ClinicalSafetyReport  # validated LLM output before guardrails
    attempts: int
    latency_s: float
    provider: str
    guardrail_actions: list[str]
    rules_triggered: list[str]
    system_prompt: str
    user_prompt: str
    validation_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval for one patient
# ---------------------------------------------------------------------------

def retrieval_questions(patient: PatientProfile) -> list[str]:
    """Targeted questions so every relevant guideline clause is retrieved for this profile."""
    return [
        f"What is the recommended dose range for a {patient.age}-year-old patient?",
        f"Does a dose of {patient.vaccine_dose_mcg:g} micrograms require secondary clinician review?",
        f"What precautions apply to a patient with {patient.pre_existing_conditions} pre-existing conditions "
        f"and a {patient.prior_reaction_history.lower()} prior reaction history?",
        "How should dosing be adjusted for an elevated inflammatory biomarker level?",
    ]


def retrieve_for_patient(index, patient: PatientProfile, k: int = 3) -> dict:
    """Run the targeted questions, merge unique chunks (best similarity first), total the latency."""
    best: dict[str, float] = {}
    latencies = []
    for question in retrieval_questions(patient):
        result = index.query(question, k)
        latencies.append(result.latency_s)
        for chunk, sim in zip(result.chunks, result.similarities):
            best[chunk] = max(sim, best.get(chunk, -1.0))
    chunks = sorted(best, key=best.get, reverse=True)
    return {"chunks": chunks, "latencies": latencies, "questions": retrieval_questions(patient)}


# ---------------------------------------------------------------------------
# FR-4.2  Prompt
# ---------------------------------------------------------------------------

def format_instructions() -> str:
    schema = ClinicalSafetyReport.model_json_schema()
    return "Return JSON matching this JSON Schema exactly (no extra fields):\n" + json.dumps(schema, indent=2)


def build_user_prompt(patient: PatientProfile, risk, chunks: list[str]) -> str:
    excerpts = "\n\n".join(f"[Excerpt {i}]\n{chunk}" for i, chunk in enumerate(chunks, 1))
    return (
        "PATIENT RECORD (input to assess)\n"
        f"- Patient ID: {patient.patient_id}\n"
        f"- Age: {patient.age} years\n"
        f"- Inflammatory biomarker level: {patient.biomarker_level:g} mg/L\n"
        f"- Pre-existing conditions: {patient.pre_existing_conditions}\n"
        f"- Candidate dose: {patient.vaccine_dose_mcg:g} micrograms\n"
        f"- Prior reaction history: {patient.prior_reaction_history}\n\n"
        "MODULE 1 RISK PREDICTION\n"
        f"- Predicted risk label: {risk.risk_label}\n"
        f"- Probability of a high-risk reaction: {risk.risk_probability:.2f}\n\n"
        "MODULE 3 RETRIEVED GUIDELINE EXCERPTS (the only allowed source of clinical facts)\n"
        f"{excerpts}\n\n"
        f"{format_instructions()}\n\n"
        "Generate the structured clinical safety report now."
    )


# ---------------------------------------------------------------------------
# FR-4.3  Parse, validate, retry once
# ---------------------------------------------------------------------------

def parse_response(raw: str) -> ClinicalSafetyReport:
    """Extract the JSON object from the reply and validate it against the schema."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    return ClinicalSafetyReport.model_validate_json(text)


def generate_report(patient: PatientProfile, risk, chunks: list[str], client, log_path: Path | None = GENERATION_LOG_PATH) -> GenerationResult:
    """Call the LLM, validate (retry exactly once on failure), then apply guardrails."""
    user_prompt = build_user_prompt(patient, risk, chunks)
    context = {"patient": patient, "risk": risk, "chunks": chunks}
    raw_responses, errors = [], []
    draft = None
    start = time.perf_counter()

    prompt = user_prompt
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = client.complete(SYSTEM_PROMPT, prompt, context=context)
        raw_responses.append(raw)
        try:
            draft = parse_response(raw)
            break
        except (ValidationError, ValueError) as err:  # ValueError covers malformed JSON
            errors.append(f"attempt {attempt}: {err}")
            prompt = (f"{user_prompt}\n\nYour previous reply failed schema validation with this error:\n{err}\n"
                      "Reply again with one corrected JSON object only.")
    latency = time.perf_counter() - start

    if draft is None:
        raise ReportGenerationError(
            f"LLM response failed validation after {MAX_ATTEMPTS} attempts (1 retry). Last error: {errors[-1]}",
            attempts=MAX_ATTEMPTS, last_error=errors[-1], raw_responses=raw_responses,
        )

    outcome = enforce(draft, patient, chunks)
    result = GenerationResult(
        report=outcome.report, draft=draft, attempts=len(raw_responses), latency_s=latency,
        provider=client.name, guardrail_actions=outcome.actions, rules_triggered=outcome.rules_triggered,
        system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, validation_errors=errors,
    )
    if log_path is not None:
        _log_generation(log_path, patient, result)
    return result


def _log_generation(path: Path, patient: PatientProfile, result: GenerationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "patient_id", "provider", "attempts", "latency_s", "eligible", "rules_triggered"])
        writer.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), patient.patient_id, result.provider,
                         result.attempts, f"{result.latency_s:.6f}", result.report.Eligible_For_Vaccine,
                         ";".join(result.rules_triggered)])


# ---------------------------------------------------------------------------
# FR-4.6  Persist
# ---------------------------------------------------------------------------

def save_report(result: GenerationResult, patient: PatientProfile, risk, retrieval: dict, out_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    """Write <patient_id>_report.json (the schema object) and a readable <patient_id>_report.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", patient.patient_id)
    json_path = out_dir / f"{safe_id}_report.json"
    md_path = out_dir / f"{safe_id}_report.md"
    json_path.write_text(result.report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result, patient, risk, retrieval), encoding="utf-8")
    return json_path, md_path


def render_markdown(result: GenerationResult, patient: PatientProfile, risk, retrieval: dict) -> str:
    r = result.report
    decision = "ELIGIBLE" if r.Eligible_For_Vaccine else "NOT ELIGIBLE / REFER FOR CLINICIAN REVIEW"
    lines = [
        f"# Clinical Safety Report: {patient.patient_id}",
        "",
        f"> **Disclaimer:** {r.Disclaimer}",
        "",
        f"**Decision:** {decision}  ",
        f"**Recommended dose:** {r.Recommended_Dose_mg:g} mg ({r.recommended_dose_mcg:g} mcg)  ",
        f"**Generated by:** `{result.provider}` in {result.attempts} attempt(s), {result.latency_s:.3f}s  ",
        f"**Created:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Patient record",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Age | {patient.age} years |",
        f"| Inflammatory biomarker | {patient.biomarker_level:g} mg/L |",
        f"| Pre-existing conditions | {patient.pre_existing_conditions} |",
        f"| Candidate dose | {patient.vaccine_dose_mcg:g} mcg |",
        f"| Prior reaction history | {patient.prior_reaction_history} |",
        "",
        "## Module 1: predicted reaction risk",
        "",
        f"**{risk.risk_label}** risk, probability {risk.risk_probability:.2f} (model: {risk.model_name})",
        "",
        "## Reasoning",
        "",
        *[f"{i}. {step}" for i, step in enumerate(r.Reasoning_Steps, 1)],
        "",
        "## Guideline sentences cited",
        "",
        *([f"> {c}\n" for c in r.FDA_Safety_Warnings_Cited] or ["_None._", ""]),
        "## Safety checks applied after generation",
        "",
        *([f"- {a}" for a in result.guardrail_actions] or ["- None: the generated draft already met every check."]),
        "",
        f"Rules triggered: {', '.join(result.rules_triggered) or 'none'}. "
        f"Biomarker threshold used: {ELEVATED_BIOMARKER_MG_L:g} mg/L (project assumption).",
        "",
        "## Module 3: retrieval",
        "",
        f"{len(retrieval['chunks'])} unique guideline chunk(s) retrieved from {len(retrieval['questions'])} queries "
        f"(total retrieval time {sum(retrieval['latencies']):.4f}s).",
        "",
        *[f"- {q}" for q in retrieval["questions"]],
        "",
        "---",
        "",
        f"*{r.Disclaimer}*",
        "",
    ]
    return "\n".join(lines)


def assess_patient(record: dict, model, index, client, save: bool = True) -> dict:
    """End-to-end assessment of one raw patient record (used by main.py)."""
    from risk_model import predict_risk

    patient = PatientProfile.from_record(record)
    risk = predict_risk(model, patient.to_record())
    retrieval = retrieve_for_patient(index, patient)
    result = generate_report(patient, risk, retrieval["chunks"], client)
    paths = save_report(result, patient, risk, retrieval) if save else (None, None)
    return {"patient": patient, "risk": risk, "retrieval": retrieval, "result": result, "paths": paths}


if __name__ == "__main__":
    # PRD sample patient: 58-year-old, 2 pre-existing conditions, candidate dose 120 mcg.
    from llm_providers import get_llm_client
    from rag_index import build_index
    from risk_model import load_model

    ensure_output_dirs()
    sample = {"Patient_ID": "SAMPLE-58Y-120MCG", "Age": 58, "Biomarker_Level": 6.2,
              "Pre_Existing_Conditions": 2, "Vaccine_Dose_mcg": 120.0, "Prior_Reaction_History": "Mild"}
    out = assess_patient(sample, load_model(), build_index(), get_llm_client())
    print(out["result"].report.model_dump_json(indent=2))
    print(f"Saved {out['paths'][0].name} and {out['paths'][1].name}")
