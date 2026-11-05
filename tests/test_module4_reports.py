"""Module 4 tests: FR-4.1 to FR-4.6, using mock LLM clients (no API key needed)."""

import json
import re
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401

from pydantic import ValidationError

import guardrails as gr
import report_generator as rg
from config import GUIDELINES_TXT
from llm_providers import OfflineGroundedClient
from rag_index import chunk_text
from risk_model import RiskResult
from schemas import REQUIRED_DISCLAIMER, ClinicalSafetyReport

ALL_CHUNKS = [c.text for c in chunk_text(GUIDELINES_TXT.read_text(encoding="utf-8"))]
RANGE_SENTENCE = gr.extract_dose_limits(ALL_CHUNKS).primary_range.sentence
LOW_RISK = RiskResult("Low", 0.2, "test")
HIGH_RISK = RiskResult("High", 0.8, "test")


def patient(**overrides):
    base = dict(Patient_ID="T1", Age=40, Biomarker_Level=4.0, Pre_Existing_Conditions=0,
                Vaccine_Dose_mcg=80.0, Prior_Reaction_History="None")
    base.update(overrides)
    return gr.PatientProfile.from_record(base)


def valid_json(**overrides):
    body = {"Eligible_For_Vaccine": True, "Recommended_Dose_mg": 0.08,
            "Reasoning_Steps": ["Dose is within the retrieved range."],
            "FDA_Safety_Warnings_Cited": [RANGE_SENTENCE], "Disclaimer": REQUIRED_DISCLAIMER}
    body.update(overrides)
    return json.dumps(body)


class ScriptedClient:
    """Mock LLM that returns pre-written replies and counts calls."""

    name = "mock"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.prompts = []

    def complete(self, system, user, context=None):
        self.prompts.append(user)
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


class TestFR41Schema(unittest.TestCase):
    def test_valid_report(self):
        report = ClinicalSafetyReport.model_validate_json(valid_json())
        self.assertTrue(report.Eligible_For_Vaccine)
        self.assertEqual(report.recommended_dose_mcg, 80.0)

    def test_exact_field_set(self):
        self.assertEqual(set(ClinicalSafetyReport.model_fields), {
            "Eligible_For_Vaccine", "Recommended_Dose_mg", "Reasoning_Steps", "FDA_Safety_Warnings_Cited", "Disclaimer"})

    def test_missing_field_raises(self):
        for field in ("Eligible_For_Vaccine", "Recommended_Dose_mg", "Reasoning_Steps", "FDA_Safety_Warnings_Cited", "Disclaimer"):
            body = json.loads(valid_json())
            del body[field]
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ClinicalSafetyReport.model_validate_json(json.dumps(body))

    def test_mistyped_field_raises(self):
        for bad in ({"Eligible_For_Vaccine": "true"}, {"Eligible_For_Vaccine": "yes"},
                    {"Recommended_Dose_mg": "0.08"}, {"Recommended_Dose_mg": -0.1},
                    {"Reasoning_Steps": "one step"}, {"FDA_Safety_Warnings_Cited": [1, 2]},
                    {"Disclaimer": ""}, {"Reasoning_Steps": []}):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                ClinicalSafetyReport.model_validate_json(valid_json(**bad))

    def test_extra_field_raises(self):
        body = json.loads(valid_json())
        body["Confidence"] = 0.9
        with self.assertRaises(ValidationError):
            ClinicalSafetyReport.model_validate_json(json.dumps(body))


class TestFR42GroundedPrompt(unittest.TestCase):
    def test_prompt_contains_only_module1_and_module3_facts(self):
        storage_chunk = next(c for c in ALL_CHUNKS if "Ultra-Low Temperature" in c)
        p = patient()
        prompt = rg.build_user_prompt(p, LOW_RISK, [storage_chunk])
        self.assertIn(storage_chunk, prompt)
        self.assertIn("Predicted risk label: Low", prompt)
        self.assertIn("0.20", prompt)
        # Guideline facts from chunks that were NOT retrieved must not leak into the prompt.
        for absent in ("Doses above 100 micrograms", "Three or more relevant pre-existing conditions",
                       "monitored for at least 15 minutes", "within 24 hours"):
            if absent not in storage_chunk:
                self.assertNotIn(absent, prompt)
                self.assertNotIn(absent, rg.SYSTEM_PROMPT)

    def test_system_prompt_has_required_safety_constraints(self):
        sp = rg.SYSTEM_PROMPT
        self.assertIn("Never recommend a dose that contradicts the retrieved guideline text", sp)
        self.assertIn("Never invent a clinical metric", sp)
        self.assertIn("conservative default", sp)
        self.assertIn(REQUIRED_DISCLAIMER, sp)

    def test_system_prompt_states_no_guideline_numbers(self):
        for number in ("25", "50", "100", "65", "15", "24"):
            self.assertIsNone(re.search(rf"\b{number}\b", rg.SYSTEM_PROMPT), number)


class TestFR43RetryOnce(unittest.TestCase):
    def test_valid_first_time_needs_one_call(self):
        client = ScriptedClient(valid_json())
        result = rg.generate_report(patient(), LOW_RISK, ALL_CHUNKS, client, log_path=None)
        self.assertEqual(client.calls, 1)
        self.assertEqual(result.attempts, 1)

    def test_malformed_then_valid_retries_exactly_once(self):
        client = ScriptedClient("this is not JSON at all", valid_json())
        result = rg.generate_report(patient(), LOW_RISK, ALL_CHUNKS, client, log_path=None)
        self.assertEqual(client.calls, 2)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(result.validation_errors), 1)
        self.assertIn("failed schema validation", client.prompts[1])

    def test_schema_violation_then_valid(self):
        client = ScriptedClient(valid_json(Eligible_For_Vaccine="yes"), valid_json())
        self.assertEqual(rg.generate_report(patient(), LOW_RISK, ALL_CHUNKS, client, log_path=None).attempts, 2)

    def test_always_malformed_raises_after_two_calls(self):
        client = ScriptedClient('{"Eligible_For_Vaccine": true}')
        with self.assertRaises(rg.ReportGenerationError) as ctx:
            rg.generate_report(patient(), LOW_RISK, ALL_CHUNKS, client, log_path=None)
        self.assertEqual(client.calls, 2)
        self.assertEqual(ctx.exception.attempts, 2)
        self.assertIn("after 2 attempts", str(ctx.exception))

    def test_json_inside_markdown_fence_is_accepted(self):
        client = ScriptedClient("Here you go:\n```json\n" + valid_json() + "\n```")
        self.assertEqual(rg.generate_report(patient(), LOW_RISK, ALL_CHUNKS, client, log_path=None).attempts, 1)


class TestFR44DoseBound(unittest.TestCase):
    def test_bound_is_parsed_from_retrieved_text(self):
        limits = gr.extract_dose_limits(ALL_CHUNKS)
        self.assertEqual(limits.max_safe_mcg, 100.0)
        self.assertEqual((limits.primary_range.low_mcg, limits.primary_range.high_mcg), (25.0, 100.0))
        self.assertEqual(limits.lower_reactogenicity_range.high_mcg, 50.0)

    def test_candidate_dose_above_threshold_is_not_eligible(self):
        client = ScriptedClient(valid_json(Recommended_Dose_mg=0.13))  # LLM wrongly approves 130 mcg
        result = rg.generate_report(patient(Vaccine_Dose_mcg=130.0), LOW_RISK, ALL_CHUNKS, client, log_path=None)
        self.assertFalse(result.report.Eligible_For_Vaccine)
        self.assertEqual(result.report.Recommended_Dose_mg, 0.0)
        self.assertIn("candidate_dose_above_review_threshold", result.rules_triggered)
        self.assertIn("recommended_dose_above_guideline_max", result.rules_triggered)

    def test_recommended_dose_above_bound_is_rejected(self):
        client = ScriptedClient(valid_json(Recommended_Dose_mg=0.12))  # 120 mcg for an 80 mcg candidate
        result = rg.generate_report(patient(), LOW_RISK, ALL_CHUNKS, client, log_path=None)
        self.assertFalse(result.report.Eligible_For_Vaccine)
        self.assertIn("recommended_dose_above_guideline_max", result.rules_triggered)

    def test_microgram_value_in_mg_field_is_caught(self):
        client = ScriptedClient(valid_json(Recommended_Dose_mg=80.0))  # unit mistake: 80 "mg"
        result = rg.generate_report(patient(), LOW_RISK, ALL_CHUNKS, client, log_path=None)
        self.assertFalse(result.report.Eligible_For_Vaccine)

    def test_dose_at_bound_is_allowed(self):
        client = ScriptedClient(valid_json(Recommended_Dose_mg=0.1))
        result = rg.generate_report(patient(Vaccine_Dose_mcg=100.0), LOW_RISK, ALL_CHUNKS, client, log_path=None)
        self.assertTrue(result.report.Eligible_For_Vaccine)

    def test_no_dose_guidance_retrieved_defaults_to_not_eligible(self):
        storage_only = [c for c in ALL_CHUNKS if "Ultra-Low Temperature" in c and "micrograms" not in c]
        if not storage_only:
            self.skipTest("chunking puts dose text in every chunk")
        result = rg.generate_report(patient(), LOW_RISK, storage_only, OfflineGroundedClient(), log_path=None)
        self.assertFalse(result.report.Eligible_For_Vaccine)


class TestGuardrailRules(unittest.TestCase):
    def run_offline(self, risk=LOW_RISK, **overrides):
        return rg.generate_report(patient(**overrides), risk, ALL_CHUNKS, OfflineGroundedClient(), log_path=None)

    def test_clean_patient_is_eligible_at_candidate_dose(self):
        r = self.run_offline().report
        self.assertTrue(r.Eligible_For_Vaccine)
        self.assertAlmostEqual(r.Recommended_Dose_mg, 0.08)

    def test_high_risk_caps_dose_at_lower_band(self):
        r = self.run_offline(HIGH_RISK).report
        self.assertTrue(r.Eligible_For_Vaccine)
        self.assertAlmostEqual(r.Recommended_Dose_mg, 0.05)

    def test_each_secondary_review_rule(self):
        cases = {
            "age_65_plus_with_2_conditions": dict(Age=70, Pre_Existing_Conditions=2),
            "three_or_more_conditions": dict(Pre_Existing_Conditions=3),
            "prior_moderate_reaction": dict(Prior_Reaction_History="Moderate"),
            "elevated_biomarker": dict(Biomarker_Level=12.5),
            "pediatric_scope": dict(Age=8),
        }
        for rule, overrides in cases.items():
            with self.subTest(rule=rule):
                result = self.run_offline(**overrides)
                self.assertFalse(result.report.Eligible_For_Vaccine)
                self.assertIn(rule, result.rules_triggered)
                self.assertEqual(result.report.Recommended_Dose_mg, 0.0)

    def test_65_plus_with_one_condition_is_fine(self):
        self.assertTrue(self.run_offline(Age=70, Pre_Existing_Conditions=1).report.Eligible_For_Vaccine)

    def test_ungrounded_citation_is_removed(self):
        invented = "Patients over 40 must always receive exactly 60 micrograms."
        client = ScriptedClient(valid_json(FDA_Safety_Warnings_Cited=[RANGE_SENTENCE, invented]))
        result = rg.generate_report(patient(), LOW_RISK, ALL_CHUNKS, client, log_path=None)
        self.assertNotIn(invented, result.report.FDA_Safety_Warnings_Cited)
        self.assertIn(RANGE_SENTENCE, result.report.FDA_Safety_Warnings_Cited)

    def test_eligible_without_any_real_citation_is_rejected(self):
        client = ScriptedClient(valid_json(FDA_Safety_Warnings_Cited=["Made-up guidance sentence that does not exist."]))
        result = rg.generate_report(patient(), LOW_RISK, ALL_CHUNKS, client, log_path=None)
        self.assertFalse(result.report.Eligible_For_Vaccine)
        self.assertIn("no_dose_range_citation", result.rules_triggered)

    def test_unrelated_grounded_citation_does_not_support_eligibility(self):
        storage = "Product must never be refrozen once thawed."
        client = ScriptedClient(valid_json(FDA_Safety_Warnings_Cited=[storage]))
        result = rg.generate_report(patient(), LOW_RISK, ALL_CHUNKS, client, log_path=None)
        self.assertFalse(result.report.Eligible_For_Vaccine)
        self.assertIn("no_dose_range_citation", result.rules_triggered)

    def test_dose_below_range_or_zero_while_eligible_is_rejected(self):
        for dose_mg in (0.0, 0.005):
            with self.subTest(dose_mg=dose_mg):
                client = ScriptedClient(valid_json(Recommended_Dose_mg=dose_mg))
                result = rg.generate_report(patient(), LOW_RISK, ALL_CHUNKS, client, log_path=None)
                self.assertFalse(result.report.Eligible_For_Vaccine)
                self.assertIn("recommended_dose_below_guideline_min", result.rules_triggered)

    def test_dose_rules_enforced_even_if_retrieval_missed_them(self):
        # Only the storage chunk was retrieved, but the LLM approves a 120 mcg candidate at 0.1 mg.
        storage_only = [c for c in ALL_CHUNKS if "Ultra-Low Temperature" in c][:1]
        client = ScriptedClient(valid_json(Recommended_Dose_mg=0.1))
        result = rg.generate_report(patient(Vaccine_Dose_mcg=120.0), LOW_RISK, storage_only, client, log_path=None)
        self.assertFalse(result.report.Eligible_For_Vaccine)
        self.assertIn("candidate_dose_above_review_threshold", result.rules_triggered)

    def test_guardrails_never_upgrade_a_refusal(self):
        client = ScriptedClient(valid_json(Eligible_For_Vaccine=False, Recommended_Dose_mg=0.0))
        result = rg.generate_report(patient(), LOW_RISK, ALL_CHUNKS, client, log_path=None)
        self.assertFalse(result.report.Eligible_For_Vaccine)

    def test_rule_quotes_exist_verbatim_in_guideline(self):
        text = GUIDELINES_TXT.read_text(encoding="utf-8")
        for rule in gr.RULES:
            with self.subTest(rule=rule.rule_id):
                quote = gr.find_sentence([text], rule.quote_pattern)
                self.assertIsNotNone(quote)
                self.assertTrue(gr.is_grounded(quote, [text]))


class TestFR45AndFR46Outputs(unittest.TestCase):
    PATIENTS = [
        dict(Patient_ID="A", Age=30),
        dict(Patient_ID="B", Vaccine_Dose_mcg=140.0),
        dict(Patient_ID="C", Age=75, Pre_Existing_Conditions=3),
        dict(Patient_ID="D", Prior_Reaction_History="Moderate", Biomarker_Level=11.0),
        dict(Patient_ID="E", Biomarker_Level=8.0, Pre_Existing_Conditions=2),
    ]

    def test_disclaimer_present_in_five_reports_even_if_llm_changes_it(self):
        clients = [OfflineGroundedClient(), ScriptedClient(valid_json(Disclaimer="Totally real advice."))]
        for client in clients:
            for overrides in self.PATIENTS:
                result = rg.generate_report(patient(**overrides), HIGH_RISK, ALL_CHUNKS, client, log_path=None)
                with self.subTest(client=client.name, patient=overrides["Patient_ID"]):
                    self.assertEqual(result.report.Disclaimer, REQUIRED_DISCLAIMER)

    def test_json_and_md_pair_written_for_each_report(self):
        retrieval = {"chunks": ALL_CHUNKS, "latencies": [0.01], "questions": ["q"]}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for overrides in self.PATIENTS:
                p = patient(**overrides)
                result = rg.generate_report(p, LOW_RISK, ALL_CHUNKS, OfflineGroundedClient(), log_path=None)
                rg.save_report(result, p, LOW_RISK, retrieval, out_dir=out)
            stems_json = {f.name.replace(".json", "") for f in out.glob("*.json")}
            stems_md = {f.name.replace(".md", "") for f in out.glob("*.md")}
            self.assertEqual(len(stems_json), 5)
            self.assertEqual(stems_json, stems_md)
            for f in out.glob("*.json"):
                ClinicalSafetyReport.model_validate_json(f.read_text(encoding="utf-8"))
            for f in out.glob("*.md"):
                text = f.read_text(encoding="utf-8")
                self.assertIn(REQUIRED_DISCLAIMER, text)
                self.assertIn("## Reasoning", text)


if __name__ == "__main__":
    unittest.main()
