"""Module 4c: LLM clients behind one small interface.

Every client has a ``name`` and ``complete(system, user, context=None) -> str``
that returns the raw text of the model's reply. The report generator does the
parsing and validation itself, so it can count attempts and retry exactly
once (FR-4.3).

Providers (env var LLM_PROVIDER, default "auto"):
  anthropic  ChatAnthropic via LangChain; needs ANTHROPIC_API_KEY
  openai     ChatOpenAI via LangChain;    needs OPENAI_API_KEY
  offline    no API call; a deterministic, guideline-grounded draft writer
  auto       anthropic if its key is set, else openai if its key is set, else offline

The offline client exists so the whole pipeline (and the test-suite) runs
without a key or internet. It is not a language model: it reads the same
retrieved excerpts and risk output the LLM would see, and writes the JSON
with fixed rules. Reports always state which provider produced them.
"""

from __future__ import annotations

import json
import os

import config  # noqa: F401  (loads .env)
from guardrails import PatientProfile, extract_dose_limits, find_sentence
from schemas import MCG_PER_MG, REQUIRED_DISCLAIMER

DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini"}


class LangChainChatClient:
    """A LangChain chat model.

    Anthropic: current Claude models reject or ignore sampling parameters such as
    temperature, and adaptive thinking counts toward max_tokens, so no temperature is
    sent and max_tokens leaves room for thinking before the JSON. OpenAI: temperature 0
    for repeatable output (remove it if LLM_MODEL is set to a reasoning model).
    """

    def __init__(self, provider: str, model: str | None = None):
        model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            self.llm = ChatAnthropic(model=model, max_tokens=16000)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(model=model, temperature=0)
        else:
            raise ValueError(f"Unknown provider '{provider}'")
        self.name = f"{provider}:{model}"

    def complete(self, system: str, user: str, context: dict | None = None) -> str:
        reply = self.llm.invoke([("system", system), ("human", user)])
        content = reply.content
        if isinstance(content, list):  # some providers return a list of content blocks
            content = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
        return content


class OfflineGroundedClient:
    """Deterministic stand-in for the LLM. Uses only the risk output and retrieved chunks.

    Decision logic:
      1. Find the primary dose range in the retrieved text. None found -> not eligible.
      2. Candidate dose above the range -> not eligible (needs clinician review).
      3. Otherwise eligible at the candidate dose; if the risk model says High,
         cap the dose at the top of the lower-reactogenicity band when the
         retrieved text gives one.
    The secondary-review rules (age 65+, conditions, prior reactions,
    biomarker) are applied afterwards by guardrails.enforce, as for any LLM.
    """

    name = "offline:grounded-rules"

    def complete(self, system: str, user: str, context: dict | None = None) -> str:
        if context is None:
            raise ValueError("The offline client needs the structured context.")
        patient: PatientProfile = context["patient"]
        risk = context["risk"]
        chunks: list[str] = context["chunks"]

        steps = [f"The Module 1 risk model predicts {risk.risk_label} reaction risk "
                 f"(probability {risk.risk_probability:.2f})."]
        citations: list[str] = []
        limits = extract_dose_limits(chunks)
        primary = limits.primary_range
        candidate = patient.vaccine_dose_mcg
        eligible, dose_mcg = False, 0.0

        if primary is None:
            steps.append("The retrieved guideline excerpts do not state a dose range, so a safe dose cannot be "
                         "confirmed; conservatively not eligible.")
        else:
            citations.append(primary.sentence)
            if candidate > primary.high_mcg:
                steps.append(f"The candidate dose of {candidate:g} mcg is above the {primary.low_mcg:g}-{primary.high_mcg:g} mcg "
                             "reference range in the retrieved guideline, so it cannot be recommended automatically.")
                if limits.review_sentence:
                    citations.append(limits.review_sentence)
            elif candidate < primary.low_mcg:
                steps.append(f"The candidate dose of {candidate:g} mcg is below the {primary.low_mcg:g}-{primary.high_mcg:g} mcg "
                             "reference range; the intended product should be confirmed, so not eligible.")
            else:
                eligible, dose_mcg = True, candidate
                steps.append(f"The candidate dose of {candidate:g} mcg is within the {primary.low_mcg:g}-{primary.high_mcg:g} mcg "
                             "reference range in the retrieved guideline.")
                lower = limits.lower_reactogenicity_range
                if risk.risk_label == "High" and lower and candidate > lower.high_mcg:
                    dose_mcg = lower.high_mcg
                    steps.append(f"Because predicted risk is High, the dose is lowered to {lower.high_mcg:g} mcg, the top of "
                                 f"the {lower.low_mcg:g}-{lower.high_mcg:g} mcg lower-reactogenicity band in the guideline.")

        monitoring = find_sentence(chunks, r"monitored for at least 15 minutes")
        if eligible and monitoring:
            steps.append("Post-administration monitoring applies as stated in the retrieved guideline.")
            citations.append(monitoring)

        return json.dumps({
            "Eligible_For_Vaccine": eligible,
            "Recommended_Dose_mg": round(dose_mcg / MCG_PER_MG, 4),
            "Reasoning_Steps": steps,
            "FDA_Safety_Warnings_Cited": citations,
            "Disclaimer": REQUIRED_DISCLAIMER,
        })


def get_llm_client(provider: str | None = None):
    """Pick a client from the argument, then LLM_PROVIDER, then available API keys."""
    provider = (provider or os.getenv("LLM_PROVIDER", "auto")).lower()
    if provider == "auto":
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            print("No ANTHROPIC_API_KEY or OPENAI_API_KEY found: using the offline grounded generator.")
            provider = "offline"
    if provider == "offline":
        return OfflineGroundedClient()
    return LangChainChatClient(provider)
