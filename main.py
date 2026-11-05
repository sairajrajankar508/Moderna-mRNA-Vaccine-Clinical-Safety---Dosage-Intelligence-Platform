"""mRNA Vaccine Clinical Safety & Dosage Intelligence Platform: command-line entry point.

Run the whole pipeline:      python main.py all
Or one module at a time:     python main.py data | train | attention | index | report | dashboard | writeup
Assess your own patient:     python main.py assess --age 58 --biomarker 6.2 --conditions 2 --dose 120 --history Mild

Student engineering project on synthetic data. Not real clinical guidance.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import CLEAN_CSV, ensure_output_dirs  # noqa: E402

warnings.formatwarning = lambda message, category, *args, **kwargs: f"WARNING: {message}\n"

# Windows consoles default to cp1252; LLM text can contain characters it cannot print.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BANNER = "=" * 72


def section(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_data(_args=None):
    import data_pipeline
    import eda

    section("MODULE 1a: data loading, validation and cleaning")
    df = data_pipeline.run()
    eda.run(df)
    return df


def cmd_train(_args=None):
    import data_pipeline
    import risk_model

    section("MODULE 1b: risk classifier training and evaluation")
    df, _ = data_pipeline.clean(data_pipeline.load_raw())
    return risk_model.run(df)


def cmd_attention(args=None):
    import attention

    section("MODULE 2: mRNA self-attention engine")
    return attention.run(getattr(args, "sequence", None) or attention.DEFAULT_SEQUENCE)


def cmd_index(args=None):
    import rag_index

    section("MODULE 3: RAG index over the safety guidelines")
    return rag_index.run(rebuild=getattr(args, "rebuild", False))


def select_sample_patients(df) -> list[dict]:
    """Five dataset patients covering different decision paths, plus the PRD's example patient."""
    profiles = [
        ("typical adult, standard dose", (df.Vaccine_Dose_mcg <= 100) & (df.Pre_Existing_Conditions <= 1)
         & (df.Prior_Reaction_History == "None") & (df.Biomarker_Level <= 6) & (df.Age < 65)),
        ("raised biomarker, 2 conditions, under 65", (df.Vaccine_Dose_mcg.between(55, 100))
         & (df.Pre_Existing_Conditions == 2) & (df.Prior_Reaction_History != "Moderate")
         & (df.Biomarker_Level.between(8, 10)) & (df.Age < 65)),
        ("dose above 100 mcg", (df.Vaccine_Dose_mcg > 100) & (df.Pre_Existing_Conditions <= 1) & (df.Age < 65)),
        ("65+ with 2+ conditions", (df.Age >= 65) & (df.Pre_Existing_Conditions == 2) & (df.Vaccine_Dose_mcg <= 100)),
        ("prior moderate reaction, high biomarker", (df.Prior_Reaction_History == "Moderate") & (df.Biomarker_Level > 10)),
    ]
    chosen = []
    for _label, mask in profiles:
        matches = df[mask & ~df.Patient_ID.isin([c["Patient_ID"] for c in chosen])]
        if not matches.empty:
            chosen.append(matches.iloc[0].to_dict())
    chosen.append({"Patient_ID": "SAMPLE-58Y-120MCG", "Age": 58, "Biomarker_Level": 6.2,
                   "Pre_Existing_Conditions": 2, "Vaccine_Dose_mcg": 120.0, "Prior_Reaction_History": "Mild"})
    return chosen


def _print_assessment(out: dict) -> None:
    p, risk, report = out["patient"], out["risk"], out["result"].report
    decision = "ELIGIBLE" if report.Eligible_For_Vaccine else "NOT ELIGIBLE (clinician review)"
    print(f"\n{p.patient_id}: age {p.age}, biomarker {p.biomarker_level:g} mg/L, {p.pre_existing_conditions} conditions, "
          f"{p.vaccine_dose_mcg:g} mcg, prior reaction {p.prior_reaction_history}")
    print(f"  Risk: {risk.risk_label} ({risk.risk_probability:.2f})  ->  {decision}, "
          f"dose {report.Recommended_Dose_mg:g} mg ({report.recommended_dose_mcg:g} mcg)")
    rules = out["result"].rules_triggered
    if rules:
        print(f"  Safety rules triggered: {', '.join(rules)}")
    if out["paths"][0]:
        print(f"  Saved {out['paths'][0].name} + {out['paths'][1].name}  "
              f"[{out['result'].provider}, {out['result'].attempts} attempt(s)]")


def _report_dependencies(provider: str | None):
    from llm_providers import get_llm_client
    from rag_index import build_index
    from risk_model import load_model

    return load_model(), build_index(), get_llm_client(provider)


def cmd_report(args=None):
    import pandas as pd

    from report_generator import ReportGenerationError, assess_patient

    section("MODULE 4: structured clinical safety reports")
    model, index, client = _report_dependencies(getattr(args, "provider", None))
    df = pd.read_csv(CLEAN_CSV, keep_default_na=False, na_values=[""])
    outputs = []
    for record in select_sample_patients(df):
        try:
            out = assess_patient(record, model, index, client)
        except ReportGenerationError as err:
            print(f"\n{record['Patient_ID']}: report failed - {err}")
            continue
        except Exception as err:  # provider errors: invalid key, model name, rate limit, network
            print(f"\n{record['Patient_ID']}: LLM call failed - {type(err).__name__}: {err}")
            continue
        _print_assessment(out)
        outputs.append(out)
    print(f"\n{len(outputs)} report pairs (.json + .md) saved to outputs/reports/")
    return outputs


def cmd_assess(args):
    from report_generator import assess_patient

    history = args.history.capitalize()
    if history not in ("None", "Mild", "Moderate"):
        sys.exit("--history must be None, Mild or Moderate")
    if not 0 < args.dose <= 1000:
        sys.exit("--dose must be a positive number of micrograms")
    section("ASSESS A SINGLE PATIENT")
    model, index, client = _report_dependencies(args.provider)
    record = {"Patient_ID": args.patient_id, "Age": args.age, "Biomarker_Level": args.biomarker,
              "Pre_Existing_Conditions": args.conditions, "Vaccine_Dose_mcg": args.dose,
              "Prior_Reaction_History": history}
    try:
        out = assess_patient(record, model, index, client)
    except Exception as err:  # validation failure after the retry, or a provider/API error
        sys.exit(f"Report could not be generated - {type(err).__name__}: {err}")
    _print_assessment(out)
    print("\n" + out["result"].report.model_dump_json(indent=2))
    return out


def cmd_dashboard(_args=None):
    import dashboard

    section("MODULE 5: dashboard")
    return dashboard.run()


def cmd_writeup(_args=None):
    import writeup

    section("PORTFOLIO REPORT")
    return writeup.run()


def cmd_all(args):
    ensure_output_dirs()
    cmd_data()
    cmd_train()
    cmd_attention(args)
    cmd_index(args)
    cmd_report(args)
    cmd_dashboard()
    cmd_writeup()
    section("DONE")
    print("All outputs are in the outputs/ folder. Start with outputs/figures/dashboard.png "
          "and outputs/reports/; the write-up is docs/PROJECT_REPORT.md.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mRNA Vaccine Clinical Safety & Dosage Intelligence Platform")
    sub = parser.add_subparsers(dest="command")

    p_all = sub.add_parser("all", help="run every module in order")
    p_all.add_argument("--rebuild", action="store_true", help="re-embed the guideline index")
    p_all.add_argument("--provider", choices=["auto", "anthropic", "openai", "offline"])
    p_all.add_argument("--sequence", help="mRNA sequence for Module 2 (A/C/G/U, 5-50 bases)")

    sub.add_parser("data", help="Module 1a: validate, clean, EDA figures")
    sub.add_parser("train", help="Module 1b: train and compare classifiers")
    p_att = sub.add_parser("attention", help="Module 2: self-attention heatmap")
    p_att.add_argument("sequence", nargs="?", help="mRNA sequence (A/C/G/U, 5-50 bases)")
    p_idx = sub.add_parser("index", help="Module 3: build/reuse the RAG index and run test queries")
    p_idx.add_argument("--rebuild", action="store_true", help="re-embed even if the index is up to date")
    p_rep = sub.add_parser("report", help="Module 4: reports for the sample patients")
    p_rep.add_argument("--provider", choices=["auto", "anthropic", "openai", "offline"])
    sub.add_parser("dashboard", help="Module 5: 3-panel dashboard")
    sub.add_parser("writeup", help="regenerate docs/PROJECT_REPORT.md from the outputs")

    p_as = sub.add_parser("assess", help="assess one patient you describe")
    p_as.add_argument("--age", type=int, required=True)
    p_as.add_argument("--biomarker", type=float, required=True, help="inflammatory biomarker, mg/L")
    p_as.add_argument("--conditions", type=int, required=True, help="number of pre-existing conditions")
    p_as.add_argument("--dose", type=float, required=True, help="candidate dose in micrograms")
    p_as.add_argument("--history", default="None", help="prior reaction: None, Mild or Moderate")
    p_as.add_argument("--patient-id", default="CUSTOM")
    p_as.add_argument("--provider", choices=["auto", "anthropic", "openai", "offline"])
    return parser


COMMANDS = {"all": cmd_all, "data": cmd_data, "train": cmd_train, "attention": cmd_attention, "index": cmd_index,
            "report": cmd_report, "assess": cmd_assess, "dashboard": cmd_dashboard, "writeup": cmd_writeup}


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        COMMANDS[args.command](args)
    except FileNotFoundError as err:
        print(f"\nERROR: {err}")
        return 1
    except ValueError as err:  # includes invalid sequences and data validation errors
        print(f"\nERROR: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
