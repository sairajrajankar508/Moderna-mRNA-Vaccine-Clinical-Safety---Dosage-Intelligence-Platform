# mRNA Vaccine Clinical Safety & Dosage Intelligence Platform (mRNA-VIP)

An AI pipeline that assesses vaccine-reaction risk for a patient and produces a structured, guideline-grounded dosage safety report. It combines classical machine learning, a from-scratch self-attention module, retrieval-augmented generation over a safety guideline, and schema-validated LLM output.

> **Disclaimer:** individual student engineering project built on synthetic data. It is not a clinical system and must never be used for medical decisions. It is a case study referencing Moderna's public mRNA work and is not affiliated with or endorsed by Moderna.

## Pipeline

```
patient_records.csv --> [1] clean + classify ----> risk label + probability --+
mRNA sequence ------> [2] self-attention ------> attention heatmap           |
safety guidelines --> [3] chunk + embed + ChromaDB --> top-k excerpts --------+--> [4] LLM -> JSON -> guardrails --> report .json + .md
                                                                                              |
                                        [5] dashboard <-- saved outputs of modules 1-4 -------+
```

| Module | What it does | Code |
|---|---|---|
| 1 | Validates and cleans the data, trains Logistic Regression and Random Forest, keeps the better one by test F1 | `src/data_pipeline.py`, `src/risk_model.py`, `src/eda.py` |
| 2 | Scaled dot-product self-attention on A/C/G/U sequences (NumPy, cross-checked with PyTorch), multi-head bonus | `src/attention.py` |
| 3 | Sentence-aware chunking (300-500 tokens, ~50 overlap), embeddings, persistent ChromaDB, top-k search with latency logging | `src/rag_index.py` |
| 4 | Grounded prompt, LLM call via LangChain, strict Pydantic validation with one retry, deterministic safety guardrails, .json + .md reports | `src/schemas.py`, `src/report_generator.py`, `src/guardrails.py`, `src/llm_providers.py` |
| 5 | 3-panel dashboard (ROC, attention, latency) built from saved outputs | `src/dashboard.py` |
| Report | Builds `docs/PROJECT_REPORT.md` (PRD Section 10) from the outputs | `src/writeup.py` |

## Setup (Windows)

```bat
cd mrna-vip
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and paste an `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (free or trial credits are enough). Without a key everything still runs: Module 4 then uses an offline, rule-based generator and the reports say so.

macOS/Linux: `source venv/bin/activate` and `cp .env.example .env`.

The first run downloads the embedding model `BAAI/bge-small-en-v1.5` (about 130 MB) from Hugging Face, so it needs internet access once.

## Run

```bat
python main.py all --rebuild
```

That runs every module in order (about 1-2 minutes, plus the model download on the first run) and writes everything to `outputs/` and `docs/PROJECT_REPORT.md`. Individual steps:

| Command | Output |
|---|---|
| `python main.py data` | `data/patient_records_clean.csv`, EDA figures |
| `python main.py train` | `outputs/risk_model.joblib`, `outputs/metrics/*`, confusion matrices |
| `python main.py attention [SEQUENCE]` | `outputs/figures/attention_heatmap.png` |
| `python main.py index [--rebuild]` | `outputs/chroma_store/`, `outputs/rag_test_queries.md` |
| `python main.py report [--provider anthropic\|openai\|offline]` | 6 report pairs in `outputs/reports/` |
| `python main.py dashboard` | `outputs/figures/dashboard.png` |
| `python main.py writeup` | `docs/PROJECT_REPORT.md` |

Assess any patient you describe:

```bat
python main.py assess --age 58 --biomarker 6.2 --conditions 2 --dose 120 --history Mild
```

Each module also runs on its own, as in the PRD: `python src/risk_model.py`, `python src/attention.py`, `python src/rag_index.py`, `python src/report_generator.py`.

The notebooks `notebooks/01_data_eda.ipynb` and `notebooks/02_attention_math.ipynb` already contain their outputs. To re-run them, open them in VS Code and select the `venv` kernel.

## Tests

```bat
python -m unittest discover -s tests -v
```

93 tests, about 20 seconds, no API key or internet needed (the retrieval tests use the offline embedder; ChromaDB is used when installed). The PyTorch cross-check is skipped if torch is missing.

## Functional requirements: where each is met and tested

| FR | Requirement | Code | Test (in `tests/`) |
|---|---|---|---|
| 1.1 | 7 columns validated, named error | `data_pipeline.validate_columns` | `TestFR11ColumnValidation` |
| 1.2 | Median / mode imputation | `data_pipeline.clean` | `TestFR12Imputation` |
| 1.3 | Standardize 3 columns (train set) | `risk_model.build_preprocessor` | `test_scaled_columns_have_zero_mean_unit_std_on_training_set` |
| 1.4 | Stratified 80/20, fixed seed | `risk_model.split_data` | `test_split_is_80_20_stratified_and_reproducible` |
| 1.5 | LR + RF on identical data | `risk_model.train_models` | `test_both_models_trained_on_same_data` |
| 1.6 | Reports + confusion matrices saved | `risk_model.save_artifacts` | `test_artifacts_saved_and_model_reloads_identically` |
| 1.7 | Best F1 serialized, reloads identically | `risk_model.select_best`, `save_artifacts` | same test |
| 2.1 | Sequence to (seq_len, embed_dim) | `attention.encode_sequence` | `TestFR21Encoding` |
| 2.2 | Reject non-ACGU | `attention.validate_sequence` | `TestFR22InvalidInput` |
| 2.3 | Q/K/V, (n, n) weights | `attention.self_attention` | `test_15_bases_gives_15x15` |
| 2.4 | Rows sum to 1 | `attention.softmax` | `test_rows_sum_to_one_for_many_sequences` |
| 2.5 | Labelled heatmap PNG | `attention.attention_figure`, `plot_attention` | `TestFR25Heatmap` |
| 3.1 | 300-500 token chunks, ~50 overlap | `rag_index.chunk_text` | `TestFR31Chunking` |
| 3.2 | One embedding per chunk | `rag_index.build_index` | `TestFR32Embeddings` |
| 3.3 | Persistent ChromaDB, survives restart | `ChromaStore`, `load_index` | `test_fresh_process_can_query_without_reembedding` |
| 3.4 | Top-k ordered by similarity | `GuidelineIndex.query` | `TestFR34TopK` |
| 3.5 | Latency recorded per query | `GuidelineIndex.query`, `_log` | `TestFR35Latency` |
| 4.1 | Strict 5-field Pydantic schema | `schemas.ClinicalSafetyReport` | `TestFR41Schema` |
| 4.2 | Prompt grounded only in Modules 1 + 3 | `report_generator.build_user_prompt`, `SYSTEM_PROMPT` | `TestFR42GroundedPrompt` |
| 4.3 | Validate, retry exactly once | `report_generator.generate_report` | `TestFR43RetryOnce` |
| 4.4 | Dose above guideline bound -> not eligible | `guardrails.enforce`, `extract_dose_limits` | `TestFR44DoseBound` |
| 4.5 | Disclaimer always present | `guardrails.enforce` | `test_disclaimer_present_in_five_reports_even_if_llm_changes_it` |
| 4.6 | .json + .md per report | `report_generator.save_report` | `test_json_and_md_pair_written_for_each_report` |
| 5.1 | Exactly 3 titled panels | `dashboard.create_dashboard_figure` | `TestFR51ThreePanels` |
| 5.2 | Panels driven by saved outputs | `dashboard.load_inputs` | `TestFR52DrivenByData` |
| 5.3 | At least 150 DPI | `dashboard.DPI` (200) | `TestFR53Resolution` |

## Design decisions that differ from the PRD templates

Each is explained in the code comments too.

1. **pandas reads `None` as missing.** In `Prior_Reaction_History`, `None` means "no prior reaction" (291 patients). With pandas' default reader those become blanks and mode imputation relabels them `Mild`. `load_raw` treats only empty cells as missing.
2. **Scaler fitted on the training split only.** Preprocessing sits inside a scikit-learn `Pipeline`, so test data never influences the scaling, and the saved model scores raw patient values directly.
3. **Ordinal encoding** (None=0 < Mild=1 < Moderate=2) instead of `LabelEncoder`, which sorts alphabetically and would put None above Moderate.
4. **Small grid search** for both models so the comparison is between tuned models.
5. **Fixed, Xavier-scaled attention weights.** The template's `* 0.1` makes softmax almost uniform (flat heatmap), and drawing new weights inside every call makes results change between calls. `encode_sequence` checks the alphabet only, because the PRD's own example `'ACGU'` is shorter than its 5-50 length rule; the length rule is enforced at the pipeline entry `analyze_sequence`.
6. **Custom sentence-aware chunker.** LangChain's splitter has no minimum chunk size, so it cannot guarantee FR-3.1's 300-token floor.
7. **`BAAI/bge-small-en-v1.5` embeddings** instead of `all-MiniLM-L6-v2`, which truncates at 256 word-pieces and would ignore most of every 300-500 token chunk.
8. **Retry implemented explicitly.** The JSON is parsed with Pydantic in our own loop (rather than `PydanticOutputParser` in a chain) so the attempts can be counted and FR-4.3's exactly-one-retry can be tested with a mock LLM. LangChain is used for the chat models and the embeddings.
9. **Units.** The PRD fixes the field name `Recommended_Dose_mg`, but the data and guideline are in micrograms. Reports store milligrams (50 mcg = 0.05 mg) and the guardrails catch a microgram value placed in the mg field.
10. **Guardrails after the LLM.** Deterministic checks keep the recommended dose inside the guideline range (FR-4.4; the bound is read from the retrieved text, with the full guideline as a safety net if retrieval missed the clause), apply the secondary-review clauses (65+ with 2+ conditions, 3+ conditions, prior moderate reaction, dose above 100 mcg, elevated biomarker), remove citations that are not verbatim guideline text, require an eligible decision to cite the dose-range sentence, and can only make a decision more conservative.
11. **One assumption.** The guideline says "meaningfully above the normal reference range" for the biomarker but gives no number. The threshold is set to 10 mg/L (`ELEVATED_BIOMARKER_MG_L`), and reports state it is an assumption.
12. **Offline fallbacks.** If ChromaDB or an API key is unavailable, the pipeline still runs (NumPy store, rule-based generator), prints a warning, and labels every output with what produced it. `.env.example` sets `EMBEDDING_BACKEND=huggingface`, so a failed embedding-model download stops with an error instead of silently switching to the offline hashing embedder; set it to `auto` to allow that fallback. Submit results produced with the full stack.

## Project structure

```
mrna-vip/
  main.py                    command-line entry point
  data/                      patient_records.csv, mrna_safety_guidelines.txt (+ generated clean CSV)
  src/                       one file per module (see table above) + config.py
  notebooks/                 01_data_eda.ipynb, 02_attention_math.ipynb
  tests/                     unittest suite, one file per module + integration adapters
  outputs/                   figures/, metrics/, reports/, logs/, chroma_store/, risk_model.joblib
  docs/PROJECT_REPORT.md     portfolio report, regenerated by `python main.py writeup`
  requirements.txt, .env.example, .gitignore
```

## Submission checklist (PRD 10.1)

- [ ] `pip freeze > requirements-lock.txt` after a successful run, to pin exact versions
- [ ] Run `python main.py all --rebuild` with an API key so `outputs/` and the report reflect the real LLM and embedding model
- [ ] Rewrite the explanation paragraphs in `docs/PROJECT_REPORT.md` in your own words (graded under Conceptual Understanding). Delete its first line (the AUTO-GENERATED marker) when you start editing; after that, reruns write fresh numbers to `docs/PROJECT_REPORT.generated.md` instead of overwriting your text
- [ ] Commit in stages (e.g. one commit per module) rather than one final commit
- [ ] Check that `.env` is not committed
