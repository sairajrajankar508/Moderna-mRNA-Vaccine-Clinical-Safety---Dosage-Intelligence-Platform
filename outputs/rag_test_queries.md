# RAG retrieval test queries

Embedding model: `BAAI/bge-small-en-v1.5`  |  Store: `chromadb`  |  Chunks: 4 (357, 362, 355, 314 tokens)  |  k = 3

A **hit** means the sentence that answers the question is inside the top-ranked chunk.

## Standard questions: 4/8 hits

Worded like the guideline itself.

| # | Question | Answer phrase | Top chunk covers | Hit@1 | Answer rank | Similarity | Top chunk starts with | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | What are the storage requirements below -20C? | "-25C and -15C" | Section 1, Section 2 | **NO** | 2 | 0.649 | Once a vial is punctured, its contents must be used within the manufacturer-specified in-use window,... | 0.1904 |
| 2 | How long can a thawed vial stay in the refrigerator before it must be discarded? | "up to 30 days" | Section 1, Section 2 | **NO** | 2 | 0.776 | Once a vial is punctured, its contents must be used within the manufacturer-specified in-use window,... | 0.0619 |
| 3 | What dose range applies to adults aged 12 to 64? | "Ages 12 to 64" | Section 2, Section 3 | yes | 1 | 0.822 | - Ages 6 months to 4 years: lowest available dose tier; administration requires pediatric-formulated... | 0.0626 |
| 4 | When does a dose need secondary clinician review? | "Doses above 100 micrograms" | Section 2, Section 3 | **NO** | 3 | 0.765 | - Ages 6 months to 4 years: lowest available dose tier; administration requires pediatric-formulated... | 0.0566 |
| 5 | Should dosing be deferred for a patient with an elevated inflammatory biomarker? | "dosing deferred" | Section 2, Section 3 | yes | 1 | 0.788 | - Ages 6 months to 4 years: lowest available dose tier; administration requires pediatric-formulated... | 0.0651 |
| 6 | What are the absolute contraindications for an mRNA product? | "anaphylaxis" | Front matter, Section 1 | **NO** | 3 | 0.741 | Clinical Safety, Storage & Dosing Reference Guide | 0.0612 |
| 7 | How long should patients be monitored after administration? | "at least 15 minutes" | Section 3, Section 4 | yes | 1 | 0.690 | - Known severe allergic reaction (e.g., anaphylaxis) to a prior dose of the same product or to any c... | 0.0702 |
| 8 | Within how many hours must a moderate adverse event be documented? | "within 24 hours" | Section 3, Section 4 | yes | 1 | 0.749 | - Known severe allergic reaction (e.g., anaphylaxis) to a prior dose of the same product or to any c... | 0.0637 |

## Paraphrase stress test: 5/5 hits

Same facts, different vocabulary (e.g. 'put back in the freezer' vs 'refrozen').

| # | Question | Answer phrase | Top chunk covers | Hit@1 | Answer rank | Similarity | Top chunk starts with | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | Can a vial be put back in the freezer after it has thawed? | "never be refrozen" | Section 1, Section 2 | yes | 1 | 0.719 | Once a vial is punctured, its contents must be used within the manufacturer-specified in-use window,... | 0.0692 |
| 2 | What should I do if the liquid in a vial looks cloudy? | "cloudiness" | Section 1, Section 2 | yes | 1 | 0.667 | Once a vial is punctured, its contents must be used within the manufacturer-specified in-use window,... | 0.0623 |
| 3 | What happens to a batch if the cold chain fails during shipping? | "quarantined" | Section 1, Section 2 | yes | 1 | 0.648 | Once a vial is punctured, its contents must be used within the manufacturer-specified in-use window,... | 0.0656 |
| 4 | Is a severe allergy to the first shot a reason to skip the next one? | "anaphylaxis" | Section 3, Section 4 | yes | 1 | 0.764 | - Known severe allergic reaction (e.g., anaphylaxis) to a prior dose of the same product or to any c... | 0.0657 |
| 5 | When must a reaction be escalated to a safety review? | "hospitalization" | Section 3, Section 4 | yes | 1 | 0.794 | - Known severe allergic reaction (e.g., anaphylaxis) to a prior dose of the same product or to any c... | 0.0653 |
