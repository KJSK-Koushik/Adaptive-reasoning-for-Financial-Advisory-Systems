# Datasets

Every question in this project must be **automatically gradeable** — the RL reward is
literally `was the answer correct?`, so free-form advice data is unusable. Each source
below yields either a numeric answer (graded within 1% relative tolerance) or a
categorical label (graded by normalised match).

## Tier 1 — automatic download (no account)

| Dataset | Rows kept | Role | Link |
|---|---|---|---|
| **FinQA** | 8,208 of 8,281 | Core numeric reasoning over financial reports. Medium/Hard tier. | [repo](https://github.com/czyssrs/FinQA) |
| **TAT-QA** | 11,162 usable → 4,000 sampled | Table + text hybrid QA. Medium tier. | [next-tat/TAT-QA](https://huggingface.co/datasets/next-tat/TAT-QA) · [repo](https://github.com/NExTplusplus/TAT-QA) |
| **Financial PhraseBank** | 3,452 @ 75% agreement | Sentiment. **Easy tier** — doubles as a sanity check: the policy must learn to stop almost immediately here. | [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) |
| **Statlog German Credit** | 1,000 | Credit risk, templated into questions. | [UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |

Licences: FinQA MIT, TAT-QA MIT, PhraseBank CC BY-NC-SA 3.0 (non-commercial — fine for
academic work, worth citing in the report), German Credit CC BY 4.0.

> **We do not use `datasets.load_dataset` for FinQA or PhraseBank.** As of `datasets`
> 3.x, repository loading scripts are refused outright, and both of those Hub repos are
> script-only (`finqa.py`, `financial_phrasebank.py` with no parquet). The downloader
> fetches the upstream archives directly instead — FinQA from the authors' GitHub,
> PhraseBank from the v1.0 zip on the Hub. This also pins us to exact upstream files,
> which is better for reproducibility.

### Adapter decisions that affect your results

**FinQA — grade against `exe_ans`, not `answer`.** The human-written `answer` string is
inconsistently scaled and sometimes empty. In `train.json[0]`, `answer='380'` while
`exe_ans=3.8` and the program is `divide(3.8, 1)`; the context says "$3.8 million".
`exe_ans` is what the official execution-accuracy metric uses. 124 of 6,251 training
rows have a non-float `exe_ans` (mostly `yes`/`no`) and become categorical questions
rather than being discarded.

**Percentages are stored as fractions.** `answer='53%'` has `exe_ans=0.53232`. The
grader accepts a 100× discrepancy, which covers this. It is a deliberate leniency — it
slightly inflates accuracy but avoids mass false negatives from a scale convention.
Worth a line in the limitations section.

**FinQA context uses the annotated gold evidence rows** (`qa.gold_inds`) rather than the
full filing, controlled by `data.use_gold_evidence`. This is the standard FinQA "gold
evidence" setting. It isolates reasoning from retrieval — which is precisely our
research question — and cuts prompt length by roughly an order of magnitude, which
directly reduces Phase 3 GPU time.

**TAT-QA keeps only numerically-gradeable questions.** Of 13,251 train questions:
`arithmetic` (5,553) and `count` (305) are kept; `span` (5,737) is kept only when the
single span parses as a number; `multi-span` (1,656) is dropped. Grading free text
against a reasoning model's phrasing is unreliable, and unreliable grading poisons the
RL reward. TAT-QA's separate `scale` field is folded into the question as an explicit
instruction ("Give your answer in thousands.") so a single numeric comparison suffices.

**German Credit omits protected attributes.** Attribute 9 encodes personal status *and
sex*; attribute 20 is foreign-worker status. Templating those into a credit-approval
question would bake sex- and nationality-based lending into the training data. Both are
excluded by default via `data.exclude_protected_attributes`; the marital-status portion
is kept as a legitimate financial signal. The flag makes the choice visible and
reversible rather than silent.

## Tier 2 — manual download

### ConvFinQA — 3,458 conversations (~12,600 turns), Hard tier
```bash
git clone https://github.com/czyssrs/ConvFinQA.git
```
Unzip `data.zip` into `data/raw/convfinqa/`.

The release ships each split twice: `train.json`/`dev.json` are conversation-level
(3,037 / 421 items, with `annotation.dialogue_break` + `annotation.exe_ans_list`), and
`train_turn.json`/`dev_turn.json` are the same content re-exploded one item per turn.
We load the conversation-level files only — loading both would duplicate everything.
`test_private.json` is excluded: it ships dialogue with no answers.

> **Never grade ConvFinQA against `qa.exe_ans`.** The `qa` block is inherited from the
> original FinQA example the conversation was built from and is **identical across
> every turn**. In `dev_turn.json`, the first conversation's five turns have true
> answers 60.94, 25.14, 35.8, 25.14 and 1.42403 — while `qa.exe_ans` reads 1.42403 for
> all five. Using it assigns the final answer to every turn: labels that look entirely
> plausible and are wrong. `annotation.exe_ans_list` holds the per-turn answers.
>
> Verified in the built data: of 724 conversations with more than one turn present,
> 709 have more than one distinct gold answer. A near-zero figure there would mean the
> bug had returned, and `tests/test_convfinqa.py` guards it.

### PaySim — mobile money fraud, 6.3M rows (we sample 5,000)
https://www.kaggle.com/datasets/ealaxi/paysim1 — free Kaggle account, ~470 MB zipped.
Place `PS_20174392719_1491204439457_log.csv` in `data/raw/paysim/`.

> **Why PaySim and not the more famous ULB `creditcardfraud` set?**
> ULB's features are `V1`–`V28`, anonymised PCA components with no human meaning. You
> cannot write a natural-language reasoning question about "V17 = −2.3". PaySim has
> readable columns (`type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`)
> that template cleanly into real questions.

## Tier 3 — generated in Phase 1

**Synthetic finance math** (~6,000 items). A Python generator produces both the question
and the exact answer for: SIP maturity, CAGR, EMI, compound interest, NPV/IRR, tax slabs,
portfolio allocation, break-even. Grading is perfect and difficulty is directly
controllable, which gives us clean coverage across all three tiers.

## Templating tabular data into questions

PaySim and German Credit are CSV rows, not questions. Phase 1 converts them:

> A **TRANSFER** of ₹181.00 was requested from an account holding exactly ₹181.00,
> leaving a zero balance, to a destination account with no prior balance and no
> resulting balance change. Is this transaction fraudulent?
> *Gold: yes*

Multiple templates per source, with randomised phrasing, so the model cannot pattern-match
on sentence structure alone.

## Unified schema

All sources converge on `data/processed/unified.parquet`:

```
id, source, domain, question, context, gold_answer, answer_type,
answer_options, difficulty, difficulty_score, split
```

`domain` ∈ {investment, report_qa, fraud, risk, sentiment}.
`difficulty` and `difficulty_score` are filled in by Phase 2, not Phase 1.

## Actual totals

Measured from the Phase 1 run (`artifacts/results/phase1_summary.json`):

| Source | Built | Domain |
|---|---|---|
| FinQA | 8,208 | report_qa |
| Synthetic | 6,000 | investment |
| PaySim | 5,000 | fraud |
| TAT-QA | 4,000 (of 11,162 usable) | report_qa |
| PhraseBank | 3,452 | sentiment |
| ConvFinQA | 3,000 (of ~12,600 turns) | report_qa |
| German Credit | 1,000 | risk |
| **Total** | **30,660** | |

Domain mix: report_qa 15,208 · investment 6,000 · fraud 5,000 · sentiment 3,452 · risk 1,000.
Answer types: numeric 21,043 · categorical 9,617.
Splits: train 21,459 · val 4,599 · test 4,602.
92 duplicates removed. Context length: p50 237 chars, p90 1,409, p99 2,597.

Only ~6,000 of these get reasoning traces in Phase 3 (stratified sample). The rest stay
as held-out evaluation data.

## Deduplication and splitting

Deduplication is on the **question + context** pair, not the id: FinQA and ConvFinQA
draw on the same filings, and TAT-QA repeats phrasings across tables. 79 duplicates
were removed. Splits are stratified by `(source, domain, answer_type)` — not by
difficulty, which does not exist until Phase 2.
