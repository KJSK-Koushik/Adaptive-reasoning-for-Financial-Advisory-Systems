# Running Phases 2b and 3 on Kaggle

Everything up to here runs on the laptop. These two phases need a GPU. Kaggle gives
30 GPU-hours per week free, and the whole run should fit inside about 12 of them.

**Total hands-on time: about 10 minutes of clicking, then it runs by itself.**

---

## What you do vs what is already done

| | |
|---|---|
| **Already written** | the packaging script, the notebook, all the code it calls |
| **You do** | upload one zip, create a notebook, flip two settings, press run |

You do not paste any code. The notebook is imported as a file.

---

## Step 0 — enable Internet on your Kaggle account

Kaggle requires **phone verification** before a notebook may access the internet, and
without it both the pip install and the model download fail.

Go to kaggle.com → your avatar → **Settings** → **Phone Verification**. Do this first;
it is the one step that can block you unexpectedly.

## Step 1 — build the upload package (local)

```bash
python scripts/package_for_kaggle.py
```

Produces `artifacts/kaggle/adaptive-reasoning.zip` (~6.5 MB) containing the source tree,
configs, scripts and `unified.parquet`.

## Step 2 — upload it as a Kaggle Dataset

1. kaggle.com → **Create** (top left) → **New Dataset**
2. Drag in `artifacts/kaggle/adaptive-reasoning.zip`
3. Title: **adaptive-reasoning-fas**
4. Leave it **Private**
5. **Create**

> **Kaggle usually auto-extracts uploaded archives**, so you will often see a `project/`
> folder in the dataset rather than the `.zip`. That is fine — cell 1 detects the
> project by looking for `configs/default.yaml` and handles the archive layout, the
> extracted-with-wrapper layout, and the extracted-without-wrapper layout. If it finds
> nothing it prints exactly what *is* attached so you can see what went wrong.

## Step 3 — create the notebook

1. kaggle.com → **Create** → **New Notebook**
2. **File → Import Notebook**
3. Upload `notebooks/phase3_kaggle.ipynb` from this repo

## Step 4 — configure the session

In the right-hand sidebar:

| Setting | Value |
|---|---|
| **Accelerator** | `GPU T4 x2` (or `GPU P100`) |
| **Internet** | `On` |
| **Input** | **+ Add Input** → **Datasets** → your `adaptive-reasoning-fas` |

Forgetting the accelerator is the most common mistake — cell 3 will stop you with a
clear message rather than silently running on CPU for a week.

## Step 5 — run

Run cells 1 through 4 **one at a time**, and read the output of cell 4.

### Cell 4 is a gate, not a formality

It checks the two things Phase 3 depends on:

1. **Does the model actually reason?** If it answers in three tokens there is nothing
   for a stopping policy to cut short and the run produces nothing learnable. This is
   measured, not assumed — locally, `Qwen2.5-0.5B-Instruct` produced a median of **3
   reasoning tokens** and would have wasted the entire budget.
2. **What is the real throughput?** Every GPU-hour figure in `PHASE3_REMOTE.md` is an
   estimate until this measures it. The pilot recomputes the projection from observed
   tokens/second.

**If the gate fails, stop.** Do not run the later cells. Try a stronger reasoning model
instead:

```bash
!python scripts/run_pilot.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

If it passes, note the printed `projected Phase 3 GPU-hours`. That number tells you
whether to run everything now or trim `traces.n_questions` first.

### Then the long cells

Cells 5–6 are the real work. Two ways to run them:

* **Interactive** — press run and leave the tab open. Simple, but the session ends if
  you close it.
* **Save & Run All (Commit)** — top right. Runs the whole notebook in the background for
  up to 12 hours and emails you. **Use this for the real run.**

Both checkpoint every 100 questions. If a session dies, re-run the same cell and it
resumes from the last shard instead of starting over.

## Step 6 — bring the results home

Cell 7 writes `/kaggle/working/phase3_results.zip`. Download it from the **Output**
panel and unzip it over your local project root:

```
artifacts/traces/traces.parquet          <- the RL training data
artifacts/traces/trace_summary.parquet   <- per-question oracle statistics
artifacts/models/difficulty_clf.joblib
data/processed/difficulty_labels.parquet
artifacts/results/*.json
```

Phases 4–9 then run locally on CPU.

---

## Watch the quota

Kaggle shows GPU hours used in **Settings → Accelerator**. The weekly 30 hours reset on
a rolling basis. If the pilot projects more than ~15 hours, reduce
`traces.n_questions` in `configs/default.yaml` (rebuild the zip and re-upload) rather
than risk running out mid-run.

## Common problems

| Symptom | Cause |
|---|---|
| `Could not find the project under /kaggle/input` | Dataset not attached — use **+ Add Input**. The error lists what *is* attached. |
| pip install hangs or fails | Internet is off, or the account is unverified |
| `No GPU` in cell 3 | Accelerator still set to None |
| Gate fails with "not reasoning" | Instruction model instead of a reasoning model |
| Session dies mid-run | Expected — re-run the cell, it resumes from checkpoints |
