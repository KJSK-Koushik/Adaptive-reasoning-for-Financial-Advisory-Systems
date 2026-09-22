# Setting the project up on another machine

The zip holds everything the project needs except two things that cannot be copied
between machines: the Python virtual environment, and the Hugging Face model cache.
Both are recreated in the steps below. Nothing else has to be downloaded.

What is in the zip

| Included | Why |
|---|---|
| `src/`, `scripts/`, `tests/`, `configs/`, `docs/`, `notebooks/` | the code |
| `data/processed/unified.parquet`, `difficulty_labels.parquet` | Phase 1-2 output; Phases 4-9 and the app read these |
| `artifacts/traces/` | the 4,000 GPU-generated traces every number is computed from |
| `artifacts/models/` | trained difficulty classifier, DQN policy, behaviour-cloning control |
| `artifacts/results/` | every phase's summary JSON; the report and deck are built from these |
| `*.docx`, `*.pptx` | the deliverables |
| `.git/` | full history; `git status` should be clean after unzipping |

Not included: `.venv/` (1.4 GB, machine-specific) and `data/raw/` (822 MB of the
original downloads - only Phase 1 reads it; a separate `data_raw.zip` exists if you
want to rerun Phase 1).

## Steps (Windows, PowerShell)

1. Unzip to a path **without spaces if possible** (spaces work, but every command then
   needs quotes). Open PowerShell in that folder.

2. Python 3.11 must be installed (`py -3.11 --version`). 3.12 also works; 3.10 does not.

3. Create the environment and install:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

   Ten minutes on a normal connection; PyTorch is the bulk of it. A GPU is not needed
   for anything except Phase 3, which runs on Kaggle.

4. Verify:

   ```powershell
   python scripts/check_env.py
   pytest
   ```

   Expect `ALL CHECKS PASSED` and `449 passed`. The first `pytest` downloads a small
   test model (about 100 MB) and the MiniLM sentence encoder into the Hugging Face
   cache, so the first run needs internet and takes a few minutes longer.

5. Confirm the results reproduce from the shipped traces:

   ```powershell
   python scripts/run_phase6.py --experiment reported
   ```

   The table it prints must match `artifacts/results/phase6_summary.json` - our RL
   agent at 40.1%, +8.0 points over the fixed step. (Phases 4 and 5 also reproduce
   bit-for-bit, but Phase 5 retrains the DQN for ten minutes; run them only if you
   want to see it.)

6. The dashboard:

   ```powershell
   streamlit run src/adaptive_reasoning/app/dashboard.py
   ```

## If something differs

* `ModuleNotFoundError: adaptive_reasoning` - step 3's `pip install -e .` was skipped
  or the venv is not activated.
* `pytest` collects fewer than 449 tests - the zip was extracted partially; compare
  `git status`, it should be clean.
* Different numbers from Phase 6 - the traces or models were not copied; check that
  `artifacts/traces/traces.parquet` is 5.4 MB and `artifacts/models/stopping_policy.pt`
  exists.
* Kaggle: the API token lives in `%USERPROFILE%\.kaggle\kaggle.json` on the old
  machine and is deliberately **not** in the zip. Create a new one from the Kaggle
  account page if the new machine needs to push runs.
