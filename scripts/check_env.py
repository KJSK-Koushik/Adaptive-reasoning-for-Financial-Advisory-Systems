"""Environment and hardware smoke test.

Verifies that the config loads, the package imports, the grader works, and
reports what compute is available - including an honest verdict on whether
Phase 3 trace generation can run on this machine.

    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning import device, grading, paths  # noqa: E402
from adaptive_reasoning.config import load_config  # noqa: E402

CORE_PACKAGES = [
    "numpy", "pandas", "pyarrow", "yaml", "pydantic", "sklearn",
    "torch", "transformers", "gymnasium", "matplotlib",
]
OPTIONAL_PACKAGES = ["datasets", "sentence_transformers", "lightgbm", "fastapi", "streamlit", "pynvml"]

OK, WARN, FAIL = "  [ok]  ", " [warn] ", " [FAIL] "


def _check_packages(names: list[str], required: bool) -> list[str]:
    missing = []
    for name in names:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"{OK}{name:<24} {version}")
        except ImportError:
            missing.append(name)
            print(f"{FAIL if required else WARN}{name:<24} not installed")
    return missing


def main() -> int:
    failures = 0

    print("=" * 68)
    print("  ENVIRONMENT")
    print("=" * 68)
    print(f"{OK}python                   {sys.version.split()[0]}")
    print(f"{OK}project root             {paths.ROOT}")

    print("\n--- required packages ---")
    missing_core = _check_packages(CORE_PACKAGES, required=True)
    failures += len(missing_core)

    print("\n--- optional packages ---")
    _check_packages(OPTIONAL_PACKAGES, required=False)

    print("\n" + "=" * 68)
    print("  CONFIGURATION")
    print("=" * 68)
    try:
        cfg = load_config()
        print(f"{OK}configs/default.yaml     loaded and validated")
        print(f"{OK}seed                     {cfg.project.seed}")
        print(f"{OK}reasoning model          {cfg.llm.model_id}")
        print(f"{OK}DQN state dimension      {cfg.rl.state_dim}")
        print(f"{OK}token budget             {cfg.llm.max_new_tokens}")
        print(f"{OK}questions to trace       {cfg.traces.n_questions}")
    except Exception as exc:  # noqa: BLE001 - surface any config error verbatim
        print(f"{FAIL}config failed to load: {exc}")
        return 1

    print("\n" + "=" * 68)
    print("  DIRECTORIES")
    print("=" * 68)
    for d in (paths.DATA_RAW, paths.DATA_PROCESSED, paths.TRACES, paths.MODELS, paths.RESULTS):
        status = OK if d.exists() else WARN
        note = "" if d.exists() else "  (run scripts/bootstrap.py)"
        print(f"{status}{str(d.relative_to(paths.ROOT)):<24}{note}")

    print("\n" + "=" * 68)
    print("  GRADER SELF-TEST")
    print("=" * 68)
    cases = [
        ("The answer is $1,234.56", "1234.56", "numeric", True),
        ("approximately 12.4%", "12.4", "numeric", True),
        ("0.124", "12.4", "numeric", True),        # fraction vs percent
        ("(1,234)", "-1234", "numeric", True),
        ("2.5 million", "2500000", "numeric", True),
        ("13.0", "12.4", "numeric", False),
        ("This looks fraudulent to me", "yes", "categorical", True),
        ("legitimate", "no", "categorical", True),
        ("high risk", "bad", "categorical", True),
    ]
    for predicted, gold, kind, expected in cases:
        got = grading.is_correct(predicted, gold, kind, cfg.data.numeric_tolerance)
        mark = OK if got == expected else FAIL
        if got != expected:
            failures += 1
        print(f"{mark}{predicted!r:<32} vs {gold!r:<12} -> {got}")

    print("\n" + "=" * 68)
    print("  HARDWARE")
    print("=" * 68)
    hw = device.detect(cfg.project.device)
    print(f"{OK}torch backend            {hw.backend}")
    print(f"{OK}torch version            {hw.torch_version}")
    print(f"{OK}cpu threads              {hw.cpu_threads}")
    if hw.ram_gb:
        print(f"{OK}system RAM               {hw.ram_gb} GB")
    if hw.gpu_name:
        print(f"{OK}GPU                      {hw.gpu_name} ({hw.gpu_memory_gb} GB)")
    for note in hw.notes:
        print(f"{WARN}{note}")

    print("\n--- Phase 3 feasibility ---")
    if hw.can_generate_traces_locally:
        print(f"{OK}This machine can generate reasoning traces locally.")
    else:
        print(f"{WARN}No usable CUDA GPU. Phases 0-2 and 4-8 run fine here.")
        print("        Phase 3 (trace generation) should run on Kaggle Notebooks,")
        print("        Google Colab, or a rented GPU. See docs/PHASE3_REMOTE.md.")

    print("\n" + "=" * 68)
    if failures:
        print(f"  {failures} CHECK(S) FAILED")
    else:
        print("  ALL CHECKS PASSED")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
