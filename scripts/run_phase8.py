"""Phase 8 - launch the advisory service or the dashboard.

    python scripts/run_phase8.py --dashboard        # Streamlit UI (the demo)
    python scripts/run_phase8.py --api              # FastAPI service
    python scripts/run_phase8.py --check            # verify both without serving

The dashboard is what gets shown in a review. The API is the same controller behind
HTTP, for anything that needs to call it programmatically.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_reasoning import paths  # noqa: E402
from adaptive_reasoning.config import load_config  # noqa: E402

DASHBOARD = ROOT / "src" / "adaptive_reasoning" / "app" / "dashboard.py"


def check(experiment: str) -> int:
    """Load everything the service needs and exercise one request end to end."""
    from fastapi.testclient import TestClient

    from adaptive_reasoning.app.api import create_app

    missing = [p for p in (paths.UNIFIED_DATASET, paths.TRACE_DATASET,
                           paths.RL_TRANSITIONS, paths.DQN_POLICY) if not p.exists()]
    if missing:
        for path in missing:
            print(f"  MISSING {path}")
        print("\n  Run the earlier phases first.")
        return 1

    client = TestClient(create_app(experiment=experiment))

    health = client.get("/health").json()
    print(f"  health           {health['status']}, policies {health['policies']}, "
          f"{health['traced_questions']:,} traced questions")

    questions = client.get("/questions", params={"limit": 5}).json()
    print(f"  /questions       returned {len(questions)}")

    ok = True
    for policy in health["policies"]:
        response = client.post("/ask", params={"question_id": questions[0]["id"],
                                               "policy": policy})
        if response.status_code != 200:
            print(f"  /ask [{policy}]      FAILED {response.status_code} {response.text[:120]}")
            ok = False
            continue
        d = response.json()
        print(f"  /ask [{policy:3}]      stopped at step {d['stopped_at_step']}, "
              f"{d['tokens_used']} of {d['tokens_if_unrestricted']} tokens "
              f"({d['token_reduction_pct']}% saved), correct={d['correct']}")

    stats = client.get("/stats").json()
    print(f"  /stats           {stats['dataset_questions']:,} questions, "
          f"{len(stats['evaluation'])} policies evaluated")

    bad = client.post("/ask", params={"question_id": "does-not-exist"})
    if bad.status_code != 404:
        print(f"  error handling   FAILED - expected 404, got {bad.status_code}")
        ok = False
    else:
        print("  error handling   unknown question returns 404")

    print("\n  ALL CHECKS PASSED" if ok else "\n  CHECKS FAILED")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 8 - advisory application")
    parser.add_argument("--dashboard", action="store_true", help="launch Streamlit")
    parser.add_argument("--api", action="store_true", help="launch FastAPI")
    parser.add_argument("--check", action="store_true", help="verify without serving")
    parser.add_argument("--experiment", default="reported")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    cfg = load_config(args.experiment)

    if args.check or not (args.dashboard or args.api):
        print(f"checking the {cfg.app.title} service\n")
        return check(args.experiment)

    if args.dashboard:
        port = args.port or 8501
        print(f"starting the dashboard on http://localhost:{port}")
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(DASHBOARD),
                                "--server.port", str(port)])

    port = args.port or cfg.serve.port
    print(f"starting the API on http://{cfg.serve.host}:{port} (docs at /docs)")
    return subprocess.call([
        sys.executable, "-m", "uvicorn",
        "adaptive_reasoning.app.api:create_app", "--factory",
        "--host", cfg.serve.host, "--port", str(port),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
