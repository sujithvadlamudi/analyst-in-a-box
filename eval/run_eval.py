"""
Run with: python eval/run_eval.py

This is the concrete, measurable artifact for your README and your
interview answer: "I built a small eval set to measure whether the
fact-checker actually catches injected errors." Run this, paste the
pass rate into your README.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # eval scripts run standalone, so they need this too

from agents.fact_checker import fact_checker_node

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def run():
    cases = json.loads(EVAL_SET_PATH.read_text())
    results = []

    for case in cases:
        if "injected_findings" not in case:
            continue  # skip sanity-check-only cases here

        state = {"findings": case["injected_findings"]}
        output = fact_checker_node(state)
        flagged = len(output["flagged_findings"]) > 0

        passed = flagged == case["expect_flagged"]
        results.append((case["id"], passed))
        print(f"[{'PASS' if passed else 'FAIL'}] {case['id']}: {case['notes']}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    print(f"\n{passed_count}/{total} eval cases passed")


if __name__ == "__main__":
    run()
