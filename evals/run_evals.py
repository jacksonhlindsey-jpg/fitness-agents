"""Evaluation harness: runs every case k times and reports pass@1 and pass^k side by side.

Each run uses a fresh SQLite database with the same profile, sends the case's message
through the same path as the public link (Orchestrator -> Tracker agent -> real tools,
including live USDA lookups), then grades what was actually saved, not just the reply.

    uv run python evals/run_evals.py                  # all cases, 3 runs each
    uv run python evals/run_evals.py --runs 5 --case banana --case run
    uv run python evals/run_evals.py --provider ollama   # free local model, no key

Needs a model: GEMINI_API_KEY (free key from https://aistudio.google.com/apikey) or Ollama.
Writes evals/results/<timestamp>.json (every run, with its trace) and a .md summary.

pass@1 = the share of all runs that pass (how often one try works).
pass^k = the share of cases that pass on all k runs (how often it works every time).
With more than k runs per case, pass^k uses the unbiased estimate C(c, k) / C(n, k).
"""

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3, help="runs per case (default 3)")
    parser.add_argument("--k", type=int, default=None, help="k for pass^k (default: --runs)")
    parser.add_argument("--case", action="append", help="only this case id (repeatable)")
    parser.add_argument("--provider", choices=["gemini", "ollama", "anthropic"], help="override FITNESS_PROVIDER")
    return parser.parse_args()


ARGS = parse_args()
if ARGS.provider:
    os.environ["FITNESS_PROVIDER"] = ARGS.provider
os.environ["DATABASE_URL"] = ""          # never touch the hosted database
os.environ["FITNESS_MULTI_USER"] = ""    # no usage caps inside the harness

from fitness_agents import config, db, llm  # noqa: E402
from fitness_agents.graph import build_graph  # noqa: E402

USER = "eval"


# --- one run ------------------------------------------------------------------

def run_case(case: dict, profile: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        config.DB_PATH = Path(tmp) / "eval.db"
        db.save_profile(USER, json.loads(json.dumps(profile)))
        seeds = case.get("seed_foods", [])
        for food in seeds:
            db.log_food(USER, food)

        start = time.monotonic()
        try:
            result = build_graph().invoke(
                {"user_id": USER, "user_input": case["input"]},
                {"configurable": {"thread_id": str(uuid.uuid4())}},
            )
            error = None
        except llm.LLMError as e:
            result, error = {}, f"LLMError: {e}"
        seconds = round(time.monotonic() - start, 1)

        state = observe(result, seeds)
        failures = [error] if error else grade(case["expect"], state)
        return {"case": case["id"], "passed": not failures, "failures": failures, "seconds": seconds,
                "state": state, "response": result.get("response"), "trace": result.get("trace")}


def observe(result: dict, seeds: list[dict]) -> dict:
    """What the run actually did, read back from the database."""
    entries = db.day_entries(USER)
    foods = entries["foods"]
    history = db.history(USER, days=1)
    trace = result.get("trace") or []
    return {
        "intent": result.get("intent"),
        "awaiting_approval": "__interrupt__" in result,
        "seed_foods": foods[:len(seeds)],
        "foods": foods[len(seeds):],
        "calories": round(sum(f["calories"] for f in foods[len(seeds):]), 1),
        "water_oz": db.day_totals(USER)["water_oz"],
        "activities": entries["activities"],
        "body_weight_lb": history["weights"][-1]["value"] if history["weights"] else None,
        "bench_max": (db.get_profile(USER) or {}).get("max_lifts", {}).get("bench"),
        "outcome": trace[-1]["type"] if trace else None,
        "steps": sum(1 for s in trace if s["step"] != "answer"),
    }


# --- grading --------------------------------------------------------------------

def within(value, bounds) -> bool:
    return value is not None and bounds[0] <= value <= bounds[1]


def grade(expect: dict, s: dict) -> list[str]:
    """Every expectation the run missed, in words. Empty = pass."""
    fails = []
    if "intent" in expect and s["intent"] != expect["intent"]:
        fails.append(f"routed to {s['intent']!r}, expected {expect['intent']!r}")
    if s["awaiting_approval"]:
        fails.append("started a plan (approval card) instead of logging")
    if "foods" in expect and not within(len(s["foods"]), expect["foods"]):
        fails.append(f"logged {len(s['foods'])} foods, expected {expect['foods']}: {[f['food'] for f in s['foods']]}")
    if "calories" in expect and not within(s["calories"], expect["calories"]):
        fails.append(f"logged {s['calories']} kcal, expected {expect['calories']}")
    if expect.get("all_usda") and any(f["source"] != "usda" for f in s["foods"]):
        fails.append("a food used the model's estimate instead of USDA data")
    if "water_oz" in expect and not within(s["water_oz"], expect["water_oz"]):
        fails.append(f"water {s['water_oz']} oz, expected {expect['water_oz']}")
    if "body_weight_lb" in expect and not within(s["body_weight_lb"], expect["body_weight_lb"]):
        fails.append(f"body weight {s['body_weight_lb']}, expected {expect['body_weight_lb']}")
    if "bench_max" in expect and not within(s["bench_max"], expect["bench_max"]):
        fails.append(f"bench max {s['bench_max']}, expected {expect['bench_max']}")
    for want in expect.get("activities", []):
        if not any(activity_matches(a, want) for a in s["activities"]):
            fails.append(f"no activity matching {want}; logged {s['activities']}")
    if expect.get("asks_question") and s["outcome"] != "ask_user":
        fails.append(f"did not ask a clarifying question (ended with {s['outcome']!r})")
    if expect.get("seed_intact") and len(s["seed_foods"]) != 1:
        fails.append("the seeded log entry is gone")
    return fails


def activity_matches(a: dict, want: dict) -> bool:
    for key, target in want.items():
        value = a.get(key)
        if isinstance(target, list):
            if not within(value, target):
                return False
        elif (value or "").lower() != target:
            return False
    return True


# --- report ---------------------------------------------------------------------

def pass_hat_k(n: int, c: int, k: int) -> float:
    """Chance that k runs drawn from these n all pass (unbiased; = all-pass when n == k)."""
    return comb(c, k) / comb(n, k) if n >= k else float("nan")


def main():
    data = json.loads((HERE / "cases.json").read_text())
    cases = [c for c in data["cases"] if not ARGS.case or c["id"] in ARGS.case]
    k = ARGS.k or ARGS.runs
    print(f"{len(cases)} cases x {ARGS.runs} runs, model {config.FAST_MODEL} via {config.PROVIDER}\n")

    runs = []
    for case in cases:
        for i in range(ARGS.runs):
            r = run_case(case, data["profile"])
            # Response-text checks need the reply, which observe() doesn't see.
            for bad in case["expect"].get("response_excludes", []):
                if bad.lower() in (r["response"] or "").lower():
                    r["failures"].append(f"reply quotes the instructions ({bad!r})")
            if case["expect"].get("nonempty_reply") and not (r["response"] or "").strip():
                r["failures"].append("empty reply")
            r["passed"] = not r["failures"]
            r["run"] = i + 1
            runs.append(r)
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"{mark}  {case['id']:<20} run {i + 1}  {r['seconds']:>5}s  {'; '.join(r['failures'])}")

    rows, total_pass, all_pass = [], 0, []
    for case in cases:
        mine = [r for r in runs if r["case"] == case["id"]]
        c = sum(r["passed"] for r in mine)
        total_pass += c
        ph = pass_hat_k(len(mine), c, k)
        all_pass.append(ph)
        reasons = sorted({f for r in mine for f in r["failures"]})
        rows.append((case, c, len(mine), c / len(mine), ph, reasons))

    pass1 = total_pass / len(runs)
    passk = sum(all_pass) / len(all_pass)
    seconds = sorted(r["seconds"] for r in runs)
    median = seconds[len(seconds) // 2]

    stamp = time.strftime("%Y-%m-%d-%H%M")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    meta = {"when": stamp, "provider": config.PROVIDER, "fast_model": config.FAST_MODEL, "model": config.MODEL,
            "runs_per_case": ARGS.runs, "k": k, "pass@1": round(pass1, 3), f"pass^{k}": round(passk, 3),
            "median_seconds": median}
    (out / f"{stamp}.json").write_text(json.dumps({"meta": meta, "runs": runs}, indent=2))

    lines = [
        f"# Eval results {stamp}",
        "",
        f"{len(cases)} cases x {ARGS.runs} runs = {len(runs)} runs · {config.FAST_MODEL} via {config.PROVIDER} · "
        f"median {median} s per run",
        "",
        f"**pass@1 = {pass1:.0%}** ({total_pass}/{len(runs)} runs) · **pass^{k} = {passk:.0%}** "
        f"({sum(1 for p in all_pass if p == 1)}/{len(cases)} cases passed every run)",
        "",
        f"| Case | Category | Passed | pass@1 | pass^{k} | Why it failed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case, c, n, p1, ph, reasons in rows:
        lines.append(f"| {case['id']} | {case['category']} | {c}/{n} | {p1:.0%} | {ph:.0%} | "
                     f"{'<br>'.join(reasons).replace('|', '/') or ''} |")
    (out / f"{stamp}.md").write_text("\n".join(lines) + "\n")

    print(f"\npass@1 = {pass1:.0%}   pass^{k} = {passk:.0%}   median {median} s/run")
    print(f"Wrote {out / f'{stamp}.md'} and .json")


if __name__ == "__main__":
    sys.exit(main())
