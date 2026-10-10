"""Plan quality evaluation: run the real initial pipeline on a few JDs and measure the plan.

Run:  python eval/evaluate_plan_quality.py --label before   (needs OPENAI_API_KEY; uses the local Qdrant store)
Results go to eval/plan_quality_results_<label>.json.

Metrics per plan:
- skill_match_rate: share of planned questions whose own skill labels include the skill they were planned for
- max_skill_share: largest share of the plan taken by one planned skill
- quota_over_supply: share of the skill quotas that exceed how many questions the bank has for that skill
- daily_min / daily_max: questions per day
- difficulty_gap: total variation distance between the planned and the target difficulty mix (0 = exact)
- max_hard_per_day, seconds
"""

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp()
os.environ["CHECKPOINT_DB"] = os.path.join(_TMP, "checkpoints.db")
os.environ["LONG_TERM_DB"] = os.path.join(_TMP, "long_term.db")
os.environ.setdefault("QDRANT_URL", "")  # empty -> local vector store in knowledge_base/vectorstores
sys.path.insert(0, str(ROOT))

from orchestration import multi_agent  # noqa: E402
from scripts.Agent3.Planning_Agent import normalize_skill_name  # noqa: E402

CASE_IDS = [
    "eval_001_apple_ml_ds_input_experience",
    "eval_008_microsoft_technical_ds_interview",
    "eval_011_netflix_data_insights_personalization",
    "eval_014_uber_marketplace_scientist",
    "eval_019_meta_product_growth_analytics",
    "eval_028_microsoft_time_series_anomaly",
]


def bank_supply() -> Counter:
    supply = Counter()
    for line in open(ROOT / "data" / "questions_normalized.jsonl"):
        for skill in json.loads(line).get("taxonomy_skills") or []:
            supply[normalize_skill_name(skill)] += 1
    return supply


def plan_metrics(result: dict, supply: Counter) -> dict:
    days = result["days"] or []
    tasks = [t for day in days for t in day]
    n = len(tasks) or 1
    matched = sum(
        normalize_skill_name(t.get("requested_skill", "")) in {normalize_skill_name(s) for s in t.get("taxonomy_skills") or []}
        for t in tasks
    )
    planned_skill = Counter(normalize_skill_name(t.get("requested_skill", "")) for t in tasks)
    quota = {normalize_skill_name(k): int(v) for k, v in (result.get("skill_plan") or {}).items()}
    over = sum(max(0, q - supply[s]) for s, q in quota.items())
    target = result.get("difficulty_distribution") or {}
    target_total = sum(target.values()) or 1
    actual = Counter(t.get("difficulty") for t in tasks)
    gap = 0.5 * sum(abs(actual[d] / n - target.get(d, 0) / target_total) for d in ("easy", "medium", "hard"))
    return {
        "questions": len(tasks),
        "days": len(days),
        "skill_match_rate": round(matched / n, 3),
        "max_skill_share": round(max(planned_skill.values()) / n, 3) if tasks else 0,
        "quota_over_supply": round(over / (sum(quota.values()) or 1), 3),
        "daily_min": min((len(d) for d in days), default=0),
        "daily_max": max((len(d) for d in days), default=0),
        "difficulty_gap": round(gap, 3),
        "difficulty_actual": dict(actual),
        "difficulty_target": target,
        "max_hard_per_day": max((sum(t.get("difficulty") == "hard" for t in d) for d in days), default=0),
        "quota": quota,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate initial plan quality.")
    parser.add_argument("--label", required=True, help="e.g. before / after")
    parser.add_argument("--only", default="", help="comma-separated case ids")
    args = parser.parse_args()

    cases = {c["case_id"]: c for c in json.loads((ROOT / "eval" / "eval_cases.json").read_text())["cases"]}
    ids = args.only.split(",") if args.only else CASE_IDS
    supply = bank_supply()
    rows = []
    for case_id in ids:
        case = cases[case_id]
        started = time.time()
        result = multi_agent(case["jd"], case["user_desc"], case["days_left"], user_id=f"quality-{case_id}")
        metrics = plan_metrics(result, supply)
        metrics.update({"case_id": case_id, "seconds": round(time.time() - started, 1)})
        rows.append(metrics)
        print(f"{case_id:52s} q={metrics['questions']:3d} match={metrics['skill_match_rate']:.2f} "
              f"maxshare={metrics['max_skill_share']:.2f} over={metrics['quota_over_supply']:.2f} "
              f"daily={metrics['daily_min']}-{metrics['daily_max']} gap={metrics['difficulty_gap']:.2f} "
              f"hard/day={metrics['max_hard_per_day']} {metrics['seconds']}s")

    keys = ["skill_match_rate", "max_skill_share", "quota_over_supply", "difficulty_gap", "seconds"]
    summary = {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in keys}
    summary["daily_range_avg"] = round(sum(r["daily_max"] - r["daily_min"] for r in rows) / len(rows), 2)
    summary["max_hard_per_day_max"] = max(r["max_hard_per_day"] for r in rows)
    print("\nmean:", json.dumps(summary))
    out = ROOT / "eval" / f"plan_quality_results_{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "summary": summary, "cases": rows}, indent=2, ensure_ascii=False))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
