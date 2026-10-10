"""Tool-calling evaluation for the controller.

The controller uses the real LLM; retrieval and the pipeline modules are replaced by
deterministic fakes, so the score reflects only the controller's decisions (model + prompt).
Each case starts a fresh thread seeded with the same fixture plan (today = day 3 of 7).

Run:  python eval/evaluate_tool_calling.py [--repeats 3] [--only id1,id2] [--cases eval/tool_calling_holdout.json]
Needs OPENAI_API_KEY (from .env). Results go to eval/tool_calling_results.json.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp()
os.environ["CHECKPOINT_DB"] = os.path.join(_TMP, "checkpoints.db")  # never touch the real .state/
os.environ["LONG_TERM_DB"] = os.path.join(_TMP, "long_term.db")
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402

import orchestration  # noqa: E402
from scripts.controller import graph as g  # noqa: E402
from scripts.controller.tools import days_until  # noqa: E402

CASES_PATH = ROOT / "eval" / "tool_calling_cases.json"
OUTPUT_PATH = ROOT / "eval" / "tool_calling_results.json"
WEEKDAYS_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


# ---------- fixture plan ----------
def q(qid, title, skill, diff, status="todo", reason=""):
    return {"id": qid, "title": title, "type": "coding" if qid.startswith("lc:") else "theory",
            "difficulty": diff, "requested_skill": skill, "taxonomy_skills": [skill], "status": status,
            "selection_reason": reason or f"Hybrid match score 0.90; {diff} difficulty; directly matches skill {skill}"}


def build_state(user_id: str, thread_id: str) -> dict:
    days = [
        [q("lc:SQL_101", "Product Sales Analysis 3", "sql_aggregation", "easy", "done"),
         q("th:theory_00062", "What are the decision trees?", "decision_trees", "easy", "done"),
         q("lc:SQL_102", "Immediate Food Delivery", "sql_aggregation", "easy", "done"),
         q("th:theory_00141", "What is clustering? When do we need it?", "clustering", "easy", "done"),
         q("lc:SQL_140", "Department Highest Salary", "sql_join", "medium", "done"),
         q("th:theory_00070", "How do we train decision trees?", "decision_trees", "easy", "done")],
        [q("th:theory_00030", "How do we evaluate classification models?", "classification", "easy", "done"),
         q("th:theory_00031", "Precision-recall trade-off?", "classification", "easy", "done"),
         q("th:theory_00081", "Feature importance in gradient boosting trees", "gradient_boosting", "medium", "done"),
         q("th:theory_00145", "When would you choose K-means and when DBScan?", "clustering", "medium", "done"),
         q("lc:SQL_150", "Consecutive Numbers", "sql_window_function", "medium", "done")],
        [q("lc:SQL_178", "Rank Scores", "sql_window_function", "medium",
           reason="Hybrid match score 1.000; medium difficulty; SQL category; directly matches skill sql_window_function"),
         q("lc:SQL_177", "Nth Highest Salary", "sql_window_function", "medium"),
         q("th:theory_00023", "What is classification? Which models would you use?", "classification", "easy"),
         q("th:theory_00138", "How to select K for K-means?", "clustering", "easy"),
         q("th:theory_00077", "What is gradient boosting trees?", "gradient_boosting", "easy"),
         q("lc:SQL_185", "Department Top Three Salaries", "sql_window_function", "hard")],
        [q("th:theory_00034", "Is accuracy always a good metric?", "classification", "easy"),
         q("lc:SQL_190", "Game Play Analysis IV", "sql_window_function", "medium"),
         q("th:theory_00082", "What are the main parameters in the gradient boosting model?", "gradient_boosting", "medium"),
         q("lc:SQL_160", "Trips and Users", "sql_join", "hard"),
         q("th:theory_00073", "What are the potential problems with many large trees?", "decision_trees", "medium")],
        [q("th:theory_00146", "Do you know how DBScan works?", "clustering", "medium"),
         q("th:theory_00083", "How to set the learning rate?", "gradient_boosting", "medium"),
         q("lc:SQL_120", "Average Waiting Time", "sql_aggregation", "medium"),
         q("lc:SQL_195", "Human Traffic of Stadium", "sql_window_function", "hard")],
        [q("th:theory_00074", "How can we know which features are more important for the decision tree model?", "decision_trees", "medium"),
         q("th:theory_00147", "What other clustering algorithms do you know?", "clustering", "medium"),
         q("th:theory_00035", "What is the ROC curve?", "classification", "medium")],
        [q("th:theory_00150", "What is the curse of dimensionality?", "clustering", "medium"),
         q("lc:SQL_103", "Customer Placing the Largest Number of Orders", "sql_aggregation", "easy")],
    ]
    # Candidate pool: plan questions plus spare ones, except sql_join (no spares: triggers "pool ran short").
    pool = defaultdict(list)
    for day in days:
        for t in day:
            pool[t["requested_skill"]].append(dict(t, status="todo"))
    for skill in ["sql_window_function", "classification", "clustering", "gradient_boosting", "decision_trees", "sql_aggregation"]:
        pool[skill] += [q(f"spare:{skill}:{i}", f"Spare {skill} question {i}", skill, "easy" if i < 2 else "medium") for i in range(4)]
    weights = {"sql_window_function": 0.3, "classification": 0.2, "clustering": 0.15, "gradient_boosting": 0.15,
               "decision_trees": 0.1, "sql_aggregation": 0.05, "sql_join": 0.05}
    return {
        "jd": "Data Scientist, Product Analytics: SQL, experimentation, classification and clustering models.",
        "user_desc": "Statistics master's student; basic SQL; some ML project experience.",
        "days_left": len(days), "user_id": user_id, "thread_id": thread_id,
        "start_date": (date.today() - timedelta(days=2)).isoformat(),
        "weights": weights, "base_weights": dict(weights), "plan": {}, "total_questions": 35,
        "difficulty_distribution": {"easy": 0.4, "medium": 0.45, "hard": 0.15},
        "retrieve_questions": dict(pool), "days": days,
        "summaries": [{"day": i + 1, "summary": ""} for i in range(len(days))],
    }


# ---------- fakes for the modules behind the tools ----------
class FakeRetriever:
    def retrieve(self, query, topk, **kw):
        return [q(f"r:{query}:{i}", f"Retrieved {query} question {i}", query, "easy" if i % 2 else "medium") for i in range(topk)]


class FakeSkill:
    def __init__(self, client): pass
    def run(self, jd_text, user_desc):
        return {"extracted": {}, "mapped": {}, "weights": {"recommender_systems": 0.4, "neural_networks": 0.3, "feature_selection": 0.3}}


class FakeScope:
    def __init__(self, client): pass
    def run(self, skill_weights, days_left, **kw):
        return {"total_questions": 12, "difficulty_distribution": {"easy": 0.4, "medium": 0.4, "hard": 0.2},
                "skill_plan": {s: 4 for s in skill_weights}}


def fake_planning(tasks, days_left, **kw):
    return [tasks[i::days_left] for i in range(days_left)], [{"day": i + 1, "summary": ""} for i in range(days_left)]


orchestration.init_agentic_retriever = lambda llm=None: FakeRetriever()
orchestration.init_retriever = lambda: FakeRetriever()
orchestration.SkillAnalyzerAgent, orchestration.ScopePlannerAgent = FakeSkill, FakeScope
orchestration.normalize_tasks = lambda t: t
orchestration.run_planning_agent = fake_planning


# ---------- scoring ----------
def check_args(case: dict, calls: list) -> list:
    errors = []
    for tool_name, spec in (case.get("expect_args") or {}).items():
        call = next((c for c in calls if c["name"] == tool_name), None)
        if call is None:
            errors.append(f"{tool_name} not called")
            continue
        args = call["args"]
        for arg, rule in spec.items():
            value = effective_days_remaining(args) if arg == "effective_days_remaining" else args.get(arg)
            if "includes" in rule and not set(rule["includes"]) <= set(value or []):
                errors.append(f"{tool_name}.{arg}={value!r}, expected to include {rule['includes']}")
            if "equals" in rule and value != rule["equals"]:
                errors.append(f"{tool_name}.{arg}={value!r}, expected {rule['equals']!r}")
            if rule.get("not_true") and value is True:
                errors.append(f"{tool_name}.{arg} should not be true")
            if rule.get("nonempty") and not value:
                errors.append(f"{tool_name}.{arg} is empty")
    # Plan-shape arguments change the plan a lot; setting one the user did not ask for is an error.
    expected_update = (case.get("expect_args") or {}).get("update_plan", {})
    for call in calls:
        if call["name"] != "update_plan":
            continue
        allowed = set(expected_update) | ({"days_remaining", "interview_day"} if "effective_days_remaining" in expected_update else set())
        for arg in ("days_remaining", "interview_day", "daily_load", "postpone_rest_of_today"):
            if call["args"].get(arg) not in (None, False) and arg not in allowed:
                errors.append(f"update_plan.{arg}={call['args'][arg]!r} was not requested")
    return errors


def effective_days_remaining(args: dict):
    """Study days after today that update_plan will use, whichever way the LLM expressed them."""
    if args.get("days_remaining") is not None:
        return args["days_remaining"]
    if args.get("interview_day"):
        until = days_until(args["interview_day"])
        return None if until is None else until - 1
    return None


def render_message(text: str) -> str:
    return text.replace("{weekday_in_4_days}", WEEKDAYS_ZH[(date.today() + timedelta(days=4)).weekday()])


def run_case(case: dict, rep: int) -> dict:
    thread_id = f"{case['id']}-{rep}"
    config = g._config(thread_id)
    g.app.update_state(config, build_state(f"user-{thread_id}", thread_id), as_node="agent3")
    message = render_message(case["message"])
    started = time.time()
    reply = g.chat(thread_id, message)
    latency = time.time() - started

    messages = g.app.get_state(config).values["messages"]
    start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    calls = [{"name": c["name"], "args": c["args"]} for m in messages[start:] for c in (getattr(m, "tool_calls", None) or [])]
    names = [c["name"] for c in calls]
    tool_ok = names == case["expect_tools"] or names in case.get("also_accept_tools", [])
    arg_errors = check_args(case, calls) if tool_ok else []
    return {"id": case["id"], "category": case["category"], "repeat": rep, "message": message,
            "expected_tools": case["expect_tools"], "actual_tools": names, "calls": calls,
            "tool_ok": tool_ok, "args_ok": tool_ok and not arg_errors, "passed": tool_ok and not arg_errors,
            "arg_errors": arg_errors, "reply": reply, "latency_s": round(latency, 2)}


def summarize(results: list) -> dict:
    def rate(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 3) if rows else None
    with_args = [r for r in results if r["tool_ok"]]
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)
    return {
        "runs": len(results),
        "tool_calling_success_rate": rate(results, "passed"),
        "tool_selection_accuracy": rate(results, "tool_ok"),
        "argument_accuracy_given_correct_tools": rate(with_args, "args_ok"),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / len(results), 2),
        "by_category": {c: {"runs": len(rows), "success_rate": rate(rows, "passed")} for c, rows in sorted(by_category.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate controller tool calling.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only", default="", help="comma-separated case ids")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--cases", type=Path, default=CASES_PATH, help="e.g. eval/tool_calling_holdout.json")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text())["cases"]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    results = []
    for rep in range(args.repeats):
        for case in cases:
            r = run_case(case, rep)
            results.append(r)
            status = "PASS" if r["passed"] else "FAIL"
            print(f"{status}  {r['id']:<32} tools={r['actual_tools']}")
            if not r["passed"]:
                if not r["tool_ok"]:
                    print(f"      expected tools {r['expected_tools']}")
                for e in r["arg_errors"]:
                    print(f"      {e}")
                print(f"      reply: {r['reply'][:160]!r}")

    summary = summarize(results)
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    args.output.write_text(json.dumps({"model": getattr(g.LLM, "model_name", "unknown"), "date": date.today().isoformat(),
                                       "summary": summary, "results": results}, indent=2, ensure_ascii=False))
    print(f"\nSaved detailed results to: {args.output}")


if __name__ == "__main__":
    main()
