"""Deterministic checks for re-planning (no API calls).

Run: python tests/test_replan.py   (or: pytest tests/test_replan.py)
"""

import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

# Isolated databases and a dummy key, set before any project import.
_TMP = tempfile.mkdtemp()
os.environ["LONG_TERM_DB"] = os.path.join(_TMP, "long_term.db")
os.environ["CHECKPOINT_DB"] = os.path.join(_TMP, "checkpoints.db")
os.environ.setdefault("OPENAI_API_KEY", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.controller.feedback import apply_feedback  # noqa: E402
from scripts.controller.memory import completed_question_ids  # noqa: E402
from scripts.controller.replan import replan, schedule, today_index  # noqa: E402

SQL, CLS = "sql_window_function", "classification"
DIFFS = ["easy", "medium", "medium", "hard"]


def q(qid, skill, difficulty, status="todo"):
    return {"id": qid, "title": qid, "type": "coding", "difficulty": difficulty,
            "requested_skill": skill, "taxonomy_skills": [skill], "status": status}


def make_state(user, start=None):
    """4 days x 4 questions (SQL and classification alternate), plus spare pool candidates."""
    days = [[q(f"d{d}_{i}", SQL if i % 2 == 0 else CLS, DIFFS[i]) for i in range(4)] for d in range(4)]
    pool = {
        SQL: [t for day in days for t in day if t["requested_skill"] == SQL]
             + [q(f"pool_sql_{i}", SQL, "medium" if i == 0 else "easy") for i in range(4)],
        CLS: [t for day in days for t in day if t["requested_skill"] == CLS]
             + [q(f"pool_cls_{i}", CLS, "easy") for i in range(4)],
    }
    weights = {SQL: 0.6, CLS: 0.4}
    return {"days": days, "retrieve_questions": pool, "weights": weights, "base_weights": dict(weights),
            "user_id": user, "thread_id": "t1", "start_date": (start or date.today()).isoformat()}


def ids(days):
    return [t["id"] for day in days for t in day]


def check_invariants(before_days, after_days, cut, cap, user, include_today=False):
    """Checks shared by every scenario. Day cut-1 is today."""
    # 1. Earlier days keep only finished questions; today is unchanged unless include_today.
    for i in range(cut):
        finished = [t["id"] for t in before_days[i] if t.get("status") != "todo"]
        assert set(finished) <= set(ids([after_days[i]])), f"day {i + 1} lost a finished question"
        if i == cut - 1 and not include_today:
            assert set(ids([before_days[i]])) <= set(ids([after_days[i]])), "today must be kept as is"
        else:
            assert all(t.get("status") != "todo" for t in after_days[i]), f"day {i + 1} still has todo questions"
    future = after_days[cut:]
    # 2. Completed questions never come back.
    assert not set(ids(future)) & completed_question_ids(user), "completed question re-scheduled"
    # 3. No duplicates anywhere.
    assert len(ids(after_days)) == len(set(ids(after_days))), "duplicate question"
    # 4. Daily cap respected.
    assert all(len(day) <= cap for day in future), f"a day exceeds cap {cap}: {[len(d) for d in future]}"
    # 5. Everything in the future is still todo.
    assert all(t["status"] == "todo" for day in future for t in day)


def test_schedule_rules():
    tasks = ([q(f"e{i}", SQL, "easy") for i in range(25)] + [q(f"m{i}", CLS, "medium") for i in range(20)]
             + [q(f"h{i}", SQL, "hard") for i in range(5)])
    days = schedule(tasks, 7)
    counts = [len(d) for d in days]
    assert sum(counts) == 50 and sorted(ids(days)) == sorted(t["id"] for t in tasks)
    assert max(counts[:-1]) - min(counts[:-1]) <= 1, counts          # balanced
    assert counts[-1] < min(counts[:-1]), counts                       # lighter review day
    assert all(sum(t["difficulty"] == "hard" for t in d) <= 1 for d in days)
    level = {"easy": 1, "medium": 2, "hard": 3}
    avg = [sum(level[t["difficulty"]] for t in d) / len(d) for d in days]
    assert avg[0] < avg[-2], avg                                       # difficulty ramps up
    assert all([level[t["difficulty"]] for t in d] == sorted(level[t["difficulty"]] for t in d) for d in days)


def test_unfinished_today():
    user = "u_unfinished"
    state = make_state(user)
    state.update(apply_feedback(state, {"d0_0": "done", "d0_1": "done"}))
    todo_before = {t["id"] for day in state["days"] for t in day if t["status"] == "todo"}
    update, note = replan(state, today_index(state), include_today=True)
    days = update["days"]
    check_invariants(state["days"], days, cut=1, cap=5, user=user, include_today=True)
    assert ids([days[0]]) == ["d0_0", "d0_1"]
    assert {"d0_2", "d0_3"} <= set(ids(days[1:])), "unfinished questions not moved forward"
    assert set(ids(days[1:])) == todo_before, "a todo question was lost or added"
    assert len(days) == 4 and "Moved 2" in note


def test_today_kept_by_default():
    user = "u_morning"
    state = make_state(user, start=date.today() - timedelta(days=1))  # today = day 2
    state.update(apply_feedback(state, {"d0_0": "done"}))            # day 1: 1 done, 3 unfinished
    # per_day=7 leaves enough room, so nothing is dropped and every moved question is visible.
    update, note = replan(state, today_index(state), struggling_skills=[CLS], per_day=7)
    days = update["days"]
    check_invariants(state["days"], days, cut=2, cap=7, user=user)
    assert ids([days[1]]) == ids([state["days"][1]]), "today's questions must not move"
    assert ids([days[0]]) == ["d0_0"] and {"d0_1", "d0_2", "d0_3"} <= set(ids(days[2:]))
    assert "Moved 3" in note


def test_wrong_and_struggling():
    user = "u_wrong"
    state = make_state(user)
    newly_wrong = [state["days"][0][0]]  # d0_0, SQL
    state.update(apply_feedback(state, {"d0_0": "wrong", "d0_1": "done", "d0_2": "done", "d0_3": "done"},
                                struggling_skills=[CLS]))
    update, note = replan(state, 0, newly_wrong=newly_wrong, struggling_skills=[CLS])
    days = update["days"]
    check_invariants(state["days"], days, cut=1, cap=5, user=user)
    added = [i for i in ids(days[1:]) if i.startswith("pool_")]
    assert sum(i.startswith("pool_sql") for i in added) == 1, added    # +1 for the wrong SQL question
    assert sum(i.startswith("pool_cls") for i in added) == 2, added    # +2 for the struggling skill
    assert "pool_sql_0" not in added, "should prefer easier reinforcement"
    assert "Added 3" in note


def test_interview_earlier():
    user = "u_earlier"
    state = make_state(user)
    update, note = replan(state, 0, days_remaining=1)
    days = update["days"]
    check_invariants(state["days"], days, cut=1, cap=5, user=user)
    assert len(days) == 2 and len(days[1]) == 5
    kept = days[1]
    # Lower-weight skill (classification) is dropped before any SQL question.
    assert all(t["requested_skill"] == SQL for t in kept), [t["id"] for t in kept]
    assert "Dropped" in note


def test_daily_load_and_today_index():
    user = "u_load"
    state = make_state(user, start=date.today() - timedelta(days=1))
    assert today_index(state) == 1
    update, _ = replan(state, today_index(state), per_day=3)
    days = update["days"]
    check_invariants(state["days"], days, cut=2, cap=3, user=user)
    assert len(days) == 4


def test_last_day_no_change():
    state = make_state("u_last", start=date.today() - timedelta(days=3))
    update, note = replan(state, today_index(state))
    assert update == {} and "No days left" in note


def test_update_plan_tool():
    from scripts.controller.tools import update_plan

    state = make_state("u_tool")
    cmd = update_plan.invoke({"type": "tool_call", "id": "c1", "name": "update_plan", "args": {
        "reason": "user says window functions went badly",
        "question_results": {"d0_0": "wrong", "nope": "done"},
        "struggling_skills": [SQL],
        "state": state,
    }})
    msg = cmd.update["messages"][0].content
    days = cmd.update["days"]
    assert days[0][0]["status"] == "wrong"
    assert cmd.update["weights"][SQL] > state["weights"][SQL], "weight should rise before re-planning"
    assert "Ignored unknown question ids ['nope']" in msg
    bad = update_plan.invoke({"type": "tool_call", "id": "c2", "name": "update_plan",
                              "args": {"reason": "x", "struggling_skills": ["sql_magic"], "state": state}})
    assert "Unknown skills" in bad.update["messages"][0].content and "days" not in bad.update


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed")
