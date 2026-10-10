"""Re-plan only the remaining days after user feedback. Deterministic, no LLM.

Rules:
- Today is kept as is (unless include_today: the user says they won't finish today).
- Earlier days keep only their done/wrong questions; their unfinished questions and all
  questions after today are rescheduled from tomorrow on.
- Reinforcement: +1 question per newly wrong question, +2 per self-reported weak skill,
  taken from the candidate pool (easier first), never repeating used/completed questions.
- Daily cap: user's daily_load if set, else original daily average + 1. If there are more
  questions than capacity, drop lowest-weight skills first, hard first.
"""

import math
from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

from scripts.Agent3.Planning_Agent import DIFF, deterministic_summaries, normalize_skill_name, schedule, summarize_day_metadata
from scripts.controller.memory import completed_question_ids


def today_index(state: dict, today: Optional[date] = None) -> int:
    """Which plan day (0-based) today is, counted from the plan's start_date."""
    start = state.get("start_date")
    if not start:
        return 0
    return max(0, ((today or date.today()) - date.fromisoformat(start)).days)


def _reinforcements(pool: Dict[str, list], skills: Iterable[str], count: int, used: set) -> List[dict]:
    """Pick `count` unused pool questions that cover any of `skills`, easier first."""
    skills = {normalize_skill_name(s) for s in skills if s}
    candidates = {
        q["id"]: {**q, "requested_skill": key}
        for key, items in pool.items()
        for q in items
        if q["id"] not in used
        and (normalize_skill_name(key) in skills or skills & {normalize_skill_name(s) for s in q.get("taxonomy_skills") or []})
    }
    ranked = sorted(
        candidates.values(),
        key=lambda q: (DIFF.get(q.get("difficulty"), 2), -float(q.get("adjusted_score") or q.get("retrieval_score") or 0)),
    )
    picked = [{**q, "status": "todo"} for q in ranked[:count]]
    used.update(q["id"] for q in picked)
    return picked


def replan(
    state: dict,
    today_idx: int,
    days_remaining: Optional[int] = None,
    newly_wrong: Iterable[dict] = (),
    struggling_skills: Iterable[str] = (),
    per_day: Optional[int] = None,
    include_today: bool = False,
) -> Tuple[dict, str]:
    """Returns (state update, short summary for the LLM). days_remaining = days after today."""
    days = state["days"]
    cut = min(today_idx + 1, len(days))  # days[:cut] are today and earlier
    n_future = days_remaining if days_remaining is not None else len(days) - cut
    if n_future <= 0:
        return {}, "No days left after today; the plan was not changed."

    is_todo = lambda t: t.get("status", "todo") == "todo"
    today_pos = today_idx if today_idx < len(days) else None

    def stays(i, task):
        # Finished questions always stay; today's unfinished ones stay unless include_today.
        return not is_todo(task) or (i == today_pos and not include_today)

    frozen = [[t for t in day if stays(i, t)] for i, day in enumerate(days[:cut])]
    if frozen:
        # Questions finished ahead of schedule are kept under today.
        frozen[-1] += [t for day in days[cut:] for t in day if not is_todo(t)]
    moved_tasks = [t for i, day in enumerate(days[:cut]) for t in day if not stays(i, t)]
    pending = moved_tasks + [t for day in days[cut:] for t in day if is_todo(t)]
    moved = len(moved_tasks)

    used = {t["id"] for day in days for t in day} | completed_question_ids(state["user_id"])
    pool = state.get("retrieve_questions") or {}
    added, short = [], []
    for task in newly_wrong:
        picked = _reinforcements(pool, task.get("taxonomy_skills") or [task.get("requested_skill")], 1, used)
        added += picked
        if not picked:
            short.append(task.get("requested_skill"))
    for skill in struggling_skills:
        picked = _reinforcements(pool, [skill], 2, used)
        added += picked
        if len(picked) < 2:
            short.append(skill)

    # Default cap: one more than the original daily average, so moved questions can be absorbed.
    # A user-set daily_load (per_day) is a hard limit instead.
    per_day = per_day or math.ceil(sum(len(d) for d in days) / len(days)) + 1
    capacity = per_day * n_future
    tasks = pending + added
    dropped = []
    if len(tasks) > capacity:
        weights = state.get("weights") or {}
        # Lowest-weight skill first; within a skill, hard before medium before easy.
        order = sorted(tasks, key=lambda t: (weights.get(t.get("requested_skill"), 0), -DIFF.get(t.get("difficulty"), 2)))
        dropped = order[: len(tasks) - capacity]
        drop_ids = {t["id"] for t in dropped}
        tasks = [t for t in tasks if t["id"] not in drop_ids]

    new_days = frozen + schedule(tasks, n_future, per_day)
    summaries = deterministic_summaries(summarize_day_metadata(new_days))

    notes = [f"Re-planned {n_future} day(s) after today (max {per_day}/day)."]
    if moved:
        notes.append(f"Moved {moved} unfinished question(s) forward.")
    if added:
        notes.append(f"Added {len(added)} reinforcement question(s).")
    if short:
        notes.append(f"Candidate pool ran short for {sorted(set(short))}; call retrieve_questions, then update_plan again.")
    if dropped:
        notes.append(f"Dropped {len(dropped)} low-priority question(s) to fit the time left.")
    update = {"days": new_days, "summaries": summaries, "days_left": len(new_days)}
    return update, " ".join(notes)
