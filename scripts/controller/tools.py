"""Existing pipeline modules wrapped as tools for the LLM controller.

Each tool reuses a node function from orchestration.py unchanged. A tool reads the
current plan from the injected graph state, writes its full result back to the state
via Command(update=...), and returns only a short summary to the LLM.
`state` and `tool_call_id` are injected by LangGraph and are not visible to the LLM.
"""

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Dict, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from orchestration import DEFAULT_USER, agent1, agent2, agent3, agent_scope
from scripts.controller.feedback import apply_feedback
from scripts.controller.memory import get_preferences, set_preferences
from scripts.controller.replan import replan, today_index

WEEKDAYS = {
    **{name: i for i, name in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])},
    **{name: i for i, name in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])},
    **{f"周{c}": i for i, c in enumerate("一二三四五六日")},
    **{f"星期{c}": i for i, c in enumerate("一二三四五六日")},
    "周天": 6, "星期天": 6,
}


def days_until(day: str, today: Optional[date] = None) -> Optional[int]:
    """Days from today to the day the user named: a weekday (next one after today), "tomorrow" /
    "明天", "day after tomorrow" / "后天", or YYYY-MM-DD. None if it cannot be parsed."""
    today = today or date.today()
    text = (day or "").strip().lower()
    if text in ("tomorrow", "明天"):
        return 1
    if text in ("day after tomorrow", "后天"):
        return 2
    if text in WEEKDAYS:
        return (WEEKDAYS[text] - today.weekday()) % 7 or 7
    try:
        return (date.fromisoformat(text) - today).days
    except ValueError:
        return None


TAXONOMY = set(json.loads((Path(__file__).resolve().parents[2] / "data" / "taxonomy_skills.json").read_text()))


def _resolve_id(qid: str, current: dict) -> str:
    """Tolerate an id copied without its prefix ("SQL_185" for "lc:SQL_185") if the match is unique."""
    if qid in current:
        return qid
    matches = [known for known in current if known.split(":", 1)[-1] == qid]
    return matches[0] if len(matches) == 1 else qid


def _reply(tool_call_id: str, summary: str, **update) -> Command:
    # Full result goes into the graph state; the LLM only sees the summary.
    return Command(update={**update, "messages": [ToolMessage(summary, tool_call_id=tool_call_id)]})


@tool
def analyze_skills(
    reason: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    jd: Optional[str] = None,
) -> Command:
    """Re-analyze the job description and user background into skill weights.
    Use only when the target role or job description changes. Pass the new job
    description as `jd` if the user provided one."""
    jd = jd or state["jd"]
    update = agent1({**state, "jd": jd})
    top = sorted(update["weights"].items(), key=lambda x: -x[1])[:5]
    return _reply(tool_call_id, f"Re-weighted {len(update['weights'])} skills. Top 5: {top}", jd=jd, **update)


@tool
def retrieve_questions(
    skills: Dict[str, int],
    reason: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Retrieve extra candidate questions for some skills and add them to the
    candidate pool. `skills` maps taxonomy skill name -> number of questions needed.
    Does not change the plan itself."""
    unknown = [s for s in skills if s not in TAXONOMY]
    if unknown:
        return _reply(tool_call_id, f"Unknown skills {unknown}. Use taxonomy skill names, e.g. {sorted(state.get('weights') or {})[:5]}.")

    new = agent2({**state, "plan": skills})["retrieve_questions"]
    # Unlike agent2, append to the existing pool (deduplicated by id) instead of replacing it.
    pool = {k: list(v) for k, v in (state.get("retrieve_questions") or {}).items()}
    added = {}
    for skill, items in new.items():
        seen = {q["id"] for q in pool.get(skill, [])}
        fresh = [q for q in items if q["id"] not in seen]
        pool.setdefault(skill, []).extend(fresh)
        added[skill] = len(fresh)
    return _reply(tool_call_id, f"Added new candidates to pool: {added}", retrieve_questions=pool)


@tool
def generate_plan(
    reason: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    days_left: Optional[int] = None,
) -> Command:
    """Rebuild the whole study plan from the current skill weights: re-plan scope,
    re-retrieve candidates, and re-schedule all days. Expensive; use only for a full
    replan (e.g. after analyze_skills for a new role), not for small adjustments."""
    current = {**state, "days_left": days_left or state["days_left"]}
    update = {"days_left": current["days_left"]}
    # Agent3 needs fresh candidates for the new quotas, so retrieval runs in between.
    for node in (agent_scope, agent2, agent3):
        result = node(current)
        current.update(result)
        update.update(result)
    summary = (
        f"New {len(update['days'])}-day plan with {update['total_questions']} questions. "
        f"Skill quotas: {update['plan']}"
    )
    return _reply(tool_call_id, summary, **update)


@tool
def update_plan(
    reason: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    question_results: Optional[Dict[str, str]] = None,
    struggling_skills: Optional[List[str]] = None,
    days_remaining: Optional[int] = None,
    interview_day: Optional[str] = None,
    daily_load: Optional[int] = None,
    postpone_rest_of_today: bool = False,
) -> Command:
    """Adjust only the remaining plan (from tomorrow on). Today and finished questions are
    kept; unfinished questions from earlier days move forward. Use for: results of questions
    (`question_results` = {question_id: "done" | "wrong"}); skills the user says they are
    weak in (`struggling_skills`, any taxonomy names, also skills not in the current plan); study days
    left after today when the user states them (`days_remaining`), or the interview day as the
    user said it (`interview_day`: a weekday such as "Wed" / "周三", "tomorrow" / "明天",
    "day after tomorrow" / "后天", or YYYY-MM-DD; the tool computes the study days); a max number of questions per day (`daily_load`). Set
    `postpone_rest_of_today` = true only if the user says they will not finish the rest of
    today's questions; those then move to later days."""
    unknown = [s for s in struggling_skills or [] if s not in TAXONOMY]
    if unknown:
        return _reply(tool_call_id, f"Unknown skills {unknown}. Use taxonomy skill names.")
    if interview_day and days_remaining is None:
        # Date arithmetic stays in code: the interview day itself is not a study day.
        until = days_until(interview_day)
        if until is None:
            return _reply(tool_call_id, f"Could not read interview_day {interview_day!r}; use a weekday, 明天/后天 or YYYY-MM-DD.")
        days_remaining = until - 1

    state = {**state, "user_id": state.get("user_id") or DEFAULT_USER, "thread_id": state.get("thread_id") or ""}
    current = {t["id"]: t for day in state["days"] for t in day}
    question_results = {_resolve_id(q, current): r for q, r in (question_results or {}).items()}
    unknown_ids = [q for q in question_results if q not in current]
    newly_wrong = [
        current[q] for q, r in (question_results or {}).items()
        if r == "wrong" and q in current and current[q].get("status") != "wrong"
    ]

    update = {}
    if question_results or struggling_skills:
        # Feedback first (statuses, long-term memory, weights), then re-plan with the new weights.
        update = apply_feedback(state, question_results, struggling_skills)
        state.update(update)
    if daily_load:
        set_preferences(state["user_id"], daily_load=daily_load)
    per_day = daily_load or (get_preferences(state["user_id"]) or {}).get("daily_load")

    plan_update, note = replan(
        state, today_index(state), days_remaining, newly_wrong, struggling_skills or [], per_day, include_today=postpone_rest_of_today
    )
    update.update(plan_update)
    if unknown_ids:
        note += f" Ignored unknown question ids {unknown_ids}."
    return _reply(tool_call_id, note, **update)


TOOLS = [analyze_skills, retrieve_questions, generate_plan, update_plan]
