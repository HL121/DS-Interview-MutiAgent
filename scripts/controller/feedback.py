"""Feedback loop: user feedback -> long-term memory -> skill weight update.

Deterministic rules, no LLM. Weights are always recomputed from the Agent1 base
weights (not multiplied on top of the previous result), so repeated feedback cannot
make a weight grow without bound.
"""

from typing import Dict, Iterable, Optional

from scripts.controller.memory import log_feedback, recent_results

WEAK_FACTOR = 1.5
MASTERED_FACTOR = 0.8


def skill_factor(user_id: str, skill: str) -> float:
    recent = recent_results(user_id, skill, limit=5)  # newest first
    if len(recent) >= 3 and all(r == "done" for r in recent[:3]):
        return MASTERED_FACTOR
    if "struggling" in recent:
        return WEAK_FACTOR
    answered = [r for r in recent if r in ("done", "wrong")]
    if len(answered) >= 2 and answered.count("wrong") / len(answered) >= 0.5:
        return WEAK_FACTOR
    return 1.0


def adjust_weights(base_weights: Dict[str, float], user_id: str) -> Dict[str, float]:
    factors = {skill: skill_factor(user_id, skill) for skill in base_weights}
    if all(f == 1.0 for f in factors.values()):
        return dict(base_weights)  # no signal yet: keep Agent1 weights exactly
    raw = {skill: w * factors[skill] for skill, w in base_weights.items()}
    total = sum(raw.values())
    if total <= 0:
        return dict(base_weights)
    return {skill: round(w / total, 4) for skill, w in raw.items()}


def apply_feedback(
    state: dict,
    question_results: Optional[Dict[str, str]] = None,
    struggling_skills: Optional[Iterable[str]] = None,
) -> dict:
    """question_results: {question_id: "done" | "wrong"}; struggling_skills: self-reported weak skills.
    Returns the state update: plan with new statuses and recomputed weights."""
    question_results = question_results or {}
    days = [[dict(task) for task in day] for day in state.get("days") or []]

    rows = []
    for day in days:
        for task in day:
            result = question_results.get(task["id"])
            # Skip invalid results and repeats of the same status, so re-clicking is not double counted.
            if result not in ("done", "wrong") or task.get("status") == result:
                continue
            task["status"] = result
            # Use the question's own skill labels; requested_skill can be wrong (see optimize.md 11.1).
            for skill in task.get("taxonomy_skills") or [task.get("requested_skill")]:
                if skill:
                    rows.append((task["id"], skill, result))
    rows += [(None, skill, "struggling") for skill in struggling_skills or []]

    log_feedback(state["user_id"], state["thread_id"], rows)
    base = state.get("base_weights") or state.get("weights") or {}
    return {"days": days, "weights": adjust_weights(base, state["user_id"])}
