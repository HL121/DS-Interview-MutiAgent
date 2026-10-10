"""Long-term memory: per-user records that persist across preparation sessions.

feedback_log keeps one row per (question, skill) result. Completed questions, past
mistakes and weak skills are all derived from it. Self-reported weak skills are rows
with question_id NULL and result "struggling".
"""

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable, Optional

from scripts.Agent3.Planning_Agent import normalize_skill_name

LONG_TERM_DB = Path(os.getenv("LONG_TERM_DB", Path(__file__).resolve().parents[2] / ".state" / "long_term.db"))


def _connect() -> sqlite3.Connection:
    # A short-lived connection per call, so it is safe under Streamlit's threads.
    LONG_TERM_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LONG_TERM_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS feedback_log ("
        "user_id TEXT, thread_id TEXT, question_id TEXT, skill TEXT, result TEXT, "
        "created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS preferences ("
        "user_id TEXT PRIMARY KEY, daily_load INTEGER, preferred_difficulty TEXT)"
    )
    return conn


def log_feedback(user_id: str, thread_id: str, rows: Iterable[tuple]) -> None:
    """rows: (question_id or None, skill, result) with result in done / wrong / struggling."""
    with closing(_connect()) as conn, conn:
        conn.executemany(
            "INSERT INTO feedback_log (user_id, thread_id, question_id, skill, result) VALUES (?, ?, ?, ?, ?)",
            [(user_id, thread_id, qid, normalize_skill_name(skill), result) for qid, skill, result in rows],
        )


def completed_question_ids(user_id: str) -> set:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT DISTINCT question_id FROM feedback_log WHERE user_id = ? AND question_id IS NOT NULL",
            (user_id,),
        ).fetchall()
    return {r[0] for r in rows}


def recent_results(user_id: str, skill: str, limit: int = 5) -> list:
    """Latest results for one skill, newest first. Skill names are stored normalized
    ("supervised_machine learning" and "supervised_machine_learning" are the same skill)."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT result FROM feedback_log WHERE user_id = ? AND skill = ? ORDER BY rowid DESC LIMIT ?",
            (user_id, normalize_skill_name(skill), limit),
        ).fetchall()
    return [r[0] for r in rows]


def recent_log(user_id: str, limit: int = 5) -> list:
    """Latest feedback records, newest first: (question_id or None, skill, result).
    One answer can produce several rows (one per skill); they are collapsed to one."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT question_id, skill, result FROM feedback_log WHERE user_id = ? ORDER BY rowid DESC LIMIT ?",
            (user_id, limit * 4),
        ).fetchall()
    seen, out = set(), []
    for qid, skill, result in rows:
        key = (qid or skill, result)
        if key not in seen:
            seen.add(key)
            out.append((qid, skill, result))
    return out[:limit]


def get_preferences(user_id: str) -> Optional[dict]:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT daily_load, preferred_difficulty FROM preferences WHERE user_id = ?", (user_id,)
        ).fetchone()
    return {"daily_load": row[0], "preferred_difficulty": row[1]} if row else None


def set_preferences(user_id: str, daily_load: Optional[int] = None, preferred_difficulty: Optional[str] = None) -> None:
    # Only overwrite the fields that are given.
    current = get_preferences(user_id) or {}
    daily_load = daily_load if daily_load is not None else current.get("daily_load")
    preferred_difficulty = preferred_difficulty or current.get("preferred_difficulty")
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO preferences (user_id, daily_load, preferred_difficulty) VALUES (?, ?, ?)",
            (user_id, daily_load, preferred_difficulty),
        )
