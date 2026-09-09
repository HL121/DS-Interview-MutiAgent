"""
Final deterministic dataset normalizer.

This script reads the already-produced coding and theory datasets and writes one
canonical JSONL file for retrieval and planning:

{
  "id": "string",
  "type": "coding | theory",
  "category": "SQL | Pandas | Algorithms | ML | Statistics | Experimentation | Product",
  "title": "string",
  "question": "string",
  "answer": "string",
  "difficulty": "easy | medium | hard",
  "taxonomy_skills": ["string"],
  "source": "string",
  "url": "string"
}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CODING_PATHS = [
    Path("data/pandas.json"),
    Path("data/algorithms.json"),
    Path("data/sql.json"),
]
DEFAULT_THEORY_PATH = Path("data/Theory.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/questions_normalized.jsonl")

REQUIRED_FIELDS = {
    "id",
    "type",
    "category",
    "title",
    "question",
    "answer",
    "difficulty",
    "taxonomy_skills",
    "source",
    "url",
}
VALID_TYPES = {"coding", "theory"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
DIFFICULTY_TAG_RE = re.compile(r"^(algo|sql|pandas)_(easy|medium|hard)$")

THEORY_CATEGORY_KEYWORDS = {
    "Experimentation": {
        "ab_testing",
        "a_b_testing",
        "causal_inference",
        "experimentation",
        "hypothesis_testing",
        "online_experiments",
    },
    "Statistics": {
        "bayes",
        "confidence_interval",
        "probability",
        "sampling",
        "statistical",
        "statistics",
        "variance",
    },
    "Product": {
        "metrics",
        "product",
        "ranking_and_search",
        "recommender_systems",
    },
    "SQL": {
        "database",
        "databases",
        "sql",
    },
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_skill(value: Any) -> str:
    skill = clean_text(value).lower()
    skill = skill.replace("-", "_")
    skill = re.sub(r"[^a-z0-9_]+", "_", skill)
    skill = re.sub(r"_+", "_", skill)
    return skill.strip("_")


def normalize_skills(values: Any) -> list[str]:
    if values is None:
        raw_values: list[Any] = []
    elif isinstance(values, list):
        raw_values = values
    else:
        raw_values = [values]

    skills: list[str] = []
    seen = set()
    for raw_skill in raw_values:
        skill = normalize_skill(raw_skill)
        if not skill or DIFFICULTY_TAG_RE.match(skill):
            continue
        if skill not in seen:
            skills.append(skill)
            seen.add(skill)
    return skills


def infer_difficulty(record: dict[str, Any], default: str) -> str:
    metadata = record.get("metadata", {}) or {}
    candidates: list[Any] = [
        record.get("difficulty"),
        metadata.get("difficulty"),
    ]

    skills = metadata.get("taxonomy_skills") or metadata.get("taxonomy_skill") or []
    if not isinstance(skills, list):
        skills = [skills]
    candidates.extend(skills)

    text = record.get("vector_content") or record.get("vector_text") or ""
    match = re.search(r"\bDifficulty:\s*(Easy|Medium|Hard)\b", str(text), re.I)
    if match:
        candidates.append(match.group(1))

    for candidate in candidates:
        value = normalize_skill(candidate)
        if value in VALID_DIFFICULTIES:
            return value
        tag_match = DIFFICULTY_TAG_RE.match(value)
        if tag_match:
            return tag_match.group(2)

    return default


def split_question_and_solution(text: str) -> tuple[str, str]:
    match = re.search(r"\n\s*Solution\s*\n", text, flags=re.I)
    if not match:
        return text.strip(), ""
    question = text[: match.start()].strip()
    solution = text[match.end() :].strip()
    return question, solution


def normalize_coding_record(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    metadata = record.get("metadata", {}) or {}
    category = clean_text(metadata.get("category")) or "Algorithms"
    raw_id = clean_text(metadata.get("id") or record.get("id"))
    title = clean_text(metadata.get("title"))
    vector_text = record.get("vector_content") or record.get("vector_text") or ""
    question, inline_solution = split_question_and_solution(str(vector_text))

    answer_parts = [
        clean_text(metadata.get("solution_summary")),
        clean_text(record.get("solution_raw")),
    ]
    if inline_solution and not any(answer_parts):
        answer_parts.append(clean_text(inline_solution))
    answer = "\n\n".join(part for part in answer_parts if part)

    return {
        "id": f"lc:{raw_id}",
        "type": "coding",
        "category": category,
        "title": title,
        "question": question,
        "answer": answer,
        "difficulty": infer_difficulty(record, default="medium"),
        "taxonomy_skills": normalize_skills(metadata.get("taxonomy_skills")),
        "source": source_path.name,
        "url": clean_text(metadata.get("url")),
    }


def infer_theory_category(title: str, skills: list[str], subdomain: str) -> str:
    haystack = " ".join([title, subdomain, *skills]).lower()
    for category, keywords in THEORY_CATEGORY_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return category
    return "ML"


def normalize_theory_record(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    metadata = record.get("metadata", {}) or {}
    raw_id = clean_text(record.get("id"))
    title = clean_text(metadata.get("title"))
    answer = clean_text(record.get("vector_content") or record.get("vector_text"))
    raw_skills = metadata.get("taxonomy_skills") or metadata.get("taxonomy_skill")
    skills = normalize_skills(raw_skills or metadata.get("subdomain"))
    subdomain = normalize_skill(metadata.get("subdomain"))

    return {
        "id": f"th:{raw_id}",
        "type": "theory",
        "category": infer_theory_category(title, skills, subdomain),
        "title": title,
        "question": title,
        "answer": answer,
        "difficulty": infer_difficulty(record, default="easy"),
        "taxonomy_skills": skills,
        "source": clean_text(metadata.get("source")) or source_path.name,
        "url": clean_text(metadata.get("url")),
    }


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return data


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")

    if record.get("type") not in VALID_TYPES:
        errors.append(f"invalid type: {record.get('type')!r}")

    if record.get("difficulty") not in VALID_DIFFICULTIES:
        errors.append(f"invalid difficulty: {record.get('difficulty')!r}")

    skills = record.get("taxonomy_skills")
    if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
        errors.append("taxonomy_skills must be a list of strings")
    else:
        leaked_tags = [s for s in skills if DIFFICULTY_TAG_RE.match(s)]
        if leaked_tags:
            errors.append(f"difficulty tags leaked into taxonomy_skills: {leaked_tags}")

    for field in ("id", "title", "question", "difficulty", "source"):
        if not clean_text(record.get(field)):
            errors.append(f"blank field: {field}")

    return errors


def normalize_all(coding_paths: list[Path], theory_path: Path) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for path in coding_paths:
        for record in load_json_array(path):
            normalized.append(normalize_coding_record(record, path))

    for record in load_jsonl(theory_path):
        normalized.append(normalize_theory_record(record, theory_path))

    seen_ids = set()
    validation_errors: list[str] = []
    for idx, record in enumerate(normalized, start=1):
        record_errors = validate_record(record)
        if record["id"] in seen_ids:
            record_errors.append(f"duplicate id: {record['id']}")
        seen_ids.add(record["id"])

        if record_errors:
            validation_errors.append(
                f"record {idx} ({record.get('id', '<missing id>')}): "
                + "; ".join(record_errors)
            )

    if validation_errors:
        sample = "\n".join(validation_errors[:20])
        extra = "" if len(validation_errors) <= 20 else f"\n... {len(validation_errors) - 20} more"
        raise ValueError(f"Normalized dataset failed validation:\n{sample}{extra}")

    return normalized


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(records: list[dict[str, Any]], output_path: Path) -> None:
    type_counts = Counter(r["type"] for r in records)
    category_counts = Counter(r["category"] for r in records)
    difficulty_counts = Counter(r["difficulty"] for r in records)
    skill_counts = Counter(skill for r in records for skill in r["taxonomy_skills"])

    print(f"Wrote {len(records)} records to {output_path}")
    print("Type counts:", dict(sorted(type_counts.items())))
    print("Difficulty counts:", dict(sorted(difficulty_counts.items())))
    print("Category counts:", dict(sorted(category_counts.items())))
    print("Top taxonomy skills:", skill_counts.most_common(15))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize DS interview question datasets.")
    parser.add_argument(
        "--coding-paths",
        nargs="+",
        type=Path,
        default=DEFAULT_CODING_PATHS,
        help="JSON array files for coding questions.",
    )
    parser.add_argument(
        "--theory-path",
        type=Path,
        default=DEFAULT_THEORY_PATH,
        help="JSONL file for theory questions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output canonical JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = normalize_all(args.coding_paths, args.theory_path)
    write_jsonl(records, args.output)
    print_summary(records, args.output)


if __name__ == "__main__":
    main()
