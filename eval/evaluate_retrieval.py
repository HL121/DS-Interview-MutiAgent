from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = PROJECT_ROOT / "eval/eval_cases.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval/retrieval_eval_results.json"
QUESTIONS_PATH = PROJECT_ROOT / "data/questions_normalized.jsonl"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def normalize_label(value: str) -> str:
    value = value.replace("\xa0", " ").lower().strip()
    value = value.replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def load_known_labels(path: Path = QUESTIONS_PATH) -> tuple[set[str], set[str]]:
    categories: set[str] = set()
    skills: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            categories.add(str(record.get("category", "")).lower())
            for skill in record.get("taxonomy_skills", []):
                skills.add(normalize_label(str(skill)))

    return categories, skills


def validate_case_labels(
    cases: list[dict[str, Any]],
    known_categories: set[str],
    known_skills: set[str],
) -> None:
    errors = []

    for case in cases:
        unknown_categories = sorted(expected_category_set(case) - known_categories)
        unknown_skills = sorted(expected_skill_set(case) - known_skills)

        if unknown_categories or unknown_skills:
            errors.append(
                {
                    "case_id": case.get("case_id", "unknown"),
                    "unknown_categories": unknown_categories,
                    "unknown_skills": unknown_skills,
                }
            )

    if errors:
        message = json.dumps(errors, indent=2)
        raise ValueError(
            "Some expected labels are not present in data/questions_normalized.jsonl.\n"
            f"{message}"
        )


def expected_category_set(case: dict[str, Any]) -> set[str]:
    return {str(category).lower() for category in case.get("expected_categories", [])}


def expected_skill_set(case: dict[str, Any]) -> set[str]:
    return {normalize_label(str(skill)) for skill in case.get("expected_skills", [])}


def result_category(result: dict[str, Any]) -> str:
    return str(result.get("category", "")).lower()


def result_skill_set(result: dict[str, Any]) -> set[str]:
    return {normalize_label(str(skill)) for skill in result.get("taxonomy_skills", [])}


def is_relevant(result: dict[str, Any], case: dict[str, Any]) -> bool:
    category_match = result_category(result) in expected_category_set(case)
    skill_match = bool(result_skill_set(result) & expected_skill_set(case))
    return category_match or skill_match


def relevance_grade(result: dict[str, Any], case: dict[str, Any]) -> int:
    grade = 0
    if result_category(result) in expected_category_set(case):
        grade += 1
    grade += len(result_skill_set(result) & expected_skill_set(case))
    return grade


def precision_at_k(results: list[dict[str, Any]], case: dict[str, Any], k: int) -> float:
    top_results = results[:k]
    if not top_results:
        return 0.0
    relevant_count = sum(1 for result in top_results if is_relevant(result, case))
    return relevant_count / len(top_results)


def recall_at_k(results: list[dict[str, Any]], case: dict[str, Any], k: int) -> float:
    expected_labels = {
        *(f"category:{category}" for category in expected_category_set(case)),
        *(f"skill:{skill}" for skill in expected_skill_set(case)),
    }
    if not expected_labels:
        return 0.0

    found_labels: set[str] = set()
    for result in results[:k]:
        category = result_category(result)
        if category:
            found_labels.add(f"category:{category}")
        for skill in result_skill_set(result):
            found_labels.add(f"skill:{skill}")

    return len(expected_labels & found_labels) / len(expected_labels)


def mrr_at_k(results: list[dict[str, Any]], case: dict[str, Any], k: int) -> float:
    for rank, result in enumerate(results[:k], start=1):
        if is_relevant(result, case):
            return 1.0 / rank
    return 0.0


def dcg_at_k(grades: list[int], k: int) -> float:
    score = 0.0
    for idx, grade in enumerate(grades[:k], start=1):
        score += (2**grade - 1) / math.log2(idx + 1)
    return score


def ndcg_at_k(results: list[dict[str, Any]], case: dict[str, Any], k: int) -> float:
    grades = [relevance_grade(result, case) for result in results[:k]]
    if not grades:
        return 0.0

    actual_dcg = dcg_at_k(grades, k)
    ideal_dcg = dcg_at_k(sorted(grades, reverse=True), k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def category_hit_rate(results: list[dict[str, Any]], case: dict[str, Any], k: int) -> float:
    expected_categories = expected_category_set(case)
    if not expected_categories:
        return 0.0

    found_categories = {result_category(result) for result in results[:k]}
    return len(expected_categories & found_categories) / len(expected_categories)


def skill_hit_rate(results: list[dict[str, Any]], case: dict[str, Any], k: int) -> float:
    expected_skills = expected_skill_set(case)
    if not expected_skills:
        return 0.0

    found_skills: set[str] = set()
    for result in results[:k]:
        found_skills.update(result_skill_set(result))

    return len(expected_skills & found_skills) / len(expected_skills)


def duplicate_rate(results: list[dict[str, Any]], k: int) -> float:
    top_results = results[:k]
    if not top_results:
        return 0.0

    titles = [normalize_label(str(result.get("title", ""))) for result in top_results]
    duplicate_count = len(titles) - len(set(titles))
    return duplicate_count / len(top_results)


def evaluate_one_case(
    retriever: Any,
    case: dict[str, Any],
    topk: int,
    fetch: int,
    mode: str,
    agent1: Any | None = None,
) -> dict[str, Any]:
    if mode == "retrieval_only":
        results = retrieve_from_manual_query(retriever, case, topk, fetch)
        agent1_output = None
        retrieval_queries = [case["query"]]
    elif mode == "agent1_retrieval":
        if agent1 is None:
            raise ValueError("agent1 is required when mode='agent1_retrieval'.")
        agent1_output = agent1.run(
            jd_text=case.get("jd", ""),
            user_desc=case.get("user_desc", ""),
        )
        retrieval_queries = select_agent1_skills(agent1_output)
        results = retrieve_from_agent1_skills(
            retriever=retriever,
            case=case,
            skills=retrieval_queries,
            topk=topk,
            fetch=fetch,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    metrics = {
        "precision_at_k": precision_at_k(results, case, topk),
        "recall_at_k": recall_at_k(results, case, topk),
        "mrr_at_k": mrr_at_k(results, case, topk),
        "ndcg_at_k": ndcg_at_k(results, case, topk),
        "category_hit_rate": category_hit_rate(results, case, topk),
        "skill_hit_rate": skill_hit_rate(results, case, topk),
        "duplicate_rate": duplicate_rate(results, topk),
    }

    return {
        "case_id": case["case_id"],
        "mode": mode,
        "query": case["query"],
        "retrieval_queries": retrieval_queries,
        "agent1_output": agent1_output,
        "metrics": metrics,
        "top_results": [
            {
                "rank": idx,
                "id": result.get("id"),
                "title": result.get("title"),
                "category": result.get("category"),
                "type": result.get("type"),
                "difficulty": result.get("difficulty"),
                "taxonomy_skills": result.get("taxonomy_skills", []),
                "retrieval_score": result.get("retrieval_score"),
                "adjusted_score": result.get("adjusted_score"),
                "is_relevant": is_relevant(result, case),
                "relevance_grade": relevance_grade(result, case),
            }
            for idx, result in enumerate(results[:topk], start=1)
        ],
    }


def retrieve_from_manual_query(
    retriever: Any,
    case: dict[str, Any],
    topk: int,
    fetch: int,
) -> list[dict[str, Any]]:
    return retriever.retrieve(
        query=case["query"],
        topk=topk,
        fetch=fetch,
        jd_text=case.get("jd", ""),
        user_desc=case.get("user_desc", ""),
    )


def select_agent1_skills(agent1_output: dict[str, Any]) -> list[str]:
    weights = agent1_output.get("weights", {}) or {}
    return [
        skill
        for skill, _ in sorted(weights.items(), key=lambda item: item[1], reverse=True)
    ]


def retrieve_from_agent1_skills(
    retriever: Any,
    case: dict[str, Any],
    skills: list[str],
    topk: int,
    fetch: int,
) -> list[dict[str, Any]]:
    if not skills:
        return []

    per_skill_topk = max(3, math.ceil(topk / len(skills)))
    candidates: list[dict[str, Any]] = []
    for skill in skills:
        candidates.extend(
            retriever.retrieve(
                query=skill,
                topk=per_skill_topk,
                fetch=fetch,
                skill_filter=skill,
                jd_text=case.get("jd", ""),
                user_desc=case.get("user_desc", ""),
            )
        )

    return merge_ranked_results(candidates)[:topk]


def merge_ranked_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_id: dict[str, dict[str, Any]] = {}
    for result in results:
        result_id = str(result.get("id") or result.get("question_id") or result.get("title"))
        current = best_by_id.get(result_id)
        if current is None or float(result.get("adjusted_score", 0.0)) > float(current.get("adjusted_score", 0.0)):
            best_by_id[result_id] = result

    return sorted(
        best_by_id.values(),
        key=lambda result: float(result.get("adjusted_score", 0.0)),
        reverse=True,
    )


def summarize_results(case_results: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = case_results[0]["metrics"].keys()
    return {
        metric_name: mean(result["metrics"][metric_name] for result in case_results)
        for metric_name in metric_names
    }


def print_summary(summary: dict[str, float]) -> None:
    print("\nOverall retrieval metrics")
    print("-" * 32)
    for metric_name, value in summary.items():
        print(f"{metric_name}: {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with simple ranking metrics.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--mode",
        choices=["retrieval_only", "agent1_retrieval"],
        default="retrieval_only",
    )
    parser.add_argument(
        "--retriever",
        choices=["base", "agentic"],
        default="base",
        help="base uses fixed hybrid RAG; agentic adds query planning and adaptive retry.",
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--fetch", type=int, default=50)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    known_categories, known_skills = load_known_labels()
    validate_case_labels(cases, known_categories, known_skills)

    if args.retriever == "agentic":
        from scripts.Agent2.agentic_retrieval import init_agentic_retriever

        retriever = init_agentic_retriever()
    else:
        from scripts.Agent2.langchain_retrieval import init_retriever

        retriever = init_retriever()

    if args.mode == "agent1_retrieval":
        from scripts.Agent1.skill_analyzer_agent import SkillAnalyzerAgent
        from scripts.langchain_llm import get_chat_model

        agent1 = SkillAnalyzerAgent(client=get_chat_model())
    else:
        agent1 = None

    case_results = [
        evaluate_one_case(
            retriever=retriever,
            case=case,
            topk=args.topk,
            fetch=args.fetch,
            mode=args.mode,
            agent1=agent1,
        )
        for case in cases
    ]
    summary = summarize_results(case_results)

    payload = {
        "mode": args.mode,
        "retriever": args.retriever,
        "topk": args.topk,
        "fetch": args.fetch,
        "num_cases": len(cases),
        "summary": summary,
        "cases": case_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print_summary(summary)
    print(f"\nSaved detailed results to: {args.output}")


if __name__ == "__main__":
    main()
