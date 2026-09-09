from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from scripts.Agent2.langchain_retrieval import (
    QdrantQuestionRetriever,
    init_retriever,
    normalize_skill,
)
from scripts.langchain_llm import get_chat_model


class RetrievalQuery(BaseModel):
    query: str = Field(description="A focused retrieval query.")
    skill_filter: str = Field(default="", description="Optional taxonomy skill filter.")
    category_filter: str = Field(default="", description="Optional category filter.")
    reason: str = Field(default="", description="Why this query should be searched.")


class RetrievalPlan(BaseModel):
    queries: List[RetrievalQuery] = Field(description="Two to four retrieval queries.")


class CoverageReport(BaseModel):
    needs_retry: bool
    retry_query: str = ""
    retry_skill_filter: str = ""
    reason: str = ""


class AgenticQuestionRetriever:
    """
    Controlled agentic RAG wrapper around the existing Qdrant/BM25 retriever.

    The LLM plans a few retrieval queries. Python code executes the retrieval,
    checks coverage, optionally retries, and merges the final candidate pool.
    """

    def __init__(
        self,
        base_retriever: QdrantQuestionRetriever,
        llm: Any,
    ) -> None:
        self.base_retriever = base_retriever
        self.llm = llm

    def retrieve(
        self,
        query: str,
        topk: int = 10,
        fetch: int = 50,
        lambda_: float = 0.7,
        type_filter: Optional[str] = None,
        skill_filter: Optional[str] = None,
        difficulty_distribution: Optional[Dict[str, float]] = None,
        category_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None,
        jd_text: str = "",
        user_desc: str = "",
        role_type: str = "",
    ) -> List[Dict[str, Any]]:
        plan = self.plan_queries(
            query=query,
            skill_filter=skill_filter,
            category_filter=category_filter,
            jd_text=jd_text,
            user_desc=user_desc,
            role_type=role_type,
            type_filter=type_filter,
            difficulty_filter=difficulty_filter,
        )
        candidates = self.run_retrieval_plan(
            plan=plan,
            topk=topk,
            fetch=fetch,
            lambda_=lambda_,
            type_filter=type_filter,
            default_skill_filter=skill_filter,
            default_category_filter=category_filter,
            difficulty_distribution=difficulty_distribution,
            difficulty_filter=difficulty_filter,
            jd_text=jd_text,
            user_desc=user_desc,
            role_type=role_type,
        )

        coverage = self.check_coverage(
            candidates=candidates,
            topk=topk,
            skill_filter=skill_filter,
            query=query,
            jd_text=jd_text,
            user_desc=user_desc,
        )
        if coverage.needs_retry and coverage.retry_query:
            retry_results = self.retrieval_tool(
                query=coverage.retry_query,
                topk=max(topk, 5),
                fetch=fetch,
                lambda_=lambda_,
                type_filter=type_filter,
                skill_filter=coverage.retry_skill_filter or None,
                category_filter=category_filter,
                difficulty_distribution=difficulty_distribution,
                difficulty_filter=difficulty_filter,
                jd_text=jd_text,
                user_desc=user_desc,
                role_type=role_type,
            )
            candidates.extend(self.mark_agentic_results(retry_results, "adaptive retry"))

        return self.merge_and_dedupe(candidates)[:topk]

    def plan_queries(
        self,
        query: str,
        skill_filter: Optional[str],
        category_filter: Optional[str],
        jd_text: str,
        user_desc: str,
        role_type: str,
        type_filter: Optional[str],
        difficulty_filter: Optional[str],
    ) -> RetrievalPlan:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a retrieval query planner for a data science interview question bank. "
                    "Return concise search queries using the provided taxonomy skill when useful. "
                    "Do not invent unsupported categories. Valid categories are SQL, Pandas, Algorithms, ML, Statistics, and Product.",
                ),
                (
                    "human",
                    """
Target skill or query:
{query}

Skill filter:
{skill_filter}

Category filter:
{category_filter}

Type filter:
{type_filter}

Difficulty filter:
{difficulty_filter}

Role type:
{role_type}

Job description:
{jd_text}

Candidate background:
{user_desc}

Create 2 to 4 retrieval queries. Prefer one strict skill-focused query and one broader context query.
""".strip(),
                ),
            ]
        )
        chain = prompt | self.llm.with_structured_output(RetrievalPlan)

        try:
            plan = chain.invoke(
                {
                    "query": query,
                    "skill_filter": skill_filter or "",
                    "category_filter": category_filter or "",
                    "type_filter": type_filter or "",
                    "difficulty_filter": difficulty_filter or "",
                    "role_type": role_type or "",
                    "jd_text": jd_text or "",
                    "user_desc": user_desc or "",
                }
            )
        except Exception:
            return self.default_plan(query, skill_filter, category_filter, jd_text, user_desc)

        if not plan.queries:
            return self.default_plan(query, skill_filter, category_filter, jd_text, user_desc)

        return self.clean_plan(plan, query, skill_filter, category_filter)

    def default_plan(
        self,
        query: str,
        skill_filter: Optional[str],
        category_filter: Optional[str],
        jd_text: str,
        user_desc: str,
    ) -> RetrievalPlan:
        base_query = query.replace("_", " ")
        context_query = " ".join(part for part in [base_query, jd_text, user_desc] if part).strip()
        return RetrievalPlan(
            queries=[
                RetrievalQuery(
                    query=base_query,
                    skill_filter=skill_filter or "",
                    category_filter=category_filter or "",
                    reason="Base skill query.",
                ),
                RetrievalQuery(
                    query=context_query or base_query,
                    skill_filter="",
                    category_filter=category_filter or "",
                    reason="Broader context query.",
                ),
            ]
        )

    def clean_plan(
        self,
        plan: RetrievalPlan,
        original_query: str,
        skill_filter: Optional[str],
        category_filter: Optional[str],
    ) -> RetrievalPlan:
        cleaned: list[RetrievalQuery] = [
            RetrievalQuery(
                query=original_query.replace("_", " "),
                skill_filter=skill_filter or "",
                category_filter=category_filter or "",
                reason="Original skill query.",
            )
        ]
        seen_queries = {normalize_skill(original_query)}

        for item in plan.queries:
            query_text = item.query.strip()
            if not query_text:
                continue
            query_key = normalize_skill(query_text)
            if query_key in seen_queries:
                continue
            seen_queries.add(query_key)
            cleaned.append(
                RetrievalQuery(
                    query=query_text,
                    skill_filter=normalize_skill(item.skill_filter) if item.skill_filter else "",
                    category_filter=item.category_filter.strip() if item.category_filter else "",
                    reason=item.reason.strip(),
                )
            )
            if len(cleaned) >= 4:
                break

        return RetrievalPlan(queries=cleaned)

    def run_retrieval_plan(
        self,
        plan: RetrievalPlan,
        topk: int,
        fetch: int,
        lambda_: float,
        type_filter: Optional[str],
        default_skill_filter: Optional[str],
        default_category_filter: Optional[str],
        difficulty_distribution: Optional[Dict[str, float]],
        difficulty_filter: Optional[str],
        jd_text: str,
        user_desc: str,
        role_type: str,
    ) -> list[dict[str, Any]]:
        per_query_topk = max(3, math.ceil(topk / max(1, len(plan.queries))) + 2)
        candidates: list[dict[str, Any]] = []

        for item in plan.queries:
            results = self.retrieval_tool(
                query=item.query,
                topk=per_query_topk,
                fetch=fetch,
                lambda_=lambda_,
                type_filter=type_filter,
                skill_filter=item.skill_filter or default_skill_filter,
                category_filter=item.category_filter or default_category_filter,
                difficulty_distribution=difficulty_distribution,
                difficulty_filter=difficulty_filter,
                jd_text=jd_text,
                user_desc=user_desc,
                role_type=role_type,
            )
            candidates.extend(self.mark_agentic_results(results, item.reason or item.query))

        return candidates

    def retrieval_tool(
        self,
        query: str,
        topk: int,
        fetch: int,
        lambda_: float,
        type_filter: Optional[str],
        skill_filter: Optional[str],
        category_filter: Optional[str],
        difficulty_distribution: Optional[Dict[str, float]],
        difficulty_filter: Optional[str],
        jd_text: str,
        user_desc: str,
        role_type: str,
    ) -> list[dict[str, Any]]:
        return self.base_retriever.retrieve(
            query=query,
            topk=topk,
            fetch=fetch,
            lambda_=lambda_,
            type_filter=type_filter,
            skill_filter=skill_filter,
            difficulty_distribution=difficulty_distribution,
            category_filter=category_filter,
            difficulty_filter=difficulty_filter,
            jd_text=jd_text,
            user_desc=user_desc,
            role_type=role_type,
        )

    def check_coverage(
        self,
        candidates: list[dict[str, Any]],
        topk: int,
        skill_filter: Optional[str],
        query: str,
        jd_text: str,
        user_desc: str,
    ) -> CoverageReport:
        if len(self.merge_and_dedupe(candidates)) < topk:
            return CoverageReport(
                needs_retry=True,
                retry_query=self.broaden_query(query, jd_text, user_desc),
                retry_skill_filter="",
                reason="Not enough unique candidates.",
            )

        if skill_filter:
            normalized_skill = normalize_skill(skill_filter)
            direct_matches = [
                item
                for item in candidates
                if normalized_skill in {normalize_skill(str(skill)) for skill in item.get("taxonomy_skills", [])}
            ]
            if len(direct_matches) < min(2, topk):
                return CoverageReport(
                    needs_retry=True,
                    retry_query=self.broaden_query(query, jd_text, user_desc),
                    retry_skill_filter="",
                    reason="Strict skill search had low direct-match coverage.",
                )

        return CoverageReport(needs_retry=False, reason="Coverage looks sufficient.")

    def broaden_query(self, query: str, jd_text: str, user_desc: str) -> str:
        return " ".join(
            part.strip()
            for part in [
                query.replace("_", " "),
                jd_text,
                user_desc,
                "related interview questions concepts skills practice",
            ]
            if part and part.strip()
        )

    def mark_agentic_results(
        self,
        results: list[dict[str, Any]],
        agentic_reason: str,
    ) -> list[dict[str, Any]]:
        marked = []
        for result in results:
            copied = dict(result)
            copied["agentic_reason"] = agentic_reason
            existing_reason = copied.get("selection_reason", "")
            copied["selection_reason"] = f"{existing_reason}; agentic step: {agentic_reason}".strip("; ")
            marked.append(copied)
        return marked

    def merge_and_dedupe(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        best_by_id: dict[str, dict[str, Any]] = {}

        for result in results:
            result_id = str(result.get("id") or result.get("question_id") or result.get("title"))
            current = best_by_id.get(result_id)
            if current is None:
                best_by_id[result_id] = result
                continue

            current_score = float(current.get("adjusted_score", 0.0) or 0.0)
            new_score = float(result.get("adjusted_score", 0.0) or 0.0)
            if new_score > current_score:
                best_by_id[result_id] = result

        return sorted(
            best_by_id.values(),
            key=lambda item: float(item.get("adjusted_score", 0.0) or 0.0),
            reverse=True,
        )


def init_agentic_retriever(llm: Any | None = None) -> AgenticQuestionRetriever:
    base_retriever = init_retriever()
    return AgenticQuestionRetriever(
        base_retriever=base_retriever,
        llm=llm or get_chat_model(temperature=0.1),
    )
