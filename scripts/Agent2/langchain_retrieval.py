from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi

from knowledge_base.build_langchain_kb import (
    DEFAULT_COLLECTION_NAME,
    build_page_content,
    create_qdrant_client,
    qdrant_collection_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = PROJECT_ROOT / "data/questions_normalized.jsonl"
QDRANT_PATH = PROJECT_ROOT / "knowledge_base/vectorstores/questions_qdrant"
COLLECTION_NAME = DEFAULT_COLLECTION_NAME
EMBEDDING_MODEL = "text-embedding-3-small"
DENSE_WEIGHT = 0.8
KEYWORD_WEIGHT = 0.2


def normalize_skill(value: str) -> str:
    """Normalize user/planner skill names so they match dataset taxonomy tags."""
    skill = value.replace("\xa0", " ").lower().strip()
    skill = skill.replace("-", "_")
    skill = re.sub(r"[^a-z0-9_]+", "_", skill)
    skill = re.sub(r"_+", "_", skill)
    return skill.strip("_")


def build_rich_query(
    target_skill: str,
    jd_text: str = "",
    user_desc: str = "",
    role_type: str = "",
    desired_type: str = "",
    target_difficulty: str = "",
) -> str:
    """Combine planner context, JD text, and user context into one retrieval query."""
    parts = [
        f"Target skill: {target_skill.replace('_', ' ')}",
    ]
    if role_type:
        parts.append(f"Role type: {role_type}")
    if desired_type:
        parts.append(f"Question type: {desired_type}")
    if target_difficulty:
        parts.append(f"Target difficulty: {target_difficulty}")
    if jd_text:
        parts.append(f"Job description context: {jd_text}")
    if user_desc:
        parts.append(f"Candidate context and weak areas: {user_desc}")
    return "\n".join(parts)


# in Qdrant, every stored question has metadata fields like "type", "category", "difficulty", and "taxonomy_skills"
# we can use these fields to filter questions during retrieval
def metadata_condition(field: str, value: str) -> FieldCondition:
    """Create one exact Qdrant metadata condition, such as metadata.category == SQL."""
    return FieldCondition(
        key=f"metadata.{field}",
        match=MatchValue(value=value),
    )


def build_qdrant_filter(
    type_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    skill_filter: Optional[str] = None,
    difficulty_filter: Optional[str] = None,
) -> Optional[Filter]:
    """Build the hard metadata filter used before Qdrant vector search ranks results."""
    must: list[FieldCondition] = []

    if type_filter:
        must.append(metadata_condition("type", type_filter))
    if category_filter:
        must.append(metadata_condition("category", category_filter))
    if difficulty_filter:
        must.append(metadata_condition("difficulty", difficulty_filter))
    if skill_filter:
        must.append(metadata_condition("taxonomy_skills", normalize_skill(skill_filter)))

    if not must:
        return None
    return Filter(must=must)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load normalized question records from a JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_to_document(record: dict[str, Any]) -> Document:
    """Convert one normalized question record into a LangChain Document for BM25."""
    return Document(
        page_content=build_page_content(record),
        metadata={
            "id": record["id"],
            "type": record["type"],
            "category": record["category"],
            "difficulty": record["difficulty"],
            "taxonomy_skills": record["taxonomy_skills"],
            "source": record["source"],
            "url": record["url"],
            "title": record["title"],
        },
    )


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens for BM25 keyword search."""
    return re.findall(r"[a-z0-9_]+", text.lower().replace("\xa0", " "))


class BM25QuestionIndex:
    """Small in-memory BM25 index over the normalized question documents."""

    def __init__(self, documents: list[Document]) -> None:
        """Pre-tokenize all documents and initialize the BM25 scorer."""
        self.documents = documents
        self.bm25 = BM25Okapi([tokenize(document.page_content) for document in documents])

    def search(
        self,
        query: str,
        topk: int,
        type_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        skill_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None,
    ) -> list[tuple[Document, float]]:
        """Return keyword-ranked documents after applying the same metadata filters."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        allowed = self._allowed_indexes(
            type_filter=type_filter,
            category_filter=category_filter,
            skill_filter=skill_filter,
            difficulty_filter=difficulty_filter,
        )
        ranked = sorted(
            ((idx, float(scores[idx])) for idx in allowed if scores[idx] > 0),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(self.documents[idx], score) for idx, score in ranked[:topk]]

    def _allowed_indexes(
        self,
        type_filter: Optional[str],
        category_filter: Optional[str],
        skill_filter: Optional[str],
        difficulty_filter: Optional[str],
    ) -> list[int]:
        """Find BM25 document indexes that satisfy exact metadata constraints."""
        allowed = []
        normalized_skill = normalize_skill(skill_filter) if skill_filter else None
        for idx, document in enumerate(self.documents):
            metadata = document.metadata
            if type_filter and metadata.get("type") != type_filter:
                continue
            if category_filter and metadata.get("category") != category_filter:
                continue
            if difficulty_filter and metadata.get("difficulty") != difficulty_filter:
                continue
            if normalized_skill and normalized_skill not in metadata.get("taxonomy_skills", []):
                continue
            allowed.append(idx)
        return allowed


class QdrantQuestionRetriever:
    """Hybrid question retriever using Qdrant dense search plus BM25 keyword search."""

    def __init__(
        self,
        vectorstore: QdrantVectorStore,
        bm25_index: BM25QuestionIndex,
    ) -> None:
        """Store the dense vector store and keyword index used by retrieval."""
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index

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
        """Run the full retrieval pipeline and return planner-compatible candidates."""
        target_difficulty = self._preferred_difficulty(difficulty_distribution)
        rich_query = build_rich_query(
            target_skill=query,
            jd_text=jd_text,
            user_desc=user_desc,
            role_type=role_type,
            desired_type=type_filter or "",
            target_difficulty=target_difficulty,
        )

        raw_results = self._hybrid_search(
            query=rich_query,
            topk=topk,
            fetch=fetch,
            type_filter=type_filter,
            category_filter=category_filter,
            skill_filter=skill_filter,
            difficulty_filter=difficulty_filter,
        )
        reranked = self._rerank_results(
            results=raw_results,
            skill_filter=skill_filter,
            difficulty_distribution=difficulty_distribution,
        )
        return [self._format_result(doc, score, adjusted_score, skill_filter) for doc, score, adjusted_score in reranked[:topk]]

    def _search_with_fallbacks(
        self,
        query: str,
        topk: int,
        fetch: int,
        type_filter: Optional[str],
        category_filter: Optional[str],
        skill_filter: Optional[str],
        difficulty_filter: Optional[str],
    ) -> list[tuple[Document, float]]:
        """Run Qdrant vector search, relaxing metadata filters if no results are found."""
        search_k = max(fetch, topk)
        filters = [
            build_qdrant_filter(type_filter, category_filter, skill_filter, difficulty_filter),
            build_qdrant_filter(type_filter, category_filter, None, difficulty_filter),
            build_qdrant_filter(type_filter, None, None, difficulty_filter),
            None,
        ]

        seen_filter_reprs = set()
        for qdrant_filter in filters:
            filter_repr = repr(qdrant_filter)
            if filter_repr in seen_filter_reprs:
                continue
            seen_filter_reprs.add(filter_repr)

            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=search_k,
                filter=qdrant_filter,
            )
            if results:
                return results
        return []

    def _hybrid_search(
        self,
        query: str,
        topk: int,
        fetch: int,
        type_filter: Optional[str],
        category_filter: Optional[str],
        skill_filter: Optional[str],
        difficulty_filter: Optional[str],
    ) -> list[tuple[Document, float]]:
        """Combine Qdrant semantic scores and BM25 keyword scores into one ranking."""
        dense_results = self._search_with_fallbacks(
            query=query,
            topk=topk,
            fetch=fetch,
            type_filter=type_filter,
            category_filter=category_filter,
            skill_filter=skill_filter,
            difficulty_filter=difficulty_filter,
        )
        keyword_results = self.bm25_index.search(
            query=query,
            topk=max(fetch, topk),
            type_filter=type_filter,
            category_filter=category_filter,
            skill_filter=skill_filter,
            difficulty_filter=difficulty_filter,
        )

        dense_by_id = {doc.metadata["id"]: (doc, score) for doc, score in dense_results}
        keyword_by_id = {doc.metadata["id"]: (doc, score) for doc, score in keyword_results}

        max_dense = max((score for _, score in dense_results), default=1.0) or 1.0
        max_keyword = max((score for _, score in keyword_results), default=1.0) or 1.0

        combined = []
        for question_id in set(dense_by_id) | set(keyword_by_id):
            dense_doc_score = dense_by_id.get(question_id)
            keyword_doc_score = keyword_by_id.get(question_id)
            doc = dense_doc_score[0] if dense_doc_score else keyword_doc_score[0]
            dense_score = dense_doc_score[1] / max_dense if dense_doc_score else 0.0
            keyword_score = keyword_doc_score[1] / max_keyword if keyword_doc_score else 0.0
            combined_score = DENSE_WEIGHT * dense_score + KEYWORD_WEIGHT * keyword_score
            combined.append((doc, combined_score))

        combined.sort(key=lambda item: item[1], reverse=True)
        return combined

    def _rerank_results(
        self,
        results: list[tuple[Document, float]],
        skill_filter: Optional[str],
        difficulty_distribution: Optional[Dict[str, float]],
    ) -> list[tuple[Document, float, float]]:
        """Apply deterministic boosts for difficulty quota fit and direct skill match."""
        reranked = []
        for doc, score in results:
            metadata = doc.metadata
            adjusted_score = float(score)

            difficulty = metadata.get("difficulty")
            if difficulty_distribution and difficulty in difficulty_distribution:
                adjusted_score *= 0.75 + float(difficulty_distribution[difficulty])

            if skill_filter and normalize_skill(skill_filter) in metadata.get("taxonomy_skills", []):
                adjusted_score *= 1.15

            reranked.append((doc, float(score), adjusted_score))

        reranked.sort(key=lambda item: item[2], reverse=True)
        return self._dedupe_by_title(reranked)

    def _dedupe_by_title(
        self,
        results: list[tuple[Document, float, float]],
    ) -> list[tuple[Document, float, float]]:
        """Remove duplicate or near-duplicate results that share the same title."""
        deduped = []
        seen_titles = set()
        for doc, score, adjusted_score in results:
            title_key = normalize_skill(str(doc.metadata.get("title", "")))
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            deduped.append((doc, score, adjusted_score))
        return deduped

    def _format_result(
        self,
        doc: Document,
        retrieval_score: float,
        adjusted_score: float,
        skill_filter: Optional[str],
    ) -> Dict[str, Any]:
        """Convert a LangChain Document into the dict shape expected by the planner."""
        metadata = dict(doc.metadata)
        skills = metadata.get("taxonomy_skills", [])
        if isinstance(skills, str):
            skills = [skills]

        question_id = metadata.get("id")
        reason = self._selection_reason(metadata, retrieval_score, adjusted_score, skill_filter)

        return {
            "question_id": question_id,
            "id": question_id,
            "title": metadata.get("title", ""),
            "type": metadata.get("type", ""),
            "category": metadata.get("category", ""),
            "difficulty": metadata.get("difficulty", ""),
            "taxonomy_skills": skills,
            "retrieval_score": retrieval_score,
            "adjusted_score": adjusted_score,
            "selection_reason": reason,
            "score": adjusted_score,
            "url": metadata.get("url", ""),
            "backup_url": "",
            "preview": doc.page_content[:220].replace("\n", " "),
            "metadata": metadata,
            "data": {
                "selection_reason": reason,
                "page_content": doc.page_content,
            },
        }

    def _selection_reason(
        self,
        metadata: dict[str, Any],
        retrieval_score: float,
        adjusted_score: float,
        skill_filter: Optional[str],
    ) -> str:
        """Create a short human-readable explanation for why the question was selected."""
        reason_parts = [
            f"Hybrid match score {retrieval_score:.3f}",
            f"{metadata.get('difficulty', 'unknown')} difficulty",
            f"{metadata.get('category', 'unknown')} category",
        ]
        if skill_filter and normalize_skill(skill_filter) in metadata.get("taxonomy_skills", []):
            reason_parts.append(f"directly matches skill {normalize_skill(skill_filter)}")
        elif metadata.get("taxonomy_skills"):
            reason_parts.append(f"related skills: {', '.join(metadata['taxonomy_skills'][:3])}")
        if adjusted_score != retrieval_score:
            reason_parts.append(f"reranked score {adjusted_score:.3f} after quota adjustments")
        return "; ".join(reason_parts)

    def _preferred_difficulty(
        self,
        difficulty_distribution: Optional[Dict[str, float]],
    ) -> str:
        """Pick the most requested difficulty as a soft hint for the rich query."""
        if not difficulty_distribution:
            return ""
        return max(difficulty_distribution.items(), key=lambda item: item[1])[0]


def init_retriever(
    qdrant_path: Path = QDRANT_PATH,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL,
) -> QdrantQuestionRetriever:
    """Load environment, Qdrant vector store, embeddings, and BM25 index."""
    load_dotenv(PROJECT_ROOT / ".env")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME") or collection_name or qdrant_collection_name()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not visible. Create .env in the project root or export it."
        )
    if not os.getenv("QDRANT_URL") and not qdrant_path.exists():
        raise RuntimeError(
            f"Qdrant knowledge base not found at {qdrant_path}. "
            "Run `python knowledge_base/build_langchain_kb.py` first."
        )

    client = create_qdrant_client()
    if not client.collection_exists(collection_name=collection_name):
        location = os.getenv("QDRANT_URL") or str(qdrant_path)
        raise RuntimeError(
            f"Qdrant collection `{collection_name}` was not found in {location}. "
            "Run `python knowledge_base/build_langchain_kb.py` first."
        )

    embeddings = OpenAIEmbeddings(model=model_name)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    documents = [record_to_document(record) for record in load_jsonl(QUESTIONS_PATH)]
    bm25_index = BM25QuestionIndex(documents=documents)
    return QdrantQuestionRetriever(vectorstore=vectorstore, bm25_index=bm25_index)


def main() -> None:
    """CLI entry point for manually testing retrieval from the terminal."""
    parser = argparse.ArgumentParser(description="Test Qdrant retrieval.")
    parser.add_argument("query", nargs="?", default="SQL window functions for product analytics")
    parser.add_argument("--skill", default="")
    parser.add_argument("--type", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--difficulty", default="")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    retriever = init_retriever()
    results = retriever.retrieve(
        query=args.query,
        topk=args.topk,
        skill_filter=args.skill or None,
        type_filter=args.type or None,
        category_filter=args.category or None,
        difficulty_filter=args.difficulty or None,
    )

    for idx, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"{idx}. {result['title']} ({result['id']})")
        print(f"type/category/difficulty: {result['type']} / {result['category']} / {result['difficulty']}")
        print(f"skills: {', '.join(result['taxonomy_skills'])}")
        print(f"score: {result['retrieval_score']:.3f} adjusted: {result['adjusted_score']:.3f}")
        print(f"reason: {result['selection_reason']}")
        if result["url"]:
            print(f"url: {result['url']}")


if __name__ == "__main__":
    main()
