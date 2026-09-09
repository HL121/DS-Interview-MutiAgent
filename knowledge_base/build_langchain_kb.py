from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data/questions_normalized.jsonl"
QDRANT_PATH = PROJECT_ROOT / "knowledge_base/vectorstores/questions_qdrant"
DEFAULT_COLLECTION_NAME = "ds_interview_questions"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
FILTER_INDEX_FIELDS = [
    "metadata.type",
    "metadata.category",
    "metadata.difficulty",
    "metadata.taxonomy_skills",
]


def qdrant_collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)


def create_qdrant_client() -> QdrantClient:
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def build_page_content(record: dict[str, Any]) -> str:
    skills = ", ".join(record.get("taxonomy_skills", []))

    return f"""
Title: {record["title"]}
Type: {record["type"]}
Category: {record["category"]}
Difficulty: {record["difficulty"]}
Skills: {skills}

Question:
{record["question"]}

Answer:
{record["answer"]}
""".strip()

def to_document(record: dict[str, Any]) -> Document:
    metadata = {
        "id": record["id"],
        "type": record["type"],
        "category": record["category"],
        "difficulty": record["difficulty"],
        "taxonomy_skills": record["taxonomy_skills"],
        "source": record["source"],
        "url": record["url"],
        "title": record["title"],
    }

    return Document(
        page_content=build_page_content(record),
        metadata=metadata,
    )

def deterministic_point_id(document: Document) -> str:
    return str(uuid5(NAMESPACE_URL, document.metadata["id"]))


def create_filter_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    for field_name in FILTER_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )


def build_qdrant_vectorstore(
    documents: list[Document],
    embeddings: OpenAIEmbeddings,
) -> QdrantVectorStore:
    client = create_qdrant_client()
    collection_name = qdrant_collection_name()

    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )
    create_filter_payload_indexes(client, collection_name)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    vectorstore.add_documents(
        documents=documents,
        ids=[deterministic_point_id(document) for document in documents],
    )
    return vectorstore

def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not visible. Create a .env file in the project root "
            "or export OPENAI_API_KEY in your shell before running this script."
        )

    records = load_jsonl(INPUT_PATH)
    documents = [to_document(record) for record in records]

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    build_qdrant_vectorstore(documents, embeddings)

    print(f"Loaded records: {len(records)}")
    print(f"Built documents: {len(documents)}")
    if os.getenv("QDRANT_URL"):
        print(f"Saved Qdrant vector store to cloud URL: {os.getenv('QDRANT_URL')}")
    else:
        print(f"Saved Qdrant vector store to: {QDRANT_PATH}")
    print(f"Qdrant collection: {qdrant_collection_name()}")


if __name__ == "__main__":
    main()
