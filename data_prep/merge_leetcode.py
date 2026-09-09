import json
from pathlib import Path

PANDAS_PATH = Path("data/pandas.json")
ALGO_PATH = Path("data/algorithms.json")
SQL_PATH = Path("data/sql.json")

OUT_PATH = Path("data/leetcode.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_record(record: dict, is_sql: bool):
    """
    Ensure all records share the same metadata schema.
    SQL has backup_url; pandas/algo do not.
    """
    metadata = record["metadata"]

    metadata.setdefault("url", "")
    metadata.setdefault("backup_url", "")

    # SQL keeps its backup_url, others get empty backup_url
    if not is_sql:
        metadata["backup_url"] = ""

    record["metadata"] = metadata
    return record


def main():
    pandas_data = load_json(PANDAS_PATH)
    algo_data = load_json(ALGO_PATH)
    sql_data = load_json(SQL_PATH)

    merged = []

    # normalize pandas
    for r in pandas_data:
        merged.append(normalize_record(r, is_sql=False))

    # normalize algorithm
    for r in algo_data:
        merged.append(normalize_record(r, is_sql=False))

    # normalize sql
    for r in sql_data:
        merged.append(normalize_record(r, is_sql=True))

    # save merged json
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Merged dataset saved → {OUT_PATH}")
    print(f"Total problems: {len(merged)}")


if __name__ == "__main__":
    main()
