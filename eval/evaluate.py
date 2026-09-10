import json
from pathlib import Path

from app.config import DATA_DIR
from app.rag import build_index, retrieve


def load_eval_set(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def evaluate(eval_path: str, data_dir: Path):
    index, chunks, metadata = build_index(data_dir)
    eval_set = load_eval_set(eval_path)

    results = []
    for item in eval_set:
        retrieved = retrieve(item["question"], index, chunks, metadata)
        retrieved_text = " ".join(r["text"] for r in retrieved).lower()

        found = [kw for kw in item["expected_keywords"] if kw.lower() in retrieved_text]
        hit = len(found) == len(item["expected_keywords"])

        results.append({
            "question": item["question"],
            "expected_keywords": item["expected_keywords"],
            "keywords_found": found,
            "hit": hit,
        })

    accuracy = sum(r["hit"] for r in results) / len(results)
    return results, accuracy


if __name__ == "__main__":
    results, accuracy = evaluate("eval/eval_set.json", DATA_DIR)
    for r in results:
        status = "PASS" if r["hit"] else "FAIL"
        print(f"[{status}] {r['question']} — found: {r['keywords_found']}")
    print(f"\nRetrieval accuracy: {accuracy:.0%} ({sum(r['hit'] for r in results)}/{len(results)})")