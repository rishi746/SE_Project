"""Small CPU BM25 retriever over downloaded text, with source-level attribution."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import heapq
import json
import math
from pathlib import Path
import re

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data/corpus/chunks.jsonl"
STOPWORDS = set("a an the is are be for to of in on and or with as by from that this it its at will should must application system design software network networking build create want need please related".split())


def tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9][a-z0-9_-]+", text.lower()) if word not in STOPWORDS]


class CorpusIndex:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, row in enumerate(rows):
            counts = Counter(tokens(row.get("title", "") + " " + row["text"]))
            self.lengths.append(sum(counts.values()))
            for token, count in counts.items():
                self.postings[token].append((index, count))
        self.average_length = sum(self.lengths) / max(len(self.rows), 1)

    def search(self, query: str, limit: int = 3, split: str | None = "train") -> list[dict]:
        if limit < 1:
            return []
        scores: dict[int, float] = defaultdict(float)
        total = len(self.rows)
        for term in set(tokens(query)):
            postings = self.postings.get(term, [])
            inverse_frequency = math.log(1 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                if split is not None and self.rows[index].get("split") != split:
                    continue
                length_norm = self.lengths[index] / max(self.average_length, 1)
                scores[index] += inverse_frequency * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length_norm))
        results, documents = [], set()
        for index in heapq.nlargest(len(scores), scores, key=scores.get):
            row = self.rows[index]
            document = row.get("document_id", row["id"])
            if document in documents:
                continue
            documents.add(document)
            results.append({**row, "retrieval_score": round(scores[index], 5)})
            if len(results) == limit:
                break
        return results


@lru_cache(maxsize=2)
def _load_index(path: str, modified_ns: int) -> CorpusIndex:
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return CorpusIndex(rows)


def retrieve_context(query: str, corpus_path: Path | None = None, limit: int = 3,
                     max_chars: int = 7500, split: str | None = "train") -> str:
    path = Path(corpus_path or DEFAULT_CORPUS)
    if not path.is_file():
        return "No downloaded reference corpus is available. Do not invent source citations."
    results = _load_index(str(path.resolve()), path.stat().st_mtime_ns).search(query, limit, split)
    if not results:
        return "No matching reference excerpts were retrieved. State unsupported assumptions explicitly."
    parts = []
    for row in results:
        header = f"[{row['id']}] {row.get('title', '')}\nSource: {row.get('source_url', '')}\nLicense: {row.get('license', '')}\n"
        remaining = max_chars - sum(len(p) for p in parts) - len(header) - 100
        if remaining < 200:
            break
        text = row['text']
        if len(text) > remaining:
            text = text[:remaining].rsplit(' ', 1)[0] + " [excerpt ends]"
        parts.append(header + text + "\n")
    return "\n".join(parts)
