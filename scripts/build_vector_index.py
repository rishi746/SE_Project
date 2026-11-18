from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "configs" / "source_catalog.json"
INDEX_DIR = ROOT / "data" / "vector_index"


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    texts = [
        f"{item['title']} | {item['publisher']} | topics={', '.join(item['topics'])}. {item['use_in_dataset']} URL: {item['url']}"
        for item in catalog
    ]

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    matrix = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(INDEX_DIR / "networking_sources.faiss"))
    (INDEX_DIR / "metadata.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"indexed_sources={len(catalog)}")
    print(f"index_path={INDEX_DIR / 'networking_sources.faiss'}")


if __name__ == "__main__":
    main()
