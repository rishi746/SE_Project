from __future__ import annotations

import json
import re
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "configs" / "source_catalog.json"
RAW_DIR = ROOT / "data" / "raw"


def extension_for(url: str, content_type: str | None) -> str:
    lowered = url.lower()
    if lowered.endswith(".pdf") or (content_type and "pdf" in content_type):
        return ".pdf"
    if lowered.endswith(".txt") or (content_type and "text/plain" in content_type):
        return ".txt"
    return ".html"


def safe_name(source_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    manifest = []

    for item in catalog:
        request = urllib.request.Request(
            item["url"],
            headers={"User-Agent": "networking-requirements-research-prototype/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
                content_type = response.headers.get("Content-Type")
                suffix = extension_for(response.geturl(), content_type)
                path = RAW_DIR / f"{safe_name(item['source_id'])}{suffix}"
                path.write_bytes(content)
                manifest.append(
                    {
                        "source_id": item["source_id"],
                        "url": item["url"],
                        "resolved_url": response.geturl(),
                        "content_type": content_type,
                        "bytes": len(content),
                        "path": str(path.relative_to(ROOT)),
                        "status": "fetched",
                    }
                )
                print(f"fetched {item['source_id']} -> {path}")
        except (HTTPError, URLError, TimeoutError) as exc:
            manifest.append(
                {
                    "source_id": item["source_id"],
                    "url": item["url"],
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"failed {item['source_id']}: {exc}")

    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest={RAW_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
