"""Download allowlisted public repositories, preserve licenses, align and deduplicate text.

CPU/stdlib only. No credentials, GPU, model API, or arbitrary URL crawling. Network
failures are recorded and cause a nonzero exit. Archives never extract paths.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import tarfile
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get(url: str, limit: int = 5_000_000) -> bytes:
    if not url.startswith(("https://api.github.com/", "https://raw.githubusercontent.com/", "https://codeload.github.com/")):
        raise ValueError("Only allowlisted public GitHub endpoints may be downloaded")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SMS-AC-corpus/1.0", "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=90) as response:
                result = response.read(limit + 1)
            if len(result) > limit:
                raise ValueError(f"Download exceeds configured byte limit: {url}")
            return result
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("unreachable")


def normalized(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"\A---\n.*?\n---\s*\n", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email omitted]", text)
    return re.sub(r"\n{4,}", "\n\n\n", text).strip()


def family(source: str, path: str, title: str) -> str:
    # All versions, translations, sections and KEP files share one family.
    if source == "kubernetes-keps":
        match = re.search(r"(?:/)(\d+)[-/]", path)
        if match:
            return f"{source}:kep-{match.group(1)}"
    path = re.sub(r"v?\d+\.\d+(?:\.\d+)?", "VERSION", path)
    return f"{source}:{path.rsplit('.', 1)[0]}"


def fetch_source(source: dict, out: Path, offline: bool) -> dict:
    folder = out / "raw" / source["id"]
    manifest_file = folder / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if all((folder / "documents" / f["local_name"]).exists() for f in manifest["files"]):
            return manifest
    if offline:
        raise FileNotFoundError(f"No complete cached source: {source['id']}")
    commit = json.loads(get(f"https://api.github.com/repos/{source['repo']}/commits/{source['ref']}"))["sha"]
    license_url = f"https://raw.githubusercontent.com/{source['repo']}/{commit}/{source['license_path']}"
    license_bytes = get(license_url)
    if source["license_marker"].lower() not in license_bytes.decode("utf-8").lower():
        raise ValueError(f"License check failed: {source['id']}")
    archive_url = f"https://codeload.github.com/{source['repo']}/tar.gz/{commit}"
    archive = get(archive_url, source.get("max_archive_mb", 100) * 1024 * 1024)
    (folder / "documents").mkdir(parents=True, exist_ok=True)
    (folder / "LICENSE").write_bytes(license_bytes)
    files = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile() or member.size > 2_000_000:
                continue
            path = member.name.split("/", 1)[-1]
            if path in ("NOTICE", "NOTICE.md", "ThirdPartyNotices", "LICENSE-CODE"):
                (folder / path).write_bytes(tar.extractfile(member).read())
            if not any(path.startswith(prefix) for prefix in source["include"]) or not any(path.endswith(ext) for ext in source["extensions"]):
                continue
            if any(p in path.lower() for p in ("third_party/", "vendor/", "node_modules/", "/images/")):
                continue
            raw = tar.extractfile(member).read()
            value = raw.decode("utf-8", errors="replace")
            if len(value.split()) < 100:
                continue
            local_name = digest(path)[:24] + ".md"
            (folder / "documents" / local_name).write_bytes(raw)
            files.append({"path": path, "local_name": local_name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    manifest = {**source, "commit": commit, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "license_url": license_url, "archive_url": archive_url, "archive_sha256": hashlib.sha256(archive).hexdigest(), "files": sorted(files, key=lambda x:x["path"])}
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def align(manifests: list[dict], out: Path, chunk_words: int = 600) -> dict:
    documents, seen_docs, seen_chunks = [], set(), set()
    count = Counter()
    with (out / "documents.jsonl").open("w", encoding="utf-8") as doc_file, (out / "chunks.jsonl").open("w", encoding="utf-8") as chunk_file:
        for manifest in sorted(manifests, key=lambda x:x["id"]):
            for item in manifest["files"]:
                raw = (out / "raw" / manifest["id"] / "documents" / item["local_name"]).read_text(encoding="utf-8", errors="replace")
                content = normalized(raw)
                doc_hash = digest(" ".join(content.lower().split()))
                if doc_hash in seen_docs:
                    count["duplicate_documents_removed"] += 1
                    continue
                seen_docs.add(doc_hash)
                title_match = re.search(r"^#\s+(.+)", content, re.M)
                title = title_match.group(1).strip() if title_match else Path(item["path"]).stem
                group = family(manifest["id"], item["path"], title)
                split_bucket = int(digest(group)[:8], 16) % 100
                split = "train" if split_bucket < 80 else "validation" if split_bucket < 90 else "test"
                row = {"id": doc_hash, "group_id": group, "split": split, "title": title, "text": content, "source_id": manifest["id"], "source_url": f"https://github.com/{manifest['repo']}/blob/{manifest['commit']}/{item['path']}", "source_path": item["path"], "source_commit": manifest["commit"], "license": manifest["license"], "attribution": manifest["attribution"], "license_url": manifest["license_url"], "review_status": "license_file_verified_content_unreviewed", "modifications": "UTF-8 decoding, front matter and comments removed, emails masked, whitespace normalized; raw original retained", "role": manifest["role"]}
                doc_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                count[f"documents_{manifest['id']}"] += 1
                count["documents"] += 1
                words = content.split()
                count["words"] += len(words)
                for offset in range(0, len(words), chunk_words - 80):
                    text = " ".join(words[offset:offset+chunk_words])
                    if len(text.split()) < 80:
                        continue
                    chunk_hash = digest(text.lower())
                    if chunk_hash in seen_chunks:
                        count["duplicate_chunks_removed"] += 1
                        continue
                    seen_chunks.add(chunk_hash)
                    chunk = {k:v for k,v in row.items() if k != "text"}
                    chunk.update(id=chunk_hash, document_id=doc_hash, text=text, word_offset=offset)
                    chunk_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    count["chunks"] += 1
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "counts": dict(count), "sources": [{k:m[k] for k in ("id","commit","license","attribution","license_url")} for m in manifests], "limitations": ["Reference documents are not supervised target answers.", "Exact text deduplication and source/version-family grouping; semantic near-duplicate audit still needed before benchmark use.", "Technical applicability and superseded proposals need expert review."]}
    (out / "corpus_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    attribution = "# Source attribution\n\nNormalization modifies formatting and masks emails; original documents and license/NOTICE files are retained under raw/.\n\n" + "\n\n".join(f"- {m['attribution']}. [{m['repo']}](https://github.com/{m['repo']}/tree/{m['commit']}). License: [{m['license']}]({m['license_url']})." for m in manifests)
    (out / "ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "configs/corpus_sources.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/corpus")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--source", action="append", help="Optional source ids; default all")
    args = parser.parse_args()
    sources = json.loads(args.catalog.read_text(encoding="utf-8"))["sources"]
    if args.source:
        sources = [s for s in sources if s["id"] in args.source]
    args.output.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(fetch_source, source, args.output, args.offline): source for source in sources}
        for job in concurrent.futures.as_completed(jobs):
            try:
                manifest = job.result()
                results.append(manifest)
                print(f"{manifest['id']}: {len(manifest['files'])} cached documents", flush=True)
            except Exception as exc:
                errors.append({"source": jobs[job]["id"], "error": str(exc)})
                print(f"ERROR {jobs[job]['id']}: {exc}", flush=True)
    summary = align(results, args.output)
    summary["errors"] = errors
    (args.output / "corpus_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
