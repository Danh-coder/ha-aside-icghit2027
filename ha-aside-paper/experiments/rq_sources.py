"""Immutable public-source snapshots. No task, answer or attack generation."""

import json
import shutil
import urllib.request
from pathlib import Path

from .common import ORIGINAL, PROJECT, ROOT, digest, file_hash, write_json

REPOS = {
    "iheval": "ytyz1307zzh/IHEval",
    "sysbench": "PKU-Baichuan-MLSystemLab/SysBench",
    "mtrag": "IBM/mt-rag-benchmark",
    "bipia": "microsoft/BIPIA",
}


def get_bytes(url):
    request = urllib.request.Request(url, headers={"User-Agent": "HA-ASIDE-public-benchmark/1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def selected(name, path):
    if path in {"README.md", "LICENSE", "LICENSE.md"}:
        return True
    if name == "iheval":
        return (path.startswith("benchmark/") and path.endswith("/input_data.json")) or (
            path.startswith("src/") and path.endswith(".py"))
    if name == "sysbench":
        return path in {"datas/system_benchmark_eval_datas.json", "utils.py",
                        "eval_system_bench.py", "eval_output.py", "models/gpt4o.py"}
    if name == "mtrag":
        return path in {"mtrag-human/generation_tasks/reference.jsonl",
                        "mtrag-human/conversations/conversations.json"} or (
            path.startswith("scripts/evaluation/") and path.endswith((".py", ".yaml", ".txt")))
    return path in {"benchmark/text_attack_test.json", "benchmark/email/test.jsonl", "benchmark/table/test.jsonl"}


def fetch_sources(destination, revisions=None, cache=None):
    """Resolve once, download by commit, and refuse reuse of modified snapshots."""
    destination = Path(destination)
    lock_path = destination / "sources.lock.json"
    if lock_path.exists():
        lock = verify_sources(destination)
        if revisions and any(lock["repositories"][k]["revision"] != v for k, v in revisions.items()):
            raise ValueError("Source pins differ; select a new snapshot directory")
        return lock
    destination.mkdir(parents=True, exist_ok=True)
    lock = {"schema_version": 1, "repositories": {}, "files": {}}
    cache_lock = verify_sources(cache) if cache else None
    for name, repo in REPOS.items():
        revision = (revisions or {}).get(name)
        if not revision:
            revision = json.loads(get_bytes(f"https://api.github.com/repos/{repo}/commits/HEAD"))["sha"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Public sources require immutable Git commit pins")
        tree = json.loads(get_bytes(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1"))
        if tree.get("truncated"):
            raise ValueError(f"GitHub tree truncated for {repo}")
        lock["repositories"][name] = {"repository": repo, "revision": revision}
        for item in sorted(tree["tree"], key=lambda x: x["path"]):
            path = item["path"]
            if item["type"] != "blob" or not selected(name, path):
                continue
            relative = f"{name}/{path}"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            cached = (Path(cache) / relative) if cache else None
            if (cached and relative in cache_lock["files"]
                    and cache_lock["repositories"][name]["revision"] == revision):
                raw = cached.read_bytes()
            else:
                raw = get_bytes(f"https://raw.githubusercontent.com/{repo}/{revision}/{path}")
            if target.exists() and target.read_bytes() != raw:
                raise ValueError(f"Partial snapshot contains changed content: {target}")
            target.write_bytes(raw)
            lock["files"][relative] = {"sha256": file_hash(target), "bytes": len(raw)}
    for name, source in {"sep.json": ORIGINAL / "data/SEP_dataset.json",
                         "gandalf.jsonl": PROJECT / "data/prepared/gandalf.jsonl"}.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required original benchmark source missing: {source}")
        target = destination / "local" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        lock["files"][f"local/{name}"] = {
            "sha256": file_hash(target), "bytes": target.stat().st_size,
            "origin": str(source.relative_to(ROOT)),
        }
    write_json(lock_path, lock)
    return verify_sources(destination)


def verify_sources(destination):
    destination = Path(destination)
    lock = json.loads((destination / "sources.lock.json").read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        path = destination / relative
        if not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Snapshot path escapes its root")
        if not path.is_file() or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Changed/missing public source: {relative}")
    return lock


def source_ref(lock, path, row, pointer, value):
    repo = path.split("/", 1)[0]
    return {"file": path, "row": row, "pointer": pointer,
            "file_sha256": lock["files"][path]["sha256"], "value_sha256": digest(value),
            "revision": lock["repositories"].get(repo, {}).get("revision", "local-vendored-source")}


def rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line, parse_constant=lambda value: None) for line in stream if line.strip()]
    # SysBench publishes NaN in non-text metadata. Preserve snapshot bytes and
    # normalize these missing metadata values to JSON null in prepared artifacts.
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: None)
    return data
