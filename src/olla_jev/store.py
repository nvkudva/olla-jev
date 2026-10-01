"""Resolve model names to a pinned Hugging Face snapshot, download it, list and delete downloads."""

from __future__ import annotations

from dataclasses import dataclass, field

from huggingface_hub import HfApi, HFCacheInfo, snapshot_download
from huggingface_hub import scan_cache_dir as _scan_cache_dir
from huggingface_hub.errors import CacheNotFound, LocalEntryNotFoundError, RepositoryNotFoundError

from . import config
from .adapters import Family, detect
from .names import Ref, parse, pick_gguf


@dataclass
class Resolved:
    ref: Ref
    family: Family
    revision: str
    files: list[str]
    gguf: str | None = None  # the one .gguf file this name selects, for GGUF repos
    allow: list[str] | None = field(default=None)

    @property
    def name(self) -> str:
        return self.ref.name

    @property
    def repo_id(self) -> str:
        return self.ref.repo_id


def scan_cache_dir(cache_dir: str | None = None) -> HFCacheInfo | None:
    """The Hugging Face cache, or None before anything was ever downloaded to it."""
    try:
        return _scan_cache_dir(cache_dir)
    except CacheNotFound:
        return None


def pins() -> dict[str, str]:
    return config.load().get("pins", {})


def _pin(repo_id: str, sha: str, created: str | None) -> None:
    data = config.load()
    data.setdefault("pins", {})[repo_id] = sha
    if created:
        data.setdefault("released", {})[repo_id] = created
    config.save(data)


def released(repo_id: str) -> str | None:
    """The repo's creation date, reported as release_date like the hosted API does."""
    return config.load().get("released", {}).get(repo_id)


def _remote_files(repo_id: str, revision: str | None) -> tuple[str, str | None, list[str]]:
    try:
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=False)
    except RepositoryNotFoundError:
        raise LookupError(
            f"model {repo_id!r} not found on Hugging Face (private or gated repos need `hf auth login`)"
        ) from None
    if info.sha is None:
        raise LookupError(f"Hugging Face returned no commit for {repo_id!r}")
    created = info.created_at.date().isoformat() if info.created_at else None
    return info.sha, created, [s.rfilename for s in info.siblings or []]


def _local_files(repo_id: str, revision: str) -> list[str] | None:
    """The file list of a downloaded snapshot, so resolving works offline."""
    info = scan_cache_dir(config.models_dir())
    for repo in info.repos if info else ():
        if repo.repo_id == repo_id:
            for rev in repo.revisions:
                if rev.commit_hash == revision:
                    return [str(f.file_path.relative_to(rev.snapshot_path)) for f in rev.files]
    return None


def resolve(name: str, *, online: bool = True) -> Resolved:
    """Name -> family, pinned revision, files to fetch. The first resolve of a repo pins its commit.

    Offline (`online=False`) it uses only what is downloaded and raises LookupError otherwise.
    """
    ref = parse(name)
    revision = pins().get(ref.repo_id)
    files = _local_files(ref.repo_id, revision) if revision else None
    gguf = None
    if files is not None:
        try:
            gguf = _pick(ref, files)
        except ValueError:
            if not online:
                raise LookupError(f"{ref.name} is not downloaded; run: olla-jev pull {ref.name}") from None
            files = None  # a quant of a downloaded repo that is not on disk yet
    if files is None:
        if not online:
            raise LookupError(f"{ref.name} is not downloaded; run: olla-jev pull {ref.name}")
        sha, created, files = _remote_files(ref.repo_id, revision)
        if revision is None:
            revision = sha
            _pin(ref.repo_id, sha, created)
        gguf = _pick(ref, files)
    assert revision is not None  # pinned above, or taken from the pin
    family = detect(ref.repo_id, files)
    resolved = Resolved(ref, family, revision, files, gguf)
    resolved.allow = family.allow_patterns(resolved)
    return resolved


def _pick(ref: Ref, files: list[str]) -> str | None:
    if any(f.lower().endswith(".gguf") for f in files):
        return pick_gguf(files, ref.tag)
    if ref.tag:
        raise ValueError(f"{ref.repo_id} has no quantized files; drop ':{ref.tag}'")
    return None


def local_path(r: Resolved) -> str | None:
    try:
        return snapshot_download(
            r.repo_id, revision=r.revision, allow_patterns=r.allow, cache_dir=config.models_dir(), local_files_only=True
        )
    except LocalEntryNotFoundError:
        return None


def download(r: Resolved) -> str:
    return snapshot_download(r.repo_id, revision=r.revision, allow_patterns=r.allow, cache_dir=config.models_dir())


def downloaded() -> dict[str, tuple[int, float]]:
    """repo_id -> (bytes on disk, last modified) for pinned repos in the cache."""
    wanted = pins()
    info = scan_cache_dir(config.models_dir())
    return {
        r.repo_id: (r.size_on_disk, r.last_modified)
        for r in (info.repos if info else ())
        if r.repo_id in wanted and r.repo_type == "model"
    }


def delete(repo_id: str) -> int:
    """Remove every downloaded revision of `repo_id` and forget its pin. Returns bytes freed."""
    info = scan_cache_dir(config.models_dir())
    revisions = [
        rev.commit_hash for repo in (info.repos if info else ()) if repo.repo_id == repo_id for rev in repo.revisions
    ]
    freed = 0
    if info and revisions:
        strategy = info.delete_revisions(*revisions)
        freed = strategy.expected_freed_size
        strategy.execute()
    data = config.load()
    data.get("pins", {}).pop(repo_id, None)
    data.get("released", {}).pop(repo_id, None)
    data["trusted"] = [t for t in data.get("trusted", []) if not t.startswith(f"{repo_id}@")]
    config.save(data)
    return freed


def is_trusted(r: Resolved) -> bool:
    return not r.family.runs_repo_code or f"{r.repo_id}@{r.revision}" in config.load().get("trusted", [])


def trust(r: Resolved) -> None:
    data = config.load()
    key = f"{r.repo_id}@{r.revision}"
    if key not in data.setdefault("trusted", []):
        data["trusted"].append(key)
    config.save(data)
