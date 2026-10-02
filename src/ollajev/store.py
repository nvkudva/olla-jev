"""Resolve model names to a pinned Hugging Face snapshot, download it, list and delete downloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import CachedRevisionInfo, HfApi, HFCacheInfo, snapshot_download
from huggingface_hub import scan_cache_dir as _scan_cache_dir
from huggingface_hub.errors import CacheNotFound, LocalEntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.utils import filter_repo_objects

from . import config
from .adapters import Family, detect
from .names import Ref, parse, pick_gguf, quant_of


@dataclass
class Resolved:
    ref: Ref
    family: Family
    revision: str
    files: list[str]
    gguf: str | None = None  # the one .gguf file this name selects, for GGUF repos
    allow: list[str] | None = field(default=None)
    created: str | None = None  # the repo's creation date, recorded as release_date once it is downloaded

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
    with config.edit() as data:
        data.setdefault("pins", {})[repo_id] = sha
        if created:
            data.setdefault("released", {})[repo_id] = created


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


def snapshot(repo_id: str, revision: str) -> CachedRevisionInfo | None:
    """The downloaded snapshot of a repo at a commit, or None."""
    info = scan_cache_dir(config.models_dir())
    for repo in info.repos if info else ():
        if repo.repo_id == repo_id:
            for rev in repo.revisions:
                if rev.commit_hash == revision:
                    return rev
    return None


def _local_files(repo_id: str, revision: str) -> list[str] | None:
    """The file list of a downloaded snapshot, so resolving works offline."""
    rev = snapshot(repo_id, revision)
    return [str(f.file_path.relative_to(rev.snapshot_path)) for f in rev.files] if rev else None


def resolve(name: str, *, online: bool = True) -> Resolved:
    """Name -> family, pinned revision, files to fetch. The first resolve of a repo pins its commit.

    Offline (`online=False`) it uses only what is downloaded and raises LookupError otherwise.
    """
    ref = parse(name)
    created = None
    revision = pins().get(ref.repo_id)
    files = _local_files(ref.repo_id, revision) if revision else None
    gguf = None
    if files is not None:
        try:
            gguf = _pick(ref, files)
        except ValueError:
            if not online:
                raise LookupError(f"{ref.name} is not downloaded; run: ollajev pull {ref.name}") from None
            files = None  # a quant of a downloaded repo that is not on disk yet
    if files is None:
        if not online:
            raise LookupError(f"{ref.name} is not downloaded; run: ollajev pull {ref.name}")
        sha, created, files = _remote_files(ref.repo_id, revision)
        if revision is None:
            revision = sha  # pinned by download(), once the weights are on disk
        gguf = _pick(ref, files)
    if revision is None:  # unreachable: set from the pin or from the remote lookup above
        raise LookupError(f"no revision for {ref.name}")
    family = detect(ref.repo_id, files)
    resolved = Resolved(ref, family, revision, files, gguf, created=created)
    resolved.allow = family.allow_patterns(resolved)
    return resolved


@dataclass(frozen=True)
class Hit:
    """A Hugging Face repo found by `search`. family is None when no adapter runs it."""

    repo_id: str
    downloads: int
    family: str | None


@dataclass(frozen=True)
class Variant:
    """One downloadable form of a repo: a GGUF quant, or the full weights. size is the download in bytes."""

    name: str  # what `pull` takes
    label: str
    size: int


def _family(repo_id: str, files: list[str]) -> Family | None:
    try:
        return detect(repo_id, files)
    except LookupError:
        return None


def search(query: str, limit: int = 40) -> list[Hit]:
    """Repos matching every word of `query`, most downloaded first. A repo name or URL finds that repo."""
    api = HfApi()
    try:
        info = api.model_info(parse(query).repo_id)
    except (ValueError, RepositoryNotFoundError):
        pass
    else:
        files = [s.rfilename for s in info.siblings or []]
        family = _family(info.id, files)
        return [Hit(info.id, info.downloads or 0, family.name if family else None)]
    words = query.lower().split()
    if not words:
        return []
    found = api.list_models(search=max(words, key=len), sort="downloads", limit=200, expand=["siblings", "downloads"])
    hits = []
    for m in found:
        if all(w in m.id.lower() for w in words):
            family = _family(m.id, [s.rfilename for s in m.siblings or []])
            hits.append(Hit(m.id, m.downloads or 0, family.name if family else None))
            if len(hits) == limit:
                break
    return hits


def variants(repo_id: str) -> list[Variant]:
    """A repo's GGUF quants, or its full weights, with download sizes, smallest first."""
    info = HfApi().model_info(repo_id, files_metadata=True)
    return _variants(repo_id, info.sha or "", {s.rfilename: s.size or 0 for s in info.siblings or []})


def _variants(repo_id: str, sha: str, sizes: dict[str, int]) -> list[Variant]:
    files = list(sizes)
    family = _family(repo_id, files)
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    quants = [quant_of(f) for f in ggufs]
    tags = [q if q and quants.count(q) == 1 else f.rsplit("/", 1)[-1] for f, q in zip(ggufs, quants, strict=True)]
    found = []
    for tag, gguf in zip(tags, ggufs, strict=True) if ggufs else [(None, None)]:
        ref = Ref(repo_id, tag)
        if family:
            allow = family.allow_patterns(Resolved(ref, family, sha, files, gguf))
            size = sum(sizes[f] for f in filter_repo_objects(files, allow_patterns=allow))
        else:
            size = sizes[gguf] if gguf else sum(sizes.values())
        found.append(Variant(ref.name, tag or "full weights", size))
    return sorted(found, key=lambda v: v.size)


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
    """Fetch the snapshot, then pin the repo to this commit if it has no pin yet."""
    path = snapshot_download(r.repo_id, revision=r.revision, allow_patterns=r.allow, cache_dir=config.models_dir())
    if r.repo_id not in pins():
        _pin(r.repo_id, r.revision, r.created)
    return path


def downloaded() -> dict[str, tuple[int, float]]:
    """repo_id -> (bytes on disk, last modified) for pinned repos in the cache."""
    wanted = pins()
    info = scan_cache_dir(config.models_dir())
    return {
        r.repo_id: (r.size_on_disk, r.last_modified)
        for r in (info.repos if info else ())
        if r.repo_id in wanted and r.repo_type == "model"
    }


def delete_file(r: Resolved) -> int:
    """Remove one weight file of a downloaded repo. Its blob goes only when no other snapshot links to it."""
    rev = snapshot(r.repo_id, r.revision)
    if rev is None or r.gguf is None:
        return 0
    link = rev.snapshot_path / r.gguf
    blob = link.resolve()
    shared = any(
        other != link and other.resolve() == blob
        for other in rev.snapshot_path.parent.glob("*/**/*")
        if other.is_symlink()
    )
    link.unlink()
    if shared:
        return 0
    freed = blob.stat().st_size
    blob.unlink()
    return freed


def remove(r: Resolved) -> int:
    """Delete what `r` names: one quant when the repo has others on disk, else the whole repo. Bytes freed."""
    path = local_path(r)
    if r.gguf and path:
        name = r.gguf.rsplit("/", 1)[-1]
        if any(p.name != name for p in Path(path).glob("**/*.gguf")):
            return delete_file(r)
    return delete(r.repo_id)


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
    with config.edit() as data:
        data.get("pins", {}).pop(repo_id, None)
        data.get("released", {}).pop(repo_id, None)
        data["trusted"] = [t for t in data.get("trusted", []) if not t.startswith(f"{repo_id}@")]
    return freed


def is_trusted(r: Resolved) -> bool:
    return not r.family.runs_repo_code or f"{r.repo_id}@{r.revision}" in config.load().get("trusted", [])


def trust(r: Resolved) -> None:
    key = f"{r.repo_id}@{r.revision}"
    with config.edit() as data:
        if key not in data.setdefault("trusted", []):
            data["trusted"].append(key)
