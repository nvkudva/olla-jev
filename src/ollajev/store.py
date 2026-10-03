"""Resolve model names to a pinned Hugging Face snapshot, download it, list and delete downloads."""

from __future__ import annotations

import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import CachedRevisionInfo, HfApi, HFCacheInfo, snapshot_download
from huggingface_hub import scan_cache_dir as _scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import CacheNotFound, LocalEntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.utils import filter_repo_objects
from huggingface_hub.utils import tqdm as hf_tqdm

from . import config, names
from .adapters import Family, detect
from .names import Ref, parse


@dataclass
class Resolved:
    ref: Ref
    family: Family
    revision: str
    files: list[str]
    weights: str | None = None  # the one .gguf or .onnx file this name selects; None for full weights
    allow: list[str] | None = field(default=None)
    created: str | None = None  # the repo's creation date, recorded as release_date once it is downloaded
    base: Resolved | None = None  # a quantized copy's base repo, which supplies config and tokenizer files

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


def bases() -> dict[str, dict[str, str]]:
    """copy repo -> {"repo", "revision"} of the base it was resolved against, so copies resolve offline."""
    return config.load().get("bases", {})


def _quantized_from(base_models: dict | None) -> list[str]:
    """The repos Hugging Face lists as this repo's base when it is a quantization of them, else none."""
    if not base_models or base_models.get("relation") != "quantized":
        return []
    return [m["id"] for m in base_models.get("models", []) if m.get("id")]


def _remote_files(repo_id: str, revision: str | None) -> tuple[str, str | None, list[str], list[str]]:
    try:
        info = HfApi().model_info(repo_id, revision=revision, expand=["sha", "createdAt", "siblings", "baseModels"])
    except RepositoryNotFoundError:
        raise LookupError(
            f"model {repo_id!r} not found on Hugging Face (private or gated repos need `hf auth login`)"
        ) from None
    if info.sha is None:
        raise LookupError(f"Hugging Face returned no commit for {repo_id!r}")
    created = info.created_at.date().isoformat() if info.created_at else None
    files = [s.rfilename for s in info.siblings or []]
    return info.sha, created, files, _quantized_from(getattr(info, "base_models", None))


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
    return _resolve(ref, pins().get(ref.repo_id), online=online, allow_base=True)


def _resolve(ref: Ref, revision: str | None, *, online: bool, allow_base: bool) -> Resolved:
    created = None
    base_ids: list[str] = []
    files = _local_files(ref.repo_id, revision) if revision else None
    weights = None
    if files is not None:
        try:
            weights = _pick(ref, files)
        except ValueError:
            if not online:
                raise LookupError(f"{ref.name} is not downloaded; run: ollajev pull {ref.name}") from None
            files = None  # a quant of a downloaded repo that is not on disk yet
    if files is None:
        if not online:
            raise LookupError(f"{ref.name} is not downloaded; run: ollajev pull {ref.name}")
        sha, created, files, base_ids = _remote_files(ref.repo_id, revision)
        if revision is None:
            revision = sha  # pinned by download(), once the weights are on disk
        weights = _pick(ref, files)
    if revision is None:  # unreachable: set from the pin or from the remote lookup above
        raise LookupError(f"no revision for {ref.name}")
    try:
        family = detect(ref.repo_id, files)
    except LookupError:
        base = _base(ref.repo_id, weights, files, base_ids, online=online) if allow_base else None
        if base is None or weights is None:
            raise
        allow = [weights, *names.sidecars(weights)]
        return Resolved(ref, base.family, revision, files, weights, allow=allow, created=created, base=base)
    if weights and not _runs(family, weights, files):
        if ref.tag:
            raise ValueError(f"the {family.name} family cannot run {weights}")
        weights = None  # e.g. an ONNX export next to the weights the family loads
    resolved = Resolved(ref, family, revision, files, weights, created=created)
    resolved.allow = family.allow_patterns(resolved)
    return resolved


def _runs(family: Family, weights: str, files: list[str]) -> bool:
    """Whether `family` loads this one weight file (a GGUF quant, an ONNX graph) of a repo with `files`."""
    runs = getattr(family, "runs_weights", None)
    return bool(runs and runs(weights, files))


def _inherits(family: Family) -> bool:
    """Whether quantized copies of this family's repos can run with the base repo's config files."""
    return bool(getattr(family, "base_files", None)) and not family.runs_repo_code


def _base(repo_id: str, weights: str | None, files: list[str], base_ids: list[str], *, online: bool) -> Resolved | None:
    """The base repo of a quantization that no family recognises by itself, when the base's family can run the
    copy's weight file with the base's config files. Offline it uses the base recorded at download time. A base
    whose family declines the file's layout makes this a LookupError that says so."""
    if weights is None:
        return None
    recorded = bases().get(repo_id)
    candidates = [(recorded["repo"], recorded["revision"])] if recorded else [(b, None) for b in base_ids]
    declined = None
    for base_id, revision in candidates:
        try:
            base = _resolve(Ref(base_id), revision, online=online, allow_base=False)
        except (LookupError, ValueError):
            continue
        if not _inherits(base.family):
            continue
        if not _runs(base.family, weights, files):
            declined = base.family.name
            continue
        base.allow = base.family.base_files  # type: ignore[attr-defined]
        return base
    if declined:
        raise LookupError(f"{repo_id}: the {declined} family does not run {weights} (unsupported file or layout)")
    return None


@dataclass(frozen=True)
class Hit:
    """A Hugging Face repo found by `search`. family is None when no adapter runs it."""

    repo_id: str
    downloads: int
    family: str | None


@dataclass(frozen=True)
class Variant:
    """One downloadable form of a repo: a GGUF quant, an ONNX export, or the full weights. size is the download in
    bytes."""

    name: str  # what `pull` takes
    label: str
    size: int


def _family(repo_id: str, files: list[str], base_models: dict | None = None) -> Family | None:
    """The family that runs a repo, through its quantized base when no family recognises the repo itself."""
    try:
        return detect(repo_id, files)
    except LookupError:
        pass
    weights = names.weight_files(files)
    for base_id in _quantized_from(base_models) if weights else []:
        family = _family(base_id, list(_repo_files(base_id)))
        if family and _inherits(family) and any(_runs(family, w, files) for w in weights):
            return family
    return None


@functools.lru_cache(maxsize=256)
def _repo_files(repo_id: str) -> tuple[str, ...]:
    try:
        info = HfApi().model_info(repo_id, expand=["siblings"])
    except RepositoryNotFoundError:
        return ()
    return tuple(s.rfilename for s in info.siblings or [])


def search(query: str, limit: int = 40) -> list[Hit]:
    """Repos matching every word of `query`, supported ones first, then most downloaded first. A repo name or URL
    finds that repo."""
    api = HfApi()
    expand: list = ["siblings", "downloads", "baseModels"]
    try:
        found = [api.model_info(parse(query).repo_id, expand=expand)]
    except (ValueError, RepositoryNotFoundError):
        words = query.lower().split()
        if not words:
            return []
        # Listing 200 repos takes 25 s with the default fields and 50 s with `siblings`, but under a second
        # with `downloads` only, so list ids and read the files of the first matches one by one.
        ids = [
            m.id
            for m in api.list_models(search=max(words, key=len), sort="downloads", limit=200, expand=["downloads"])
            if all(w in m.id.lower() for w in words)
        ][:limit]
        found = None

    def hit(m) -> Hit:
        family = _family(m.id, [s.rfilename for s in m.siblings or []], getattr(m, "base_models", None))
        return Hit(m.id, m.downloads or 0, family.name if family else None)

    if found is not None:
        hits = [hit(m) for m in found]
    else:  # one chain per repo: its files, then (for a quantized copy) its base repo's files
        with ThreadPoolExecutor(16) as pool:
            hits = list(pool.map(lambda repo_id: hit(api.model_info(repo_id, expand=expand)), ids))
    return sorted(hits, key=lambda h: h.family is None)[:limit]  # stable: keeps the download order


def variants(repo_id: str) -> list[Variant]:
    """A repo's GGUF quants or ONNX exports, or its full weights, with download sizes, smallest first."""
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings or []}
    family = _family(repo_id, list(sizes))
    copy = False
    if family is None:
        family = _family(
            repo_id, list(sizes), getattr(api.model_info(repo_id, expand=["baseModels"]), "base_models", None)
        )
        copy = family is not None
    return _variants(repo_id, info.sha or "", sizes, family, copy=copy)


def _variants(
    repo_id: str, sha: str, sizes: dict[str, int], family: Family | None, *, copy: bool = False
) -> list[Variant]:
    """copy: the weights come from this repo and the config files from its base, so only the weights count."""
    files = list(sizes)
    tags = {w: t for w, t in names.labels(files).items() if family is None or _runs(family, w, files)}
    found = []
    for weights, tag in tags.items() if tags else [(None, None)]:
        ref = Ref(repo_id, tag)
        if family and not copy:
            allow = family.allow_patterns(Resolved(ref, family, sha, files, weights))
        else:
            allow = [weights, *names.sidecars(weights)] if weights else None
        size = sum(sizes[f] for f in filter_repo_objects(files, allow_patterns=allow))
        found.append(Variant(ref.name, tag or "full weights", size))
    return sorted(found, key=lambda v: v.size)


def _pick(ref: Ref, files: list[str]) -> str | None:
    if names.weight_files(files):
        return names.pick_weights(files, ref.tag)
    if ref.tag:
        raise ValueError(f"{ref.repo_id} has no quantized files; drop ':{ref.tag}'")
    return None


def local_path(r: Resolved) -> str | None:
    """The snapshot folder holding `r`'s weights, or None until it and, for a copy, its base files are on disk."""
    if r.base and local_path(r.base) is None:
        return None
    try:
        return snapshot_download(
            r.repo_id, revision=r.revision, allow_patterns=r.allow, cache_dir=config.models_dir(), local_files_only=True
        )
    except LocalEntryNotFoundError:
        return None


class Cancelled(Exception):
    pass


def _cancellable(cancel: threading.Event | None) -> type[hf_tqdm] | None:
    """A progress bar class that aborts the download at its next update once `cancel` is set. Partial files stay
    in the cache, so the next pull resumes."""
    if cancel is None:
        return None

    class Bar(hf_tqdm):
        def update(self, n: float | None = 1) -> bool | None:
            if cancel.is_set():
                raise Cancelled
            return super().update(n)

    return Bar


def download(r: Resolved, cancel: threading.Event | None = None) -> str:
    """Fetch the snapshot, then pin the repo to this commit if it has no pin yet. A copy also fetches its base's
    config files and records which base commit they came from. Setting `cancel` aborts it with Cancelled."""
    fetch = functools.partial(snapshot_download, cache_dir=config.models_dir(), tqdm_class=_cancellable(cancel))
    if r.base:
        fetch(r.base.repo_id, revision=r.base.revision, allow_patterns=r.base.allow)
        with config.edit() as data:
            data.setdefault("bases", {})[r.repo_id] = {"repo": r.base.repo_id, "revision": r.base.revision}
    path = fetch(r.repo_id, revision=r.revision, allow_patterns=r.allow)
    if r.repo_id not in pins():
        _pin(r.repo_id, r.revision, r.created)
    return path


def download_size(r: Resolved) -> int:
    """Bytes `download` fetches for `r`'s own repo."""
    info = HfApi().model_info(r.repo_id, revision=r.revision, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings or []}
    return sum(sizes[f] for f in filter_repo_objects(list(sizes), allow_patterns=r.allow))


def bytes_on_disk(repo_id: str) -> int:
    """Bytes of `repo_id` in the cache, partial downloads included."""
    blobs = Path(config.models_dir() or HF_HUB_CACHE) / repo_folder_name(repo_id=repo_id, repo_type="model") / "blobs"
    return sum(p.stat().st_size for p in blobs.glob("*") if p.is_file()) if blobs.is_dir() else 0


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
    """Remove one weight file of a downloaded repo, with its ONNX external data. A blob goes only when no other
    snapshot links to it."""
    rev = snapshot(r.repo_id, r.revision)
    if rev is None or r.weights is None:
        return 0
    freed = 0
    for name in [r.weights, *names.sidecars(r.weights)]:
        link = rev.snapshot_path / name
        if not link.is_symlink():
            continue
        blob = link.resolve()
        shared = any(
            other != link and other.resolve() == blob
            for other in rev.snapshot_path.parent.glob("*/**/*")
            if other.is_symlink()
        )
        link.unlink()
        if not shared:
            freed += blob.stat().st_size
            blob.unlink()
    return freed


def remove(r: Resolved) -> int:
    """Delete what `r` names: one variant when the repo has others on disk, else the whole repo. Bytes freed."""
    path = local_path(r)
    if r.weights and path:
        on_disk = names.weight_files([str(p.relative_to(path)) for p in Path(path).glob("**/*")])
        if any(f != r.weights for f in on_disk):
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
        data.get("bases", {}).pop(repo_id, None)
        data["trusted"] = [t for t in data.get("trusted", []) if not t.startswith(f"{repo_id}@")]
    return freed


def is_trusted(r: Resolved) -> bool:
    return not r.family.runs_repo_code or f"{r.repo_id}@{r.revision}" in config.load().get("trusted", [])


def trust(r: Resolved) -> None:
    key = f"{r.repo_id}@{r.revision}"
    with config.edit() as data:
        if key not in data.setdefault("trusted", []):
            data["trusted"].append(key)
