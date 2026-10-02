"""Build a multi-version repository from a base repository and per-version overlays (Phase 5).

samples/js_history/ holds v2/, v3/, ... directories, each containing only the files that changed in
that version (full contents of the new file). Applying them cumulatively on top of samples/js_repo
gives a committed, reproducible version history with a known lineage for every function.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def version_dirs(overlays: str | Path) -> list[Path]:
    """The overlay directories in version order (v2, v3, ..., v10: numeric, not alphabetical)."""
    dirs = [d for d in Path(overlays).iterdir() if d.is_dir() and d.name[1:].isdigit()]
    return sorted(dirs, key=lambda d: int(d.name[1:]))


def build_timeline(base: str | Path, overlays: str | Path, workdir: str | Path):
    """Yield (version, repo_path) for v1 (the base) and then each overlay applied in turn. The same
    working directory is updated in place between yields, as a checkout moving forward in time."""
    work = Path(workdir) / "repo"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(base, work)
    yield "v1", work
    for d in version_dirs(overlays):
        for f in d.rglob("*"):
            if f.is_file():
                target = work / f.relative_to(d)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, target)
        yield d.name, work
