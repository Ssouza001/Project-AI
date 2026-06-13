from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackendPaths:
    base_dir: Path
    root_dir: Path
    data_dir: Path
    model_dir: Path


def build_paths(base_dir: Path) -> BackendPaths:
    root_dir = base_dir.parent
    return BackendPaths(
        base_dir=base_dir,
        root_dir=root_dir,
        data_dir=base_dir / "data",
        model_dir=base_dir / "Modelo_V3",
    )
