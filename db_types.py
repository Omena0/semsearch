from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FileEntry:
    path: str
    mtime: float
    size: int
    hash: int


@dataclass(slots=True)
class Database:
    files: list[FileEntry]
    embeddings: np.ndarray  # shape (N, D)
    path_to_index: dict[str, int]

