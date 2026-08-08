from __future__ import annotations

import fnmatch
import hashlib
import os
import pickle
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from db_types import Database, FileEntry


MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="cache",
)

INDEX_DIRS = [
    "C:/",
    "D:/",
]

SKIP_DIRS = {
    # Version control
    ".git",
    ".svn",
    ".hg",

    # Python
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",

    # JS
    "node_modules",
    ".next",
    ".nuxt",

    # Build outputs
    "build",
    "dist",
    "out",
    "target",
    "bin",
    "obj",

    # IDE
    ".idea",
    ".vs",
    ".vscode-server",

    # Package caches
    ".cargo",
    ".gradle",

    # OS
    "$Recycle.Bin",
    "System Volume Information",

    # Browser
    "Cache",
    "Code Cache",
    "GPUCache",

    # Temporary
    "Temp",
    "tmp",
}

SKIP_PATHS = {
    r"C:\Windows",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\ProgramData",
    r"C:\Recovery",
    r"C:\PerfLogs",
}

SKIP_PATTERNS = {
    "*-lock*",
    "*.min.*",
}

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",

    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".kt",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".zig",
    ".lua",
    ".php",
    ".rb",
    ".swift",
    ".scala",
    ".sh",
    ".bat",
    ".ps1",

    ".json",
    ".json5",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".xml",

    ".html",
    ".css",
    ".scss",

    ".csv",
    ".tsv",

    ".tex",

    ".sql",

    ".log",

    ".dockerfile",

    ".gitignore",
    ".editorconfig",
}

MAX_BYTES = 10 * 1024 * 1024
BATCH_SIZE = 128


def is_text_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(256)

        if b"\0" in chunk:
            return False

        chunk.decode("utf-8", errors="strict")
        return True
    except Exception:
        return False


def sha256_int_for_file(path: str) -> int:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return int.from_bytes(h.digest(), "big", signed=False)


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _should_skip_dirname(dirname: str) -> bool:
    return dirname in SKIP_DIRS or (dirname and dirname[0] == ".")


def _should_skip_path(path: str) -> bool:
    npth = _norm(path)

    # 2) filter out SKIP_PATHS
    for sp in SKIP_PATHS:
        nsp = _norm(sp).rstrip("\\/ ")
        if npth.startswith(nsp.rstrip("\\/") + os.sep) or npth == nsp:
            return True

    # 4) filter out SKIP_PATTERNS
    base = os.path.basename(path)
    for pat in SKIP_PATTERNS:
        if fnmatch.fnmatch(base, pat) or fnmatch.fnmatch(path, pat):
            return True

    # 5) filter out everything but TEXT_EXTENSIONS
    ext = os.path.splitext(path)[1].lower()
    if ext not in TEXT_EXTENSIONS:
        base_lower = os.path.basename(path).lower()
        if base_lower not in TEXT_EXTENSIONS:
            return True

    return False


def iter_text_files() -> Iterable[str]:
    # 1) recurse all INDEX_DIRS
    for root in INDEX_DIRS:
        if not os.path.exists(root):
            continue

        # 3) filter out SKIP_DIRS
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            dirnames[:] = [d for d in dirnames if not _should_skip_dirname(d)]

            if _should_skip_path(dirpath):
                dirnames[:] = []
                continue

            for fn in filenames:
                path = os.path.join(dirpath, fn)
                if _should_skip_path(path):
                    continue

                # 6) run binary check via is_text_file
                if not is_text_file(path):
                    continue

                yield path


def load_text_for_embedding(path: str) -> str | None:
    if not is_text_file(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(MAX_BYTES)
        return f"{path}\n{content}"
    except Exception:
        return None


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def load_db(path: str = "db.pickle") -> Database | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # old dict format -> ignore
    if isinstance(obj, dict):
        return None
    return obj


def save_db(db: Database, path: str = "db.pickle") -> None:
    with open(path, "wb") as f:
        pickle.dump(db, f)


def main() -> None:
    db = load_db()

    if db is None:
        files: list[FileEntry] = []
        embeddings = np.zeros((0, 384), dtype=np.float32)
        path_to_index: dict[str, int] = {}
    else:
        files = db.files
        embeddings = db.embeddings
        path_to_index = db.path_to_index

    # hash_to_index for dedup-by-content (requested)
    hash_to_index: dict[int, int] = {}
    for i, fe in enumerate(files):
        hash_to_index.setdefault(fe.hash, i)

    # Decide which paths need embedding update or are new
    pending_update: list[str] = []

    for path in iter_text_files():
        key = _norm(path)
        try:
            st = os.stat(path)
        except Exception:
            continue

        mtime = float(st.st_mtime)
        size = int(st.st_size)

        if key in path_to_index:
            i = path_to_index[key]
            fe = files[i]
            if fe.mtime == mtime and fe.size == size:
                continue

        pending_update.append(path)

    # Embed in batches
    batch_paths: list[str] = []
    batch_texts: list[str] = []

    def flush() -> None:
        nonlocal batch_paths, batch_texts, embeddings
        if not batch_texts:
            return

        vecs = MODEL.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        vecs = _normalize_rows(vecs)

        for p, v in zip(batch_paths, vecs):
            key = _norm(p)
            # content hash dedup
            hval = sha256_int_for_file(p)
            if hval in hash_to_index and key not in path_to_index:
                reuse_i = hash_to_index[hval]
                idx = len(files)
                path_to_index[key] = idx
                files.append(FileEntry(path=p, mtime=float(os.path.getmtime(p)), size=int(os.path.getsize(p)), hash=hval))
                embeddings = np.vstack([embeddings, embeddings[reuse_i].reshape(1, -1)])
            elif key in path_to_index:
                idx = path_to_index[key]
                files[idx].mtime = float(os.path.getmtime(p))
                files[idx].size = int(os.path.getsize(p))
                files[idx].hash = hval
                hash_to_index.setdefault(hval, idx)
                # if this hash maps to another index, reuse embedding there
                if hval in hash_to_index and hash_to_index[hval] != idx:
                    embeddings[idx] = embeddings[hash_to_index[hval]]
                else:
                    embeddings[idx] = v
            else:
                idx = len(files)
                path_to_index[key] = idx
                files.append(FileEntry(path=p, mtime=float(os.path.getmtime(p)), size=int(os.path.getsize(p)), hash=hval))
                hash_to_index.setdefault(hval, idx)
                if embeddings.shape[0] == idx:
                    embeddings = np.vstack([embeddings, v.reshape(1, -1)])
                else:
                    embeddings[idx] = v

        batch_paths = []
        batch_texts = []

    for p in pending_update:
        text = load_text_for_embedding(p)
        if text is None:
            continue
        batch_paths.append(p)
        batch_texts.append(text)
        if len(batch_texts) >= BATCH_SIZE:
            flush()

    flush()

    save_db(Database(files=files, embeddings=embeddings, path_to_index=path_to_index))


if __name__ == "__main__":
    main()

