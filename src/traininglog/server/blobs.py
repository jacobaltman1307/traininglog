"""Content-addressed local blob storage for uploaded checkpoints, replays, and traces."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import BinaryIO


class BlobStore:
    """Stores immutable blobs keyed by (category, sha256) under <data_dir>/blobs/."""

    def __init__(self, blobs_dir: Path):
        self.blobs_dir = Path(blobs_dir)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

    def _category_dir(self, category: str) -> Path:
        d = self.blobs_dir / category
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_bytes(self, category: str, data: bytes) -> tuple[str, int]:
        digest = hashlib.sha256(data).hexdigest()
        cdir = self._category_dir(category)
        dest = cdir / digest
        if not dest.exists():
            tmp = cdir / f".{digest}.{os.getpid()}.tmp"
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        return digest, len(data)

    def save_stream(
        self, category: str, stream: BinaryIO, chunk_size: int = 1024 * 1024
    ) -> tuple[str, int]:
        cdir = self._category_dir(category)
        tmp = cdir / f".upload.{os.getpid()}.{id(stream)}.tmp"
        hasher = hashlib.sha256()
        total = 0
        try:
            with open(tmp, "wb") as out:
                while True:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    out.write(chunk)
                    total += len(chunk)
            digest = hasher.hexdigest()
            dest = cdir / digest
            if dest.exists():
                tmp.unlink(missing_ok=True)
            else:
                os.replace(tmp, dest)
            return digest, total
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def save_file(self, category: str, src_path: Path) -> tuple[str, int]:
        with open(src_path, "rb") as f:
            return self.save_stream(category, f)

    def get_path(self, category: str, sha256: str) -> Path | None:
        if not sha256 or "/" in sha256 or "\\" in sha256 or ".." in sha256:
            return None
        p = self.blobs_dir / category / sha256
        return p if p.is_file() else None

    def delete(self, category: str, sha256: str) -> bool:
        p = self.get_path(category, sha256)
        if p is None:
            return False
        p.unlink(missing_ok=True)
        return True
