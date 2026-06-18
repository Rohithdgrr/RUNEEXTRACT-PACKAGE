"""File sync — sync files between source and destination directories."""

import fnmatch
import hashlib
import os
import shutil
from dataclasses import dataclass, field
from typing import List


@dataclass
class FileSync:
    source: str
    dest: str
    patterns: List[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: List[str] = field(default_factory=list)
    recursive: bool = True
    overwrite: bool = False
    dry_run: bool = False
    _synced: List[str] = field(default_factory=list)

    def sync(self, patterns: List[str] = None) -> List[str]:
        if patterns is not None:
            self.patterns = patterns
        self._synced = []
        for dirpath, dirnames, filenames in os.walk(self.source):
            if not self.recursive:
                dirnames.clear()
            rel = os.path.relpath(dirpath, self.source)
            dest_dir = os.path.join(self.dest, rel) if rel != "." else self.dest
            for fn in filenames:
                if not self._matches(fn):
                    continue
                src = os.path.join(dirpath, fn)
                dst = os.path.join(dest_dir, fn)
                if self._should_copy(src, dst):
                    self._copy_file(src, dst)
                    self._synced.append(src)
        return self._synced

    def _matches(self, filename: str) -> bool:
        if not any(fnmatch.fnmatch(filename, p) for p in self.patterns):
            return False
        if any(fnmatch.fnmatch(filename, p) for p in self.exclude_patterns):
            return False
        return True

    def _should_copy(self, src: str, dst: str) -> bool:
        if not os.path.exists(dst):
            return True
        if not self.overwrite:
            return False
        return self._file_hash(src) != self._file_hash(dst)

    def _copy_file(self, src: str, dst: str):
        if self.dry_run:
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    @staticmethod
    def _file_hash(path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


def sync_directories(source: str, dest: str, **kwargs) -> List[str]:
    syncer = FileSync(source=source, dest=dest, **kwargs)
    return syncer.sync()
