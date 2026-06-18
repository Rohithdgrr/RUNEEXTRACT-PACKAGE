"""Directory watcher — polls a directory for new/modified/deleted files."""

import os
import time
import fnmatch
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class FileEvent:
    path: str
    event_type: str  # "created", "modified", "deleted"
    timestamp: float = field(default_factory=time.time)


@dataclass
class DirectoryWatcher:
    directory: str
    patterns: List[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: List[str] = field(default_factory=list)
    poll_interval: float = 1.0
    recursive: bool = True
    _snapshot: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self._snapshot = self._take_snapshot()

    def _take_snapshot(self) -> dict:
        snapshot = {}
        for dirpath, dirnames, filenames in os.walk(self.directory):
            if not self.recursive:
                dirnames.clear()
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                if self._matches(fn):
                    try:
                        st = os.stat(full)
                        snapshot[full] = (st.st_mtime, st.st_size)
                    except (OSError, PermissionError):
                        continue
        return snapshot

    def _matches(self, filename: str) -> bool:
        if not any(fnmatch.fnmatch(filename, p) for p in self.patterns):
            return False
        if any(fnmatch.fnmatch(filename, p) for p in self.exclude_patterns):
            return False
        return True

    def poll(self) -> List[FileEvent]:
        current = self._take_snapshot()
        events = []

        for path in current:
            if path not in self._snapshot:
                events.append(FileEvent(path=path, event_type="created"))
            elif current[path] != self._snapshot[path]:
                events.append(FileEvent(path=path, event_type="modified"))

        for path in self._snapshot:
            if path not in current:
                events.append(FileEvent(path=path, event_type="deleted"))

        self._snapshot = current
        events.sort(key=lambda e: e.timestamp)
        return events

    def poll_until(self, timeout: float = 10.0, predicate: Optional[Callable[[FileEvent], bool]] = None) -> List[FileEvent]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            events = self.poll()
            if predicate:
                events = [e for e in events if predicate(e)]
            if events:
                return events
            time.sleep(min(self.poll_interval, deadline - time.time()))
        return []

    def reset(self):
        self._snapshot = self._take_snapshot()


def poll_directory(directory: str, patterns: List[str] = None, exclude: List[str] = None, **kwargs) -> DirectoryWatcher:
    if patterns is None:
        patterns = ["*"]
    watcher = DirectoryWatcher(directory, patterns=patterns, exclude_patterns=exclude or [], **kwargs)
    return watcher
