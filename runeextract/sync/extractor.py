"""Batch extraction — scan a directory and extract files, or watch and extract."""

from typing import Callable, List, Optional

from runeextract.sync.watcher import DirectoryWatcher, FileEvent


def scan_and_extract(
    directory: str,
    patterns: List[str] = None,
    exclude: List[str] = None,
    extract_fn: Callable[[str], object] = None,
    recursive: bool = True,
) -> dict:
    if patterns is None:
        patterns = ["*"]
    if extract_fn is None:
        extract_fn = lambda p: {"path": p, "extracted": True}

    watcher = DirectoryWatcher(directory, patterns=patterns, exclude_patterns=exclude or [], recursive=recursive)
    results = {}
    for path in watcher._snapshot:
        try:
            results[path] = extract_fn(path)
        except Exception as e:
            results[path] = {"path": path, "error": str(e)}
    return results


def watch_and_extract(
    directory: str,
    patterns: List[str] = None,
    exclude: List[str] = None,
    extract_fn: Callable[[str], object] = None,
    poll_interval: float = 1.0,
    recursive: bool = True,
    on_error: Optional[Callable[[str, Exception], None]] = None,
    max_events: int = 0,
) -> List[dict]:
    if patterns is None:
        patterns = ["*"]

    if extract_fn is None:
        extract_fn = lambda p: {"path": p, "extracted": True}

    watcher = DirectoryWatcher(
        directory,
        patterns=patterns,
        exclude_patterns=exclude or [],
        poll_interval=poll_interval,
        recursive=recursive,
    )

    collected = []
    def _process(path, event_type):
        try:
            result = extract_fn(path)
            if isinstance(result, dict):
                result["event_type"] = event_type
            collected.append(result)
        except Exception as e:
            if on_error:
                on_error(path, e)
            else:
                collected.append({"path": path, "event_type": event_type, "error": str(e)})

    for path in watcher._snapshot:
        _process(path, "existing")
        if max_events and len(collected) >= max_events:
            return collected

    while True:
        if max_events and len(collected) >= max_events:
            break
        events = watcher.poll()
        for event in events:
            if event.event_type == "deleted":
                continue
            _process(event.path, event.event_type)
            if max_events and len(collected) >= max_events:
                break
        if max_events == -1:
            break
        if not events:
            break

    return collected
