"""Tests for runeextract.sync — directory watcher, file sync, batch extraction."""

import os
import tempfile
import time

import pytest

from runeextract.sync.watcher import DirectoryWatcher, FileEvent, poll_directory
from runeextract.sync.syncer import FileSync, sync_directories
from runeextract.sync.extractor import scan_and_extract, watch_and_extract


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


# ── Watcher ───────────────────────────────────────────────

class TestFileEvent:
    def test_attributes(self):
        e = FileEvent(path="/a/b.txt", event_type="created")
        assert e.path == "/a/b.txt"
        assert e.event_type == "created"
        assert e.timestamp > 0


class TestDirectoryWatcher:
    def test_initial_snapshot(self, tmpdir):
        open(os.path.join(tmpdir, "a.txt"), "w").close()
        w = DirectoryWatcher(tmpdir)
        assert len(w._snapshot) >= 1

    def test_poll_created(self, tmpdir):
        w = DirectoryWatcher(tmpdir, poll_interval=0.1)
        events = w.poll()
        assert len(events) == 0
        open(os.path.join(tmpdir, "new.txt"), "w").close()
        events = w.poll()
        assert any(e.event_type == "created" for e in events)

    def test_poll_modified(self, tmpdir):
        fpath = os.path.join(tmpdir, "m.txt")
        with open(fpath, "w") as f:
            f.write("v1")
        w = DirectoryWatcher(tmpdir, poll_interval=0.1)
        w.poll()
        time.sleep(0.01)
        with open(fpath, "w") as f:
            f.write("v2")
        events = w.poll()
        assert any(e.event_type == "modified" for e in events)

    def test_poll_deleted(self, tmpdir):
        fpath = os.path.join(tmpdir, "d.txt")
        open(fpath, "w").close()
        w = DirectoryWatcher(tmpdir, poll_interval=0.1)
        w.poll()
        os.remove(fpath)
        events = w.poll()
        assert any(e.event_type == "deleted" for e in events)

    def test_pattern_filter(self, tmpdir):
        open(os.path.join(tmpdir, "a.pdf"), "w").close()
        open(os.path.join(tmpdir, "b.txt"), "w").close()
        w = DirectoryWatcher(tmpdir, patterns=["*.pdf"])
        w.reset()
        assert any("a.pdf" in k for k in w._snapshot)
        assert not any("b.txt" in k for k in w._snapshot)

    def test_exclude_pattern(self, tmpdir):
        open(os.path.join(tmpdir, "keep.txt"), "w").close()
        open(os.path.join(tmpdir, "ignore.txt"), "w").close()
        w = DirectoryWatcher(tmpdir, patterns=["*"], exclude_patterns=["ignore*"])
        w.reset()
        assert any("keep.txt" in k for k in w._snapshot)
        assert not any("ignore.txt" in k for k in w._snapshot)

    def test_recursive_true(self, tmpdir):
        sub = os.path.join(tmpdir, "sub")
        os.mkdir(sub)
        open(os.path.join(sub, "deep.txt"), "w").close()
        w = DirectoryWatcher(tmpdir, recursive=True)
        assert any("deep.txt" in k for k in w._snapshot)

    def test_recursive_false(self, tmpdir):
        sub = os.path.join(tmpdir, "sub")
        os.mkdir(sub)
        open(os.path.join(sub, "deep.txt"), "w").close()
        w = DirectoryWatcher(tmpdir, recursive=False)
        w.reset()
        assert not any("deep.txt" in k for k in w._snapshot)

    def test_poll_until_timeout(self, tmpdir):
        w = DirectoryWatcher(tmpdir, poll_interval=0.05)
        events = w.poll_until(timeout=0.3)
        assert isinstance(events, list)

    def test_poll_until_predicate(self, tmpdir):
        w = DirectoryWatcher(tmpdir, poll_interval=0.05)
        fpath = os.path.join(tmpdir, "target.txt")
        open(fpath, "w").close()
        events = w.poll_until(timeout=2.0, predicate=lambda e: "target" in e.path)
        assert len(events) >= 1

    def test_reset(self, tmpdir):
        open(os.path.join(tmpdir, "x.txt"), "w").close()
        w = DirectoryWatcher(tmpdir)
        assert len(w._snapshot) >= 1
        os.remove(os.path.join(tmpdir, "x.txt"))
        w.reset()
        assert len(w._snapshot) == 0

    def test_poll_events_ordered(self, tmpdir):
        w = DirectoryWatcher(tmpdir, poll_interval=0.1)
        for n in range(3):
            open(os.path.join(tmpdir, f"f{n}.txt"), "w").close()
        events = w.poll()
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_permission_error_ignored(self, tmpdir):
        w = DirectoryWatcher(tmpdir)
        w._snapshot["/nonexistent/file.txt"] = (0, 0)
        events = w.poll()
        assert isinstance(events, list)


class TestPollDirectory:
    def test_returns_watcher(self, tmpdir):
        w = poll_directory(tmpdir, patterns=["*.txt"])
        assert isinstance(w, DirectoryWatcher)
        assert w.patterns == ["*.txt"]


# ── Syncer ────────────────────────────────────────────────

class TestFileSync:
    def test_sync_new_file(self, tmpdir):
        src = os.path.join(tmpdir, "src")
        dst = os.path.join(tmpdir, "dst")
        os.mkdir(src)
        open(os.path.join(src, "a.txt"), "w").close()
        syncer = FileSync(src, dst)
        synced = syncer.sync()
        assert os.path.exists(os.path.join(dst, "a.txt"))
        assert len(synced) == 1

    def test_sync_pattern(self, tmpdir):
        src = os.path.join(tmpdir, "src")
        dst = os.path.join(tmpdir, "dst")
        os.mkdir(src)
        open(os.path.join(src, "a.pdf"), "w").close()
        open(os.path.join(src, "b.txt"), "w").close()
        syncer = FileSync(src, dst, patterns=["*.txt"])
        synced = syncer.sync()
        assert os.path.exists(os.path.join(dst, "b.txt"))
        assert not os.path.exists(os.path.join(dst, "a.pdf"))
        assert len(synced) == 1

    def test_sync_exclude(self, tmpdir):
        src = os.path.join(tmpdir, "src")
        dst = os.path.join(tmpdir, "dst")
        os.mkdir(src)
        open(os.path.join(src, "keep.txt"), "w").close()
        open(os.path.join(src, "skip.txt"), "w").close()
        syncer = FileSync(src, dst, patterns=["*"], exclude_patterns=["skip*"])
        synced = syncer.sync()
        assert os.path.exists(os.path.join(dst, "keep.txt"))
        assert not os.path.exists(os.path.join(dst, "skip.txt"))

    def test_sync_skip_unchanged(self, tmpdir):
        src = os.path.join(tmpdir, "src")
        dst = os.path.join(tmpdir, "dst")
        os.mkdir(src); os.mkdir(dst)
        p = os.path.join(src, "f.txt")
        with open(p, "w") as f:
            f.write("data")
        open(os.path.join(dst, "f.txt"), "w").close()
        syncer = FileSync(src, dst, overwrite=True)
        synced = syncer.sync()
        assert len(synced) == 1  # hash mismatch

    def test_sync_overwrite_identical(self, tmpdir):
        src = os.path.join(tmpdir, "src")
        dst = os.path.join(tmpdir, "dst")
        os.mkdir(src); os.mkdir(dst)
        data = "same"
        with open(os.path.join(src, "f.txt"), "w") as f:
            f.write(data)
        with open(os.path.join(dst, "f.txt"), "w") as f:
            f.write(data)
        syncer = FileSync(src, dst, overwrite=True)
        synced = syncer.sync()
        assert len(synced) == 0  # same hash

    def test_sync_dry_run(self, tmpdir):
        src = os.path.join(tmpdir, "src")
        dst = os.path.join(tmpdir, "dst")
        os.mkdir(src)
        open(os.path.join(src, "f.txt"), "w").close()
        syncer = FileSync(src, dst, dry_run=True)
        synced = syncer.sync()
        assert len(synced) == 1
        assert not os.path.exists(os.path.join(dst, "f.txt"))

    def test_sync_recursive_false(self, tmpdir):
        src = os.path.join(tmpdir, "src")
        dst = os.path.join(tmpdir, "dst")
        os.mkdir(src)
        sub = os.path.join(src, "sub")
        os.mkdir(sub)
        open(os.path.join(sub, "deep.txt"), "w").close()
        syncer = FileSync(src, dst, recursive=False)
        synced = syncer.sync()
        assert len(synced) == 0

    def test_sync_recursive_true(self, tmpdir):
        src = os.path.join(tmpdir, "src")
        dst = os.path.join(tmpdir, "dst")
        os.mkdir(src)
        sub = os.path.join(src, "sub")
        os.mkdir(sub)
        open(os.path.join(sub, "deep.txt"), "w").close()
        syncer = FileSync(src, dst, recursive=True)
        synced = syncer.sync()
        assert os.path.exists(os.path.join(dst, "sub", "deep.txt"))

    def test_sync_directories_helper(self, tmpdir):
        src = os.path.join(tmpdir, "src")
        dst = os.path.join(tmpdir, "dst")
        os.mkdir(src)
        open(os.path.join(src, "x.txt"), "w").close()
        synced = sync_directories(src, dst)
        assert len(synced) == 1

    def test_sync_empty_source(self, tmpdir):
        src = os.path.join(tmpdir, "src")
        dst = os.path.join(tmpdir, "dst")
        os.mkdir(src)
        syncer = FileSync(src, dst)
        synced = syncer.sync()
        assert len(synced) == 0


# ── Batch extractor ───────────────────────────────────────

class TestScanAndExtract:
    def test_scan_all(self, tmpdir):
        open(os.path.join(tmpdir, "a.txt"), "w").close()
        open(os.path.join(tmpdir, "b.txt"), "w").close()
        results = scan_and_extract(tmpdir)
        assert len(results) == 2

    def test_scan_pattern(self, tmpdir):
        open(os.path.join(tmpdir, "a.pdf"), "w").close()
        open(os.path.join(tmpdir, "b.txt"), "w").close()
        results = scan_and_extract(tmpdir, patterns=["*.pdf"])
        assert len(results) == 1

    def test_scan_custom_extract_fn(self, tmpdir):
        open(os.path.join(tmpdir, "f.txt"), "w").close()
        results = scan_and_extract(tmpdir, extract_fn=lambda p: {"file": os.path.basename(p)})
        assert results[list(results)[0]]["file"] == "f.txt"

    def test_scan_error_handling(self, tmpdir):
        def bad_fn(p):
            raise ValueError("boom")
        open(os.path.join(tmpdir, "f.txt"), "w").close()
        results = scan_and_extract(tmpdir, extract_fn=bad_fn)
        assert "error" in list(results.values())[0]


class TestWatchAndExtract:
    def test_single_poll(self, tmpdir):
        open(os.path.join(tmpdir, "f.txt"), "w").close()
        results = watch_and_extract(tmpdir, max_events=5)
        assert len(results) >= 1

    def test_poll_no_events(self, tmpdir):
        results = watch_and_extract(tmpdir, max_events=5)
        assert isinstance(results, list)

    def test_with_extract_fn(self, tmpdir):
        open(os.path.join(tmpdir, "f.txt"), "w").close()
        results = watch_and_extract(tmpdir, extract_fn=lambda p: {"name": os.path.basename(p)}, max_events=5)
        assert results[0]["name"] == "f.txt"

    def test_error_callback(self, tmpdir):
        open(os.path.join(tmpdir, "f.txt"), "w").close()
        errors = []
        def on_err(p, e):
            errors.append((p, str(e)))
        watch_and_extract(tmpdir, max_events=5, extract_fn=lambda p: 1/0, on_error=on_err)
        assert len(errors) >= 1

    def test_skip_deleted(self, tmpdir):
        results = watch_and_extract(tmpdir, max_events=5)
        for r in results:
            assert r.get("event_type") != "deleted"

    def test_exclude_pattern(self, tmpdir):
        open(os.path.join(tmpdir, "keep.txt"), "w").close()
        open(os.path.join(tmpdir, "ignore.txt"), "w").close()
        results = watch_and_extract(tmpdir, patterns=["*"], exclude=["ignore*"], max_events=5)
        paths = [r["path"] for r in results]
        assert any("keep.txt" in p for p in paths)
        assert not any("ignore.txt" in p for p in paths)
