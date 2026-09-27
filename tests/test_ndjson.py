"""ndjson: запись/чтение, конкурентная запись, журнал ревью."""
import json
import threading

from doodle.ndjson_io import REVIEW_FILE, append_record, custom_samples, iter_records, read_records, review_state


def test_round_trip(tmp_path):
    p = tmp_path / "x.ndjson"
    recs = [{"key_id": str(i), "word": "банан", "drawing": [[[1, 2], [3, 4], [0, 16]]]} for i in range(5)]
    for r in recs:
        append_record(p, r)
    assert read_records(p) == recs


def test_skips_broken_tail_and_recovers(tmp_path):
    p = tmp_path / "x.ndjson"
    append_record(p, {"key_id": "a"})
    with open(p, "ab") as f:
        f.write(b'{"key_id": "brok')            # оборванная запись (сбой посреди write)
    append_record(p, {"key_id": "b"})
    assert [r["key_id"] for r in iter_records(p)] == ["a", "b"]


def test_concurrent_appends(tmp_path):
    p = tmp_path / "c.ndjson"
    big = [[list(range(300)), list(range(300)), list(range(300))]]

    def worker(t):
        for i in range(40):
            append_record(p, {"key_id": f"{t}-{i}", "drawing": big})

    th = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    lines = p.read_bytes().splitlines()
    assert len(lines) == 320
    assert len({json.loads(line)["key_id"] for line in lines}) == 320


def test_review_soft_delete(tmp_path):
    for k in "abc":
        append_record(tmp_path / "rude.ndjson", {"key_id": k, "word": "rude"})
    append_record(tmp_path / REVIEW_FILE, {"key_id": "a", "action": "delete"})
    append_record(tmp_path / REVIEW_FILE, {"key_id": "b", "action": "flag"})
    append_record(tmp_path / REVIEW_FILE, {"key_id": "b", "action": "ok"})
    assert review_state(tmp_path) == {"a": "delete", "b": "ok"}
    assert [r["key_id"] for r in custom_samples(tmp_path, "rude")] == ["b", "c"]
    assert len(custom_samples(tmp_path, "rude", exclude=())) == 3
