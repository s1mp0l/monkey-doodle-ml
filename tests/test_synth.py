"""Генератор синтетических rude: валидные записи, воспроизводимость, проходят проверку импортёра."""
import json

from doodle import synth
from doodle.preprocess import normalize
from import_contrib import Reject, validate


def _strip(recs):
    return [{k: v for k, v in r.items() if k != "timestamp"} for r in recs]


def test_records_valid_and_pass_importer():
    recs = synth.generate(60, seed=5)
    assert len(recs) == 60 and len({r["key_id"] for r in recs}) == 60
    for r in recs:
        assert r["word"] == "rude" and r["source"] == "synthetic" and r["generator"]["seed"] == 5
        assert r["simplified"] == normalize(r["drawing"])
        for s in r["drawing"]:
            assert len(s) == 3 and len(s[0]) == len(s[1]) == len(s[2]) >= 1
            assert all(b >= a for a, b in zip(s[2], s[2][1:]))           # время не убывает
        try:
            validate({**r, "key_id": r["key_id"].replace("syn-", "syn_")}, {"rude"})
        except Reject as e:  # pragma: no cover
            raise AssertionError(f"{r['key_id']}: {e}")
    json.dumps(recs)                                                      # сериализуемо


def test_reproducible_with_seed():
    a, b, c = synth.generate(20, seed=11), synth.generate(20, seed=11), synth.generate(20, seed=12)
    assert _strip(a) == _strip(b)
    assert _strip(a) != _strip(c)


def test_all_styles_render():
    import numpy as np

    for i, st in enumerate(synth.STYLES):
        rng = np.random.default_rng(i)
        P = synth.sample_params(rng, style=st)
        P["size"] = synth.sample_size(rng, P, synth.DEFAULT_LOW)
        d = synth.render(P, rng, synth.DEFAULT_LOW)
        assert d and normalize(d)


def test_calibration_round_trip():
    base = synth.generate(80, seed=3)
    cal = synth.calibrate([r["drawing"] for r in base])
    assert set(cal["marginals"]) == set(synth.FEATURES)
    json.dumps(cal)
    recs = synth.generate(30, seed=4, calibration=cal, pool_factor=2)
    assert len(recs) == 30 and all(r["generator"]["calibrated"] for r in recs)


def test_human_split_blocks_and_authors():
    from prepare import human_split

    one = [(f"k{i}", {"author": "аноним", "timestamp": f"2026-09-27T20:{i // 60:02d}:{i % 60:02d}"}) for i in range(200)]
    sp = human_split(one, {"test": 25, "val": 15, "block": 8})
    assert set(sp) == {k for k, _ in one}
    n = {s: sum(v == s for v in sp.values()) for s in ("train", "val", "test")}
    assert 40 <= n["test"] <= 60 and 20 <= n["val"] <= 40 and n["train"] > 100
    # блоки по 8 подряд целиком в одном split
    order = [k for k, _ in one]
    for b in range(0, 200, 8):
        assert len({sp[k] for k in order[b:b + 8]}) == 1
    many = [(f"a{i}", {"author": f"u{i % 5}"}) for i in range(100)]
    sp2 = human_split(many, {"test": 25, "val": 15})
    for u in range(5):
        assert len({sp2[f"a{i}"] for i in range(100) if i % 5 == u}) == 1   # автор целиком в одном split
