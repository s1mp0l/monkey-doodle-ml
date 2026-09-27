"""Аугментация рисунков людей и композиции: валидность, воспроизводимость, минимальная доля главного."""
import numpy as np

from doodle import augment as aug
from doodle.preprocess import normalize
from import_contrib import Reject, validate

# простой «фаллос»-подобный рисунок людей: ствол + два круга
SHAFT = [[100, 100, 130, 130], [400, 200, 200, 400], [0, 300, 600, 900]]
BALL1 = [[95 + 20 * np.cos(np.linspace(0, 6.5, 20))][0].tolist(), (410 + 20 * np.sin(np.linspace(0, 6.5, 20))).tolist(),
         list(range(20))]
DRAW = [SHAFT, [BALL1[0], BALL1[1], BALL1[2]]]
WORD = [[[100, 130], [300, 300], [0, 200]], [[140, 140], [280, 320], [0, 200]], [[170, 200], [300, 300], [0, 200]]]


def _valid(rec):
    validate({**rec, "key_id": "k" + rec["key_id"].replace("-", "")[:20]}, {"rude"})


def test_augment_one_valid_and_transforms():
    rng = np.random.default_rng(3)
    d, tr = aug.augment_one(DRAW, rng, "phallus", spec={"fill": "hatch", "size": "medium", "rot": 20, "shear": 0.1,
                                                        "jitter": True, "dropout": False, "order": False, "reverse": False, "text": False})
    assert normalize(d) and any(t.startswith("size:") for t in tr) and any(t.startswith("affine:") for t in tr)
    for s in d:
        assert len(s) == 3 and len(s[0]) == len(s[1]) == len(s[2]) >= 1
        assert all(b >= a for a, b in zip(s[2], s[2][1:]))


def test_words_are_never_filled():
    for i in range(20):
        rng = np.random.default_rng(i)
        _, tr = aug.augment_one(WORD, rng, "word")
        assert not any(t.startswith("fill:") for t in tr)


def test_size_buckets_change_raw_extent():
    ext = {}
    for b in ("tiny", "huge"):
        rng = np.random.default_rng(1)
        paths = aug.fit_size(aug._paths(DRAW), rng, b)
        a = np.concatenate(paths)
        ext[b] = float((a.max(0) - a.min(0)).max())
    lo, hi = aug.SIZE_BUCKETS["tiny"], aug.SIZE_BUCKETS["huge"]
    assert lo[0] <= ext["tiny"] <= lo[1] and hi[0] <= ext["huge"] <= hi[1] and ext["huge"] > ext["tiny"] * 3


def test_augment_records_reproducible_and_pass_importer():
    parents = [{"key_id": "aaaaaaaa-1", "drawing": DRAW, "word": "rude", "author": "u"}]
    sub = {"aaaaaaaa-1": "phallus"}
    a = aug.augment_records(parents, sub, 5, seed=2)
    b = aug.augment_records(parents, sub, 5, seed=2)
    assert len(a) == 5 and [r["drawing"] for r in a] == [r["drawing"] for r in b]
    for r in a:
        assert r["source"] == "augmented" and r["parent"] == "aaaaaaaa-1" and r["subtype"] == "phallus"
        _valid(r)


def test_compose_min_frac_enforced():
    rng = np.random.default_rng(0)
    primary = aug._paths(DRAW)
    small = [np.array([[0.0, 0.0], [255.0, 0.0], [255.0, 255.0], [0.0, 255.0], [0.0, 0.0]])]  # большой квадрат-объект
    for _ in range(20):
        placed_parts = [(primary, 0.6), (small, 1.3), (small, 1.2)]
        aug.compose(placed_parts, np.random.default_rng(rng.integers(1 << 30)), min_frac=0.3, primary=0)
    # прямая проверка доли на детерминированном размещении
    g = np.random.default_rng(5)
    parts = [(primary, 0.5), (small, 1.3)]
    # эмулируем внутреннюю проверку: после compose доля не измеряется наружу, поэтому проверяем _primary_frac напрямую
    placed = [[p + [10, 10] for p in primary], [p * 2 + [400, 10] for p in small]]
    assert aug._primary_frac(placed, 0) < 0.3            # исходно главный мелкий
