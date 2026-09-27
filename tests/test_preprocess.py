"""Каноническая предобработка против эталонных векторов tests/vectors/."""
import json
from pathlib import Path

import numpy as np
import pytest

from doodle import preprocess as pp

VEC = Path(__file__).parent / "vectors"
CASES = sorted(VEC.glob("*.json"))


def test_vectors_exist():
    assert len(CASES) >= 10


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_vector(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    prm = d["params"]
    simp = pp.normalize(d["input"], prm["epsilon"])
    assert simp == d["simplified"]
    img = pp.rasterize(simp, prm["size"], prm["pad"], prm["line_width"])
    exp = np.load(path.with_suffix(".npy"))
    assert img.dtype == np.float32 and img.shape == (prm["size"], prm["size"])
    np.testing.assert_allclose(img, exp, atol=1e-6)
    # preprocess() целиком = normalize → rasterize
    full = pp.preprocess(d["input"], prm["size"], prm["pad"], prm["line_width"], prm["epsilon"])
    np.testing.assert_allclose(full, exp, atol=1e-6)


def test_texture_vector_matches_game_conversion():
    d = json.loads((VEC / "game_texture_banana.json").read_text(encoding="utf-8"))
    tex = d["texture"]
    assert pp.from_texture_strokes(tex["strokes"], tex["height"]) == d["input"]


def test_normalized_range_and_ints():
    rng = np.random.default_rng(0)
    for _ in range(50):
        strokes = [[(rng.random(20) * 500 - 100).tolist(), (rng.random(20) * 300).tolist()] for _ in range(3)]
        s = pp.normalize(strokes)
        xs = [v for st in s for v in st[0]]
        ys = [v for st in s for v in st[1]]
        assert all(isinstance(v, int) for v in xs + ys)
        assert min(xs) == 0 and min(ys) == 0
        assert max(max(xs), max(ys)) == 255


def test_translation_and_scale_invariance():
    base = [[[10, 60, 90, 40], [5, 80, 20, 60]], [[30, 35], [30, 90]]]
    a = pp.preprocess(base)
    moved = [[[x * 3.0 + 1000 for x in s[0]], [y * 3.0 - 77 for y in s[1]]] for s in base]
    np.testing.assert_allclose(pp.preprocess(moved), a, atol=1e-6)


def test_timestamps_ignored():
    s = [[[0, 10, 20], [0, 15, 5], [0, 16, 33]]]
    np.testing.assert_array_equal(pp.preprocess(s), pp.preprocess([st[:2] for st in s]))


def test_resample_is_noop_for_rdp():
    """«Resample 1 px» из конвейера Google не меняет результат RDP (поэтому эталон его не делает)."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        n = int(rng.integers(3, 40))
        p = np.cumsum(rng.normal(0, 12, (n, 2)), axis=0)
        p = pp.dedupe(p)
        a = pp.round_half_up(pp.rdp(p, 2.0))
        b = pp.round_half_up(pp.rdp(pp.resample(p, 1.0), 2.0))
        np.testing.assert_array_equal(a, b)


def test_rdp_keeps_endpoints_and_corners():
    p = np.array([[0, 0], [50, 0.5], [100, 0], [100, 100]], dtype=float)
    r = pp.rdp(p, 2.0)
    np.testing.assert_array_equal(r, [[0, 0], [100, 0], [100, 100]])


def test_png_bytes():
    png = pp.to_png_bytes(np.eye(8, dtype=np.float32), scale=2)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 50
