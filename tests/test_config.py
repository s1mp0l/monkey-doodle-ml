"""Конфиг классов и список заданий для Unity."""
import lesson_prompts
from doodle.config import load_config


def test_config_loads_and_profiles():
    full = load_config()
    ids = full.class_ids
    assert "banana" in ids and "other" in ids and "rude" in ids
    assert 25 <= sum(c.task for c in full.classes) <= 35
    smoke = load_config(profile="smoke")
    assert set(smoke.class_ids) <= set(ids)
    other = smoke.by_id("other")
    assert len(other.quickdraw) == 10 and other.cap == 1000


def test_tasks_have_accusative_and_single_qd_source():
    for c in load_config().classes:
        if c.task:
            assert c.ru_acc and len(c.quickdraw) == 1
            assert c.id == c.quickdraw[0].replace(" ", "_")


def test_other_does_not_overlap_targets():
    cfg = load_config()
    targets = {q for c in cfg.classes if c.kind != "other" for q in c.quickdraw}
    assert not targets & set(cfg.by_id("other").quickdraw)


def test_lesson_prompts_json_in_sync():
    assert lesson_prompts.OUT.read_text(encoding="utf-8") == lesson_prompts.build()
