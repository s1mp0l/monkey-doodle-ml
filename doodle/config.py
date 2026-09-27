"""Загрузка config/classes.yaml и разрешение профилей (full / smoke / ...)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "classes.yaml"


@dataclass
class ClassSpec:
    id: str
    ru: str
    kind: str = "quickdraw"          # quickdraw | other | custom
    quickdraw: list[str] = field(default_factory=list)
    cap: int = 20000                 # на каждый источник Quick Draw
    task: bool = False               # задание урока рисования
    ru_acc: str | None = None        # винительный падеж: «Нарисуйте: …»
    sampling_weight: float = 1.0
    augment: str = "normal"
    target_precision: float = 0.9
    min_samples: int = 0
    target: int | None = None
    hard_negatives: list[str] = field(default_factory=list)   # other: «трудные» источники
    hard_cap: int = 0
    human_split: dict | None = None      # {test, val, block} — отдельное деление рисунков людей
    synthetic: dict | None = None        # {file, max_share}
    augmented: dict | None = None        # {file} — офлайн-аугментация рисунков людей (train, split родителя)
    composites: dict | None = None       # {rude, other} — файлы композиций по целевым классам
    subtype_balance: dict | None = None  # {target} — предел синтетика+аугментация на подтип

    @property
    def sources(self) -> list[str]:
        return self.quickdraw + [q for q in self.hard_negatives if q not in self.quickdraw]

    def cap_for(self, source: str) -> int:
        return self.hard_cap if source in self.hard_negatives and self.hard_cap else self.cap


@dataclass
class Config:
    path: Path
    raw: dict
    profile: str
    classes: list[ClassSpec]
    preprocess: dict
    split: dict
    custom_dir: Path
    custom_repeat: int
    source_loss_weights: dict

    @property
    def class_ids(self) -> list[str]:
        return [c.id for c in self.classes]

    def by_id(self, cid: str) -> ClassSpec:
        for c in self.classes:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def quickdraw_sources(self) -> list[tuple[str, int]]:
        """Все пары (имя Quick Draw, лимит) для скачивания, без повторов (берётся больший лимит)."""
        caps: dict[str, int] = {}
        for c in self.classes:
            for q in c.sources:
                caps[q] = max(caps.get(q, 0), c.cap_for(q))
        return sorted(caps.items())

    def lesson_prompts(self) -> list[dict]:
        """Список заданий урока (task: true) — формат config/lesson_prompts.json."""
        return [{"id": c.id, "quickdraw": c.quickdraw[0], "ru": c.ru, "ru_acc": c.ru_acc or c.ru.lower()}
                for c in self.classes if c.task]

    @property
    def editor(self) -> dict:
        return self.raw.get("editor", {})


def load_config(path: str | os.PathLike | None = None, profile: str | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    d = raw.get("defaults", {})
    prof = (raw.get("profiles") or {}).get(profile) if profile and profile != "full" else None
    if profile and profile != "full" and prof is None:
        raise SystemExit(f"Профиль '{profile}' не найден в {path}")

    classes: list[ClassSpec] = []
    for c in raw["classes"]:
        kind = c.get("kind", "quickdraw")
        spec = ClassSpec(
            id=c["id"], ru=c["ru"], kind=kind,
            quickdraw=list(c.get("quickdraw") or []),
            cap=int(c.get("cap", d.get("cap", 20000))),
            task=bool(c.get("task", False)),
            ru_acc=c.get("ru_acc"),
            sampling_weight=float(c.get("sampling_weight", d.get("sampling_weight", 1.0))),
            augment=c.get("augment", d.get("augment", "normal")),
            target_precision=float(c.get("target_precision", d.get("target_precision", 0.9))),
            min_samples=int(c.get("min_samples", 0)),
            target=c.get("target"),
            hard_negatives=list(c.get("hard_negatives") or []),
            hard_cap=int(c.get("hard_cap", 0)),
            human_split=c.get("human_split"),
            synthetic=c.get("synthetic"),
            augmented=c.get("augmented"),
            composites=c.get("composites"),
            subtype_balance=c.get("subtype_balance"),
        )
        classes.append(spec)

    if prof:
        keep = prof.get("classes")
        if keep:
            missing = set(keep) - {c.id for c in classes}
            if missing:
                raise SystemExit(f"Профиль '{profile}': неизвестные классы {sorted(missing)}")
            classes = [c for c in classes if c.id in keep]
        for c in classes:
            if c.kind == "other":
                if "other_sources" in prof:
                    c.quickdraw = c.quickdraw[: int(prof["other_sources"])]
                if "other_cap" in prof:
                    c.cap = int(prof["other_cap"])
                if "other_hard_cap" in prof:
                    c.hard_cap = int(prof["other_hard_cap"])
            elif "cap" in prof:
                c.cap = int(prof["cap"])

    ids = [c.id for c in classes]
    if len(ids) != len(set(ids)):
        raise SystemExit("Повторяющиеся id классов в конфиге")
    custom = raw.get("custom", {})
    return Config(
        path=path, raw=raw, profile=profile or "full", classes=classes,
        preprocess=dict(raw.get("preprocess", {})),
        split=dict(d.get("split", {"val": 5, "test": 5})),
        custom_dir=ROOT / custom.get("dir", "data/custom"),
        custom_repeat=int(custom.get("repeat", 1)),
        source_loss_weights={"quickdraw": 1.0, "human": 1.0, "synthetic": 1.0, **(raw.get("source_loss_weights") or {})},
    )
