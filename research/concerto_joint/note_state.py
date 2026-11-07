"""Public note state and deterministic purchases for a future joint critic.

This is a rules/data foundation, not a full episode simulator or trained model.
Requirements come from extracted per-skill rows, never from note-count heuristics.
Only visible, explicitly specified goods are supported; no shelves are invented.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
TABLE = Path(__file__).resolve().parent / "data/note_system.json"
NOTE_IDS = tuple(range(90011, 90024))
FEATURE_VERSION = "concerto_public_context_v0"


def natural(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a known nonnegative integer")
    return value


@dataclass(frozen=True)
class Skill:
    group_id: int
    requirements: tuple[tuple[tuple[int, int], ...], ...]

    def level(self, counts: Mapping[int, int]) -> int:
        result = 0
        for level, requirement in enumerate(self.requirements, 1):
            if not all(counts[note] >= quantity for note, quantity in requirement):
                break
            result = level
        return result

    def next_deficit(self, counts: Mapping[int, int]) -> tuple[int, ...]:
        level = self.level(counts)
        if level == len(self.requirements):
            return (0,) * len(NOTE_IDS)
        need = dict(self.requirements[level])
        return tuple(max(0, need.get(note, 0) - counts[note]) for note in NOTE_IDS)


def load_catalog(path=TABLE) -> dict[int, Skill]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = {}
    for row in data["secondary_skills"]:
        if row["group"] is None or row["level"] is None:
            continue  # Ungrouped special skill has no ordinary level sequence.
        group, level = int(row["group"]), int(row["level"])
        requirements = tuple(sorted((int(k), v) for k, v in row["need"].items()))
        if not requirements or any(n not in NOTE_IDS or natural(q, "requirement") == 0
                                   for n, q in requirements):
            raise ValueError(f"Unsupported requirements in group {group}")
        if level in groups.setdefault(group, {}):
            raise ValueError(f"Duplicate level in group {group}")
        groups[group][level] = requirements
    catalog = {}
    for group, levels in groups.items():
        if sorted(levels) != list(range(1, len(levels) + 1)) or len(levels) > 5:
            raise ValueError(f"Noncontiguous levels in group {group}")
        previous = {}
        for level in sorted(levels):
            current = dict(levels[level])
            if previous and (current.keys() != previous.keys() or
                             any(current[n] < previous[n] for n in previous)):
                raise ValueError(f"Nonmonotonic requirements in group {group}")
            previous = current
        catalog[group] = Skill(group, tuple(levels[i] for i in sorted(levels)))
    return catalog


@dataclass(frozen=True)
class NoteState:
    counts: tuple[int, ...]
    coins: int
    floors_left: int
    swaps_left: int
    skills: tuple[Skill, ...]

    def __post_init__(self):
        if len(self.counts) != len(NOTE_IDS):
            raise ValueError("All 13 note counts must be explicitly known")
        for count in self.counts:
            natural(count, "note count")
        natural(self.coins, "coins")
        natural(self.floors_left, "floors_left")
        natural(self.swaps_left, "swaps_left")
        if self.floors_left > 20 or self.swaps_left > 15:
            raise ValueError("Outside the current tower-8 context")
        if not 1 <= len(self.skills) <= 6:
            raise ValueError("One to six explicitly selected skills required")
        if len({s.group_id for s in self.skills}) != len(self.skills):
            raise ValueError("Duplicate skills require explicit stacking semantics")

    @classmethod
    def from_counts(cls, counts, **kwargs):
        if set(counts) != set(NOTE_IDS):
            raise ValueError("Unknown notes must not be silently filled with zero")
        return cls(tuple(counts[n] for n in NOTE_IDS), **kwargs)

    def inventory(self):
        return dict(zip(NOTE_IDS, self.counts))

    def levels(self):
        counts = self.inventory()
        return tuple(skill.level(counts) for skill in self.skills)


@dataclass(frozen=True)
class VisibleNoteOffer:
    item_id: str
    price: int
    contents: tuple[tuple[int, int], ...]
    stock: int = 1

    def __post_init__(self):
        if not self.item_id:
            raise ValueError("Visible item identity required")
        natural(self.price, "price")
        natural(self.stock, "stock")
        if not self.contents or len(dict(self.contents)) != len(self.contents):
            raise ValueError("Known, nonempty, unambiguous item contents required")
        for note, quantity in self.contents:
            if note not in NOTE_IDS or natural(quantity, "quantity") == 0:
                raise ValueError("Unknown/random goods need a separate transition model")


def purchase(state: NoteState, offer: VisibleNoteOffer):
    if offer.stock == 0 or state.coins < offer.price:
        raise ValueError("Purchase is not legal for the observed wallet/stock")
    counts = state.inventory()
    for note, quantity in offer.contents:
        counts[note] += quantity
    return (replace(state, counts=tuple(counts[n] for n in NOTE_IDS),
                    coins=state.coins - offer.price),
            replace(offer, stock=offer.stock - 1))


def encode_context(state: NoteState) -> tuple[float, ...]:
    """Fixed 496 dimensions: context appended to a separately encoded action.

    Each of six skill slots has a mask, five levels of 13 requirements,
    current level, and 13 next-level deficits. No RNG or future outcome enters.
    A full learner must ALSO encode the action and all visible shelf items.
    """
    counts = state.inventory()
    features = [count / 75 for count in state.counts]
    features.extend((state.coins / 2000, state.floors_left / 20, state.swaps_left / 15))
    for slot in range(6):
        if slot >= len(state.skills):
            features.extend([0.] * 80)
            continue
        skill = state.skills[slot]
        features.append(1.)
        for level in range(5):
            requirements = dict(skill.requirements[level]) if level < len(skill.requirements) else {}
            features.extend(requirements.get(note, 0) / 75 for note in NOTE_IDS)
        features.append(skill.level(counts) / 5)
        features.extend(value / 75 for value in skill.next_deficit(counts))
    return tuple(features)


def outcome_vector(ordinary_score: float, state: NoteState) -> dict:
    """Keep objectives separate; concerto levels are not a damage measurement."""
    if not 0 <= ordinary_score < float("inf"):
        raise ValueError("Invalid ordinary potential score")
    levels = state.levels()
    return {"ordinary_potential": ordinary_score, "concerto_levels": list(levels),
            "concerto_level_sum": sum(levels), "coins": state.coins}
