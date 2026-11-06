"""Shared OCR name resolution for neural decisions and recorded identities."""
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata


GLYPHS = str.maketrans({"増": "增", "壊": "坏"})


def clean_name(value):
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).translate(GLYPHS)
    return re.sub(r"\W", "", text).replace("ー", "")


@dataclass(frozen=True)
class NameMatch:
    items: tuple
    method: str
    similarity: float


def match_name(name, aliases, *, role=None, rarity=None, preferred=None):
    """Prefer normalized aliases; otherwise take the closest in the party/role.

    Similarity is a string comparison score, not an identity probability.
    Shared names keep their multiple roles for the portrait reader to resolve.
    """
    key = clean_name(name)
    if not key:
        return NameMatch((), "unresolved", 0.)

    def in_role(items):
        return tuple(item for item in items if role is None or item["role"] == role)

    def by_rarity(items):
        if not rarity:
            return items
        return tuple(item for item in items if item["rarity_class"] == rarity) or items

    exact = in_role((preferred or {}).get(key, ())) or in_role(aliases.get(key, ()))
    if exact:
        return NameMatch(by_rarity(exact), "normalized_name", 1.)

    candidates = [(alias, in_role(items)) for alias, items in aliases.items()]
    candidates = [(alias, items) for alias, items in candidates if items]
    if rarity:
        same_rarity = [(alias, tuple(item for item in items if item["rarity_class"] == rarity))
                       for alias, items in candidates]
        candidates = [(alias, items) for alias, items in same_rarity if items] or candidates
    ratio, items = max(((SequenceMatcher(None, key, alias).ratio(), items) for alias, items in candidates),
                       key=lambda row: row[0], default=(0., ()))
    return NameMatch(items, "closest_name" if items else "unresolved", ratio)
