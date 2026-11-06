"""Full ordered 36-slot preset contracts and reproducible in-scope sampling.

All sums below concern non-core target levels; six fixed core levels never
enter the preset, score, or sampling distribution. No retained-pool shortcut
is used to determine whether a training/evaluation example is in scope.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from random import Random
from typing import Iterable

POOL_LENGTHS = (3, 9, 3, 9, 3, 9)
BUCKETS = ((72, 95), (96, 119), (120, 144))
BUCKET_NAMES = ("72–95", "96–119", "120–144")
STYLES = ("mixed", "concentrated", "balanced", "role_skew", "rarity_skew")
STYLE_WEIGHTS = (0.35, 0.25, 0.15, 0.15, 0.10)
ROLE_TARGET_RANGE = (18, 54)
PRESET_VERSION = "full36_role18_54_total72_144_v2"


def validate_role_totals(totals) -> tuple:
    totals = tuple(totals)
    if len(totals) != 3:
        raise ValueError("预设须包含三名角色的原始普通目标总级数。")
    for role, total in enumerate(totals, 1):
        if not ROLE_TARGET_RANGE[0] <= total <= ROLE_TARGET_RANGE[1]:
            raise ValueError(f"角色{role}的原始普通目标总级数 {total} 不在18–54；不包括核心。")
    return totals


def canonical_seed(*parts) -> int:
    """Domain-separated seed; never incorporates model IDs or checkpoints."""
    payload = json.dumps(parts, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def validate_pools(pools: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    ordered = tuple(tuple(row) for row in pools)
    if tuple(map(len, ordered)) != POOL_LENGTHS:
        raise ValueError("预设必须包含完整 36 个普通槽位，六池长度须为 3/9/3/9/3/9；保留零目标。")
    if any(type(v) is not int or not 0 <= v <= 6 for row in ordered for v in row):
        raise ValueError("每个普通目标等级必须是 0–6 的整数。")
    total = sum(map(sum, ordered))
    if not 72 <= total <= 144:
        raise ValueError(f"普通目标总级数 {total} 不在 72–144；不包括核心。")
    validate_role_totals(sum(ordered[2*r]) + sum(ordered[2*r+1]) for r in range(3))
    return ordered


@dataclass(frozen=True, slots=True)
class Preset:
    name: str
    pools: tuple
    group: str = "custom"

    def __post_init__(self):
        object.__setattr__(self, "pools", validate_pools(self.pools))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Preset name must be nonempty")

    @property
    def total(self) -> int:
        return sum(map(sum, self.pools))

    @property
    def cap(self) -> int:
        return self.total

    @property
    def role_totals(self) -> tuple[int, int, int]:
        return tuple(sum(self.pools[2*r]) + sum(self.pools[2*r+1]) for r in range(3))

    @property
    def flat(self) -> tuple[int, ...]:
        return tuple(v for row in self.pools for v in row)

    @property
    def bucket(self) -> int:
        return next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= self.total <= hi)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.pools, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {"name": self.name, "pools": [list(p) for p in self.pools],
                "group": self.group, "total": self.total,
                "role_totals": list(self.role_totals), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: dict) -> "Preset":
        preset = cls(str(value["name"]), tuple(tuple(p) for p in value["pools"]), value.get("group", "custom"))
        if "total" in value and value["total"] != preset.total:
            raise ValueError("Preset total disagrees with the full target vector")
        if "role_totals" in value and tuple(value["role_totals"]) != preset.role_totals:
            raise ValueError("Preset role totals disagree with the full target vector")
        if "fingerprint" in value and value["fingerprint"] != preset.fingerprint:
            raise ValueError("Preset fingerprint mismatch")
        return preset


PRESET_843 = Preset("843 / 90", ((6,0,6),(6,6,0,0,6,6,6,0,6),
                               (6,0,6),(6,6,0,0,0,0,0,0,0),
                               (6,0,0),(0,6,6,0,0,0,0,0,0)), "anchor_843")
PRESET_JP97 = Preset("JP97", ((6,0,6),(6,6,0,0,3,0,0,3,3),
                             (6,6,6),(6,0,0,6,6,0,0,0,0),
                             (1,0,6),(6,0,0,6,3,0,0,6,0)), "anchor_jp97")
ANCHORS = (PRESET_843, PRESET_JP97)


def _pools(flat: list[int]) -> tuple:
    rows, offset = [], 0
    for n in POOL_LENGTHS:
        rows.append(tuple(flat[offset:offset+n]))
        offset += n
    return tuple(rows)


def _constrain_roles(flat: list[int], rng: Random) -> None:
    """Transfer levels between roles, preserving total and every 0..6 slot.

    Keep already-valid shapes untouched. Fix violations in finite positive
    transfers rather than rejecting entire presets, so chosen total/style
    probabilities remain unchanged even for strongly role-skewed proposals.
    """
    low, high = ROLE_TARGET_RANGE
    totals = [sum(flat[12*r:12*r+12]) for r in range(3)]
    while min(totals) < low or max(totals) > high:
        donors = [r for r in range(3) if totals[r] > high]
        receivers = [r for r in range(3) if totals[r] < low]
        donor = rng.choice(donors or [r for r in range(3) if totals[r] > low])
        receiver = rng.choice(receivers or [r for r in range(3) if totals[r] < high])
        source = rng.choice([i for i in range(12*donor, 12*donor+12) if flat[i] > 0])
        target = rng.choice([i for i in range(12*receiver, 12*receiver+12) if flat[i] < 6])
        amount = min(flat[source], 6-flat[target],
                     totals[donor]-(high if donors else low),
                     (low if receivers else high)-totals[receiver])
        flat[source] -= amount
        flat[target] += amount
        totals[donor] -= amount
        totals[receiver] += amount


def sample_preset(rng: Random, bucket: int | None = None, *, name: str | None = None,
                  style: str | None = None, total: int | None = None) -> Preset:
    """Three target-total strata have equal probability by construction.

    Total is uniform within the selected stratum. Five shape families use fixed
    weights 35/25/15/15/10%. Each role has 18..54 original target levels.
    Known historical anchors are deliberately outside this procedural sampler.
    """
    if bucket is None:
        if total is None:
            bucket = rng.randrange(3)
        else:
            bucket = next((i for i, (lo, hi) in enumerate(BUCKETS) if lo <= total <= hi), -1)
    if bucket not in range(3):
        raise ValueError("bucket must be 0, 1, or 2")
    lo, hi = BUCKETS[bucket]
    if total is None:
        total = rng.randint(lo, hi)
    if type(total) is not int or not lo <= total <= hi:
        raise ValueError("Requested total is outside its stratum")
    style = style or rng.choices(STYLES, weights=STYLE_WEIGHTS, k=1)[0]
    if style not in STYLES:
        raise ValueError("Unknown preset style")
    flat = [0] * 36
    if style == "concentrated":
        order = list(range(36)); rng.shuffle(order)
        remaining = total
        for index in order:
            flat[index] = min(6, remaining)
            remaining -= flat[index]
    elif style == "balanced":
        flat = [total // 36] * 36
        order = list(range(36)); rng.shuffle(order)
        for index in order[:total % 36]:
            flat[index] += 1
        # Preserve the exact total while allowing nonidentical target levels.
        for _ in range(24):
            a, b = rng.sample(range(36), 2)
            if flat[a] > 0 and flat[b] < 6:
                flat[a] -= 1; flat[b] += 1
    elif style == "mixed":
        flat = [rng.randint(0, 6) for _ in range(36)]
        while sum(flat) != total:
            direction = 1 if sum(flat) < total else -1
            choices = [i for i, value in enumerate(flat) if (value < 6 if direction > 0 else value > 0)]
            flat[rng.choice(choices)] += direction
    else:
        favored_role, favor_rare = rng.randrange(3), rng.choice((True, False))
        weights = []
        for index in range(36):
            role = index // 12
            rare = index % 12 < 3
            favored = role == favored_role if style == "role_skew" else rare == favor_rare
            weights.append(6.0 if favored else 0.5)
        for _ in range(total):
            choices = [i for i, value in enumerate(flat) if value < 6]
            index = rng.choices(choices, weights=[weights[i] for i in choices], k=1)[0]
            flat[index] += 1
    _constrain_roles(flat, rng)
    short = hashlib.blake2b(bytes(flat), digest_size=5).hexdigest()
    return Preset(name or f"{BUCKET_NAMES[bucket]}_{style}_{short}", _pools(flat), style)


def evaluation_presets(seed: int = 20261026, per_bucket: int = 4,
                       include_anchors: bool = True) -> list[Preset]:
    """Frozen display panel. Rotating promotion panels should use fresh seeds.

    The first and last bucket include the exact 72 and 144 boundaries; anchors
    are returned last and should be reported separately from main statistics.
    """
    if per_bucket < 1:
        raise ValueError("At least one preset per stratum is required")
    rng = Random(canonical_seed("evaluation_preset_panel", seed))
    result, occupied = [], set()
    for bucket in range(3):
        for j in range(per_bucket):
            total = 72 if bucket == 0 and j == 0 else 144 if bucket == 2 and j == 0 else None
            for _ in range(1000):
                preset = sample_preset(rng, bucket, name=f"E{bucket+1}-{j+1:02d}", total=total)
                if preset.fingerprint not in occupied:
                    break
            else:
                raise RuntimeError("Unique evaluation preset generation failed")
            occupied.add(preset.fingerprint)
            result.append(preset)
    if include_anchors:
        result.extend(ANCHORS)
    return result
