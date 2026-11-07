"""Versioned offline joint episode. Assumptions are explicit in RULES below."""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from functools import lru_cache
import hashlib
import json
from math import comb
from random import Random

from bootstrap import HERE, ROOT
from note_state import NOTE_IDS, NoteState, VisibleNoteOffer, load_catalog, purchase
from price_data import PRICE_SCHEMA, validate_note_goods
from maa_value_plugin.critic.environment import Episode, sim, legal_actions
from maa_value_plugin.critic.presets import Preset, canonical_seed, sample_preset

RULES = {
    "version": "joint_offline_v3_environment_fixed",
    "note_prices": "Quantity-specific legal prices; confirmed 9045->45 and 9072->72 OCR concatenations repaired before aggregation; unknown values fail validation",
    "ordinary_kernel": sim.MODEL_VERSION,
    "ordinary_route": "unchanged frozen 45-node route",
    "shelves": "Keep kernel drinks; sample remaining discount tiers from the exact conditional four-of-eight shelf law (at most two half-price slots), then empirical marginal note sizes; middle initial shops retain the kernel half-price prior",
    "note_product_identity": "Uniform among tower drop-group notes, independent of selected skills",
    "battle_notes": "Eight individual uniform draws per battle by default; uncalibrated, evaluate sensitivity",
    "initial_supply": "Actual local wiki per-sigil supportSupply, not a blanket eight per type",
    "note_events": "Client-table effects with uniform random-note draws; generic fixed-note IDs 10701..10707; any-note costs use a deterministic least-loss donor",
    "visibility": "All modeled note options exposed for 107/108/114; actual server option sampling is unknown",
    "event_legality": "All visible table options, without ordinary-objective dominance pruning; coin costs, every positive-mass wallet outcome and note costs checked before selection",
    "adjustments": "At terminal, at most 15 one-for-one transfers between any owned donor and a required target; legality/timing need game evidence",
    "objective": "Separate ordinary/concerto returns; lambda is an experimental preference, not measured damage",
    "scope": "Offline research only, no Maa or live-game integration",
}


class JointKernelEpisode(Episode):
    """Change event legality locally; retain the frozen kernel's transitions.

    Episode.step validates against this property as well, so a newly exposed
    option is executable, not merely offered by the outer adapter.
    """
    @property
    def actions(self):
        state = self.state
        if state.phase != "EVENT":
            return super().actions
        if state.eventId not in sim.EVENT_DATA:
            raise sim.ModelGap("UNKNOWN_EVENT", state.eventId)
        options = sim.EVENT_DATA[state.eventId][2]
        out = []
        for oid in sorted(set(state.eventOptions)):
            if oid not in options:
                raise sim.ModelGap("UNKNOWN_EVENT_OPTION", oid)
            cost, outcomes = options[oid]
            if state.coins < cost or any(n and state.coins + delta < 0
                                         for n, _d, delta, _kind, _scope in outcomes):
                continue
            out.append(sim.Action("SKIP_EVENT", type=state.eventId) if oid == 0
                       else sim.Action("EVENT_OPTION", oid, cost, state.eventId))
        if not out:
            raise sim.ModelGap("NO_COVERED_ACTION", state.eventId)
        return tuple(out)


def remaining_discount_tiers(stock, rng, *, middle_initial=False):
    """Condition the reference eight-slot law on the already visible drinks.

    Four slots are discounted. H=min(2, Binomial(3,p)) of those are half
    price. Given h half-price and m 20%-off drinks, posterior weight for H
    is prior(H)*C(H,h)*C(4-H,m); all omitted factors are constant in H.
    This preserves the drink marginal instead of resampling or repricing it.
    """
    h, m, full = (sum(row[i] for row in stock) for i in range(3))
    p = .75 if middle_initial else .7
    prior = ((1-p)**3, 3*p*(1-p)**2, 3*p*p*(1-p) + p**3)
    possible, weights = [], []
    for total_half in range(3):
        if h <= total_half and m <= 4-total_half and full <= 4:
            possible.append(total_half)
            weights.append(prior[total_half] * comb(total_half, h) * comb(4-total_half, m))
    if not possible or sum(weights) <= 0:
        raise ValueError("Drink stock is incompatible with the joint shelf law")
    total_half = rng.choices(possible, weights=weights)[0]
    tiers = [0]*(total_half-h) + [1]*(4-total_half-m) + [2]*(4-full)
    rng.shuffle(tiers)
    return tiers


@lru_cache(maxsize=1)
def inputs():
    path = HERE / "data/inputs.json"
    if not path.exists():
        raise FileNotFoundError("Frozen research inputs are missing: " + str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("price_schema") != PRICE_SCHEMA:
        raise ValueError("Unvalidated note-price data; regenerate inputs with repair_prices.py")
    validate_note_goods(data["note_goods"])
    return data


@lru_cache(maxsize=1)
def catalog():
    return load_catalog()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def scenario(main, sub, tower, preset, name="custom"):
    resolved = inputs()["sigils"]
    groups, supply = [], dict.fromkeys(NOTE_IDS, 0)
    for name_ in main:
        groups.extend(resolved[name_]["groups"])
    for name_ in sub:
        for note, quantity in resolved[name_]["supply"].items():
            supply[int(note)] += quantity
    value = {"name": name, "main_sigils": list(main), "sub_sigils": list(sub),
             "tower": tower, "preset": preset.to_dict(), "groups": groups,
             "supply": list(supply.values())}
    value["fingerprint"] = fingerprint({k: v for k, v in value.items() if k != "name"})
    return value


def random_scenarios(count, seed):
    rng, result = Random(seed), []
    names = sorted(inputs()["sigils"])
    for i in range(count):
        tower = rng.randrange(1, 4)
        element = {1: {"光", "地"}, 2: {"水", "风"}, 3: {"火", "暗"}}[tower]
        eligible = [name for name in names if inputs()["sigils"][name]["element"] in element]
        main = rng.sample(eligible, 3)
        sub = rng.sample(names, 3)
        preset = sample_preset(rng, total=rng.choice((90, 108, 126)))
        result.append(scenario(main, sub, tower, preset, f"random_{seed}_{i}"))
    return result


def reference_scenarios():
    """Anonymous fixed target/sigil fixtures, containing no player records."""
    path = HERE / "data/reference_scenarios.json"
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class JointAction:
    kind: str
    index: int = 0
    donor: int = 0
    target: int = 0
    base: sim.Action | None = None

    def key(self):
        return sim.action_key(self.base) if self.base else f"{self.kind}:{self.index}:{self.donor}:{self.target}"


class JointEpisode:
    def __init__(self, config, seed, drop_per_battle=8, swaps=15, note_events=True):
        self.config = config
        self.preset = Preset.from_dict(config["preset"])
        self.ep = JointKernelEpisode(self.preset, canonical_seed("ordinary", seed))
        self.notes = tuple(config["supply"])
        self.skills = tuple(catalog()[g] for g in config["groups"])
        self.allowed = tuple(range(90011, 90018)) + {1: (90021, 90022), 2: (90018, 90020), 3: (90019, 90023)}[config["tower"]]
        self.drop_per_battle, self.swaps_left = drop_per_battle, swaps
        self.note_events = note_events
        self.note_rng = Random(canonical_seed("drops", seed))
        self.shelf_rng = Random(canonical_seed("shelf", seed))
        self.event_rng = Random(canonical_seed("note_events", seed))
        self.crossed, self.generation, self.core = 0, 0, 0
        self.shelf, self.finished = (), False
        self.note_spent, self.note_purchases, self.swaps_used = 0, 0, 0
        self.steps = 0
        self._sync(new_shelf=self.ep.state.phase == "SHOP")

    @property
    def phase(self):
        return "DONE" if self.finished else "ADJUST" if self.ep.done else self.ep.state.phase

    @property
    def done(self):
        return self.finished

    def note_state(self):
        return NoteState(self.notes, self.ep.state.coins, max(0, 20 - self.ep.state.floor),
                         self.swaps_left, self.skills)

    def levels(self):
        counts = dict(zip(NOTE_IDS, self.notes))
        return tuple(skill.level(counts) for skill in self.skills)

    def _give(self, note, quantity):
        counts = list(self.notes)
        counts[note - 90011] += quantity
        if counts[note - 90011] < 0:
            raise ValueError("Negative note inventory")
        self.notes = tuple(counts)

    def _random_notes(self, quantity, rng):
        for _ in range(quantity):
            self._give(rng.choice(self.allowed), 1)

    def _shelf(self, *, initial=True):
        self.generation += 1
        goods = inputs()["note_goods"]
        sizes = (5, 15)
        weights = [sum(row["observations"] for row in goods if row["quantity"] == size) for size in sizes]
        tiers = remaining_discount_tiers(self.ep.state.stock, self.shelf_rng,
                    middle_initial=initial and self.ep.state.shopOrdinal in (2, 3))
        self.shelf = tuple(VisibleNoteOffer(f"{self.ep.state.shopOrdinal}-{self.generation}-{slot}",
                            {5: (45, 72, 90), 15: (200, 320, 400)}[size][tier],
                            ((self.shelf_rng.choice(self.allowed), size),))
                           for slot, tier in enumerate(tiers)
                           for size in [self.shelf_rng.choices(sizes, weights=weights)[0]])

    def _sync(self, new_shelf=False, *, initial=True):
        limit = min(self.ep.state.node - 1, len(sim.ROUTE))
        for index in range(self.crossed, limit):
            kind = sim.ROUTE[index][0]
            if kind == "BATTLE":
                self._random_notes(self.drop_per_battle, self.note_rng)
            elif kind == "CORE":
                self.core += 1
        self.crossed = max(self.crossed, limit)
        if new_shelf:
            self._shelf(initial=initial)
        if self.note_events and self.ep.state.phase == "EVENT":
            eid = self.ep.state.eventId
            extra = {107: range(10701, 10709), 108: range(10801, 10810), 114: range(11401, 11406)}.get(eid, ())
            options = tuple(sorted(set(self.ep.state.eventOptions) | set(extra)))
            self.ep.state = replace(self.ep.state, eventOptions=options)

    @property
    def actions(self):
        if self.done:
            return ()
        if self.phase == "ADJUST":
            out = [JointAction("FINISH")]
            if self.swaps_left:
                needed = {n for skill in self.skills for requirement in skill.requirements for n, _ in requirement}
                out.extend(JointAction("SWAP", donor=note, target=target)
                           for note, count in zip(NOTE_IDS, self.notes) if count > 0
                           for target in sorted(needed) if target != note)
            return tuple(out)
        out = []
        for action in self.ep.actions:
            if self.note_events and action.kind == "EVENT_OPTION":
                required = {10301: 5, 10401: 10}.get(action.index, 0)
                if sum(self.notes) < required:
                    continue
            out.append(JointAction("BASE", base=action))
        if self.phase == "SHOP":
            out.extend(JointAction("NOTE", index=i) for i, item in enumerate(self.shelf)
                       if item.stock and self.ep.state.coins >= item.price)
        return tuple(out)

    def _consume(self, quantity):
        # Explicit experiment convention: least immediate concerto/progress loss.
        for _ in range(quantity):
            choices = []
            for note, count in zip(NOTE_IDS, self.notes):
                if count:
                    candidate = list(self.notes); candidate[note - 90011] -= 1
                    choices.append((note_progress(tuple(candidate), self.skills), note))
            self._give(max(choices)[1], -1)

    def step(self, action):
        if action not in self.actions:
            raise ValueError("Illegal joint action")
        if self.steps >= 1200:
            raise RuntimeError("Joint episode exceeded decision budget")
        self.steps += 1
        if action.kind == "FINISH":
            self.finished = True
            return
        if action.kind == "SWAP":
            self._give(action.donor, -1); self._give(action.target, 1)
            self.swaps_left -= 1; self.swaps_used += 1
            return
        if action.kind == "NOTE":
            item = self.shelf[action.index]
            after, sold = purchase(self.note_state(), item)
            self.notes = after.counts
            self.ep.state = replace(self.ep.state, coins=after.coins)
            shelf = list(self.shelf); shelf[action.index] = sold; self.shelf = tuple(shelf)
            self.note_spent += item.price; self.note_purchases += 1
            return
        before = self.ep.state
        base = action.base
        if self.note_events and base.kind == "EVENT_OPTION":
            oid = base.index
            if oid == 10301:
                self._consume(5)
            elif oid == 10401:
                self._consume(10)
            elif 10701 <= oid <= 10707:
                self._give(90010 + oid % 100, 5)
            elif oid == 10708 or oid == 12702 or oid in (12001, 12002):
                self._random_notes(5, self.event_rng)
            elif 10801 <= oid <= 10807:
                self._give(90010 + oid % 100, 10)
            elif oid == 10808 or 11401 <= oid <= 11405:
                self._random_notes(10, self.event_rng)
        self.ep.step(base)
        new = self.ep.state
        new_shelf = new.phase == "SHOP" and (new.shopOrdinal != before.shopOrdinal or base.kind == "RESTOCK")
        self._sync(new_shelf, initial=base.kind != "RESTOCK")

    def outcome(self):
        return {"ordinary": sim.score(self.ep.state), "core": self.core,
                "potential_total": sim.score(self.ep.state) + self.core,
                "concerto": sum(self.levels()), "concerto_levels": list(self.levels()),
                "coins": self.ep.state.coins, "notes": list(self.notes),
                "note_spent": self.note_spent, "note_purchases": self.note_purchases,
                "swaps_used": self.swaps_used, "steps": self.steps}

    def snapshot(self):
        return {"rules_version": RULES["version"], "config": self.config,
                "ordinary_state": sim.to_jsonable(self.ep.state), "notes": list(self.notes),
                "shelf": [asdict(item) for item in self.shelf], "crossed": self.crossed,
                "generation": self.generation, "core": self.core, "swaps_left": self.swaps_left,
                "drop_per_battle": self.drop_per_battle, "note_events": self.note_events,
                "note_spent": self.note_spent, "note_purchases": self.note_purchases,
                "swaps_used": self.swaps_used, "steps": self.steps, "finished": self.finished}

    @classmethod
    def from_public(cls, snapshot, seed):
        if snapshot["rules_version"] != RULES["version"]:
            raise ValueError("Wrong joint rule version")
        obj = cls.__new__(cls)
        obj.config = snapshot["config"]
        obj.preset = Preset.from_dict(obj.config["preset"])
        state = sim.state_from_dict(snapshot["ordinary_state"])
        obj.ep = JointKernelEpisode.from_public(state, canonical_seed("ordinary_branch", seed), obj.preset)
        obj.notes = tuple(snapshot["notes"])
        obj.skills = tuple(catalog()[g] for g in obj.config["groups"])
        obj.allowed = tuple(range(90011, 90018)) + {1: (90021, 90022), 2: (90018, 90020), 3: (90019, 90023)}[obj.config["tower"]]
        obj.shelf = tuple(VisibleNoteOffer(row["item_id"], row["price"], tuple(map(tuple, row["contents"])), row["stock"]) for row in snapshot["shelf"])
        for key in ("crossed", "generation", "core", "swaps_left", "drop_per_battle", "note_events",
                    "note_spent", "note_purchases", "swaps_used", "steps", "finished"):
            setattr(obj, key, snapshot[key])
        obj.note_rng = Random(canonical_seed("drops_branch", seed))
        obj.shelf_rng = Random(canonical_seed("shelf_branch", seed))
        obj.event_rng = Random(canonical_seed("note_events_branch", seed))
        return obj


def note_progress(notes, skills):
    counts = dict(zip(NOTE_IDS, notes))
    value = 0.
    for skill in skills:
        level = skill.level(counts)
        value += level
        if level < len(skill.requirements):
            need = skill.requirements[level]
            value += .2 * sum(min(1., counts[n] / q) for n, q in need) / len(need)
    return value
