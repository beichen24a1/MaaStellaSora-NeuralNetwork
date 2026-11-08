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
from rule_alignment import (SANDBOX_PROFILE, MissingEvidence, auxiliary_supply,
                            phase_catalog, require_sandbox, sample_visible_menu)
from maa_value_plugin.critic.environment import Episode, sim, legal_actions
from maa_value_plugin.critic.presets import Preset, canonical_seed, sample_preset

RULES = {
    "version": "joint_rules_alignment_v5_no_trained_checkpoint",
    "note_prices": "Quantity-specific legal prices; confirmed 9045->45 and 9072->72 OCR concatenations repaired before aggregation; unknown values fail validation",
    "ordinary_kernel": sim.MODEL_VERSION,
    "ordinary_route": "unchanged frozen 45-node route",
    "shelves": "Four discount slots, at most two half-price and one large bundle; sandbox large-bundle shelf rate 1451/1809, not an exact server probability; remaining conditional correlations uncalibrated",
    "note_product_identity": "Uniform among tower drop-group notes, independent of selected skills",
    "battle_notes": "Eight individual uniform draws per battle by default; uncalibrated, evaluate sensitivity",
    "initial_supply": "Auxiliary disc ID and explicit phase: nearest existing row <= phase; observed complete entry inventory replaces calculated supply",
    "note_events": "114 wrong answer retains event and disables option without notes; 12002 branches 50% heal/50% five notes; random-note allocation and costs require explicit uncalibrated sandbox profile",
    "visibility": "Observed snapshots retain exact menus; sandbox 107/108 generate two distinct specific options plus random (and 30 coins for 108), uniform pair law unvalidated",
    "event_legality": "All visible table options, without ordinary-objective dominance pruning; coin costs, every positive-mass wallet outcome and note costs checked before selection",
    "adjustments": "At terminal, at most 15 one-for-one transfers between any owned donor and a required target; legality/timing need game evidence",
    "objective": "Separate ordinary/concerto returns; lambda is an experimental preference, not measured damage",
    "scope": "Offline research only, no Maa or live-game integration",
    "joint_training_ready": False,
    "sandbox": "Explicit opt-in profile required. Drop, cost, menu-pair and shelf joint kernels remain uncalibrated; healing is counted, not a combat HP simulation",
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


def scenario(main, sub, tower, preset, name="custom", *, auxiliary_phases=None):
    resolved = inputs()["sigils"]
    groups = []
    for name_ in main:
        groups.extend(resolved[name_]["groups"])
    if auxiliary_phases is None or len(auxiliary_phases) != len(sub):
        raise ValueError('Explicit phases are required for every auxiliary disc')
    ids = phase_catalog()['legacy_name_to_disc_id']
    auxiliary = [{'disc_id': ids[name_], 'phase': phase} for name_, phase in zip(sub, auxiliary_phases)]
    supply = auxiliary_supply(auxiliary)
    value = {"name": name, "main_sigils": list(main), "sub_sigils": list(sub),
             "tower": tower, "preset": preset.to_dict(), "groups": groups,
             "auxiliary_discs": auxiliary, "supply_basis": "phase_catalog",
             "supply": list(supply)}
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
        result.append(scenario(main, sub, tower, preset, f"sandbox_phase0_{seed}_{i}", auxiliary_phases=[0]*3))
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
    def __init__(self, config, seed, drop_per_battle=8, swaps=15, note_events=True, *, assumptions=None):
        require_sandbox(assumptions)
        self.assumptions = assumptions
        self.config = config
        self.preset = Preset.from_dict(config["preset"])
        self.ep = JointKernelEpisode(self.preset, canonical_seed("ordinary", seed))
        if config.get('supply_basis') == 'phase_catalog':
            self.notes = auxiliary_supply(config['auxiliary_discs'])
            if tuple(config['supply']) != self.notes:
                raise ValueError('Cached supply disagrees with auxiliary disc phases')
        elif config.get('supply_basis') == 'observed_inventory':
            self.notes = auxiliary_supply([], observed_inventory=config['supply'])
        else:
            raise MissingEvidence('Supply requires explicit disc phases or complete observed entry inventory')
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
        self.healing_events, self.failed_answers = 0, 0
        self.steps = 0
        self._sync(new_shelf=self.ep.state.phase == "SHOP", new_event=self.ep.state.phase == "EVENT")

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
        tiers = remaining_discount_tiers(self.ep.state.stock, self.shelf_rng,
                    middle_initial=initial and self.ep.state.shopOrdinal in (2, 3))
        large_slot = (self.shelf_rng.randrange(len(tiers))
                      if tiers and self.shelf_rng.random() < 1451/1809 else None)
        self.shelf = tuple(VisibleNoteOffer(f"{self.ep.state.shopOrdinal}-{self.generation}-{slot}",
                            {5: (45, 72, 90), 15: (200, 320, 400)}[size][tier],
                            ((self.shelf_rng.choice(self.allowed), size),))
                           for slot, tier in enumerate(tiers)
                           for size in [15 if slot == large_slot else 5])

    def _sync(self, new_shelf=False, *, initial=True, new_event=False):
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
        if self.note_events and new_event and self.ep.state.phase == "EVENT":
            eid = self.ep.state.eventId
            if eid in (107, 108):
                self.ep.state = replace(self.ep.state, eventOptions=sample_visible_menu(eid, self.event_rng))

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
        # Explicit sandbox hypothesis: sample individual owned notes without
        # replacement, weighted by counts. Actual server deduction law unknown.
        for _ in range(quantity):
            note = self.event_rng.choices(NOTE_IDS, weights=self.notes)[0]
            self._give(note, -1)

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
            if oid in (11401, 11402, 11404, 11405):
                self.failed_answers += 1
                self.ep.state = replace(before, eventOptions=tuple(x for x in before.eventOptions if x != oid))
                return  # Failed answer: no reward, no advance, this option disabled.
            if oid == 10301:
                self._consume(5)
            elif oid == 10401:
                self._consume(10)
            elif 10701 <= oid <= 10707:
                self._give(90010 + oid % 100, 5)
            elif oid == 10708 or oid == 12702 or oid == 12001:
                self._random_notes(5, self.event_rng)
            elif oid == 12002:
                if self.event_rng.random() < .5:
                    self.healing_events += 1  # 30% HP branch; combat HP remains outside sandbox scope.
                else:
                    self._random_notes(5, self.event_rng)
            elif 10801 <= oid <= 10807:
                self._give(90010 + oid % 100, 10)
            elif oid == 10808 or oid == 11403:
                self._random_notes(10, self.event_rng)
        self.ep.step(base)
        new = self.ep.state
        new_shelf = new.phase == "SHOP" and (new.shopOrdinal != before.shopOrdinal or base.kind == "RESTOCK")
        self._sync(new_shelf, initial=base.kind != "RESTOCK",
                   new_event=new.phase == "EVENT" and (before.phase != "EVENT" or new.eventId != before.eventId))

    def outcome(self):
        return {"ordinary": sim.score(self.ep.state), "core": self.core,
                "potential_total": sim.score(self.ep.state) + self.core,
                "concerto": sum(self.levels()), "concerto_levels": list(self.levels()),
                "coins": self.ep.state.coins, "notes": list(self.notes),
                "note_spent": self.note_spent, "note_purchases": self.note_purchases,
                "swaps_used": self.swaps_used, "steps": self.steps,
                "healing_events": self.healing_events, "failed_answers": self.failed_answers,
                "assumptions": self.assumptions, "training_ready": False}

    def snapshot(self):
        return {"rules_version": RULES["version"], "config": self.config, "assumptions": self.assumptions,
                "healing_events": self.healing_events, "failed_answers": self.failed_answers,
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
        require_sandbox(snapshot.get('assumptions'))
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
                    "note_spent", "note_purchases", "swaps_used", "steps", "finished",
                    "assumptions", "healing_events", "failed_answers"):
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
