"""Evidence-backed rules and explicit gates for uncalibrated transitions."""
from functools import lru_cache
import json

from bootstrap import HERE

NOTE_IDS = tuple(range(90011, 90024))
SANDBOX_PROFILE = 'uncalibrated_uniform_notes_count_weighted_costs_v1'


class MissingEvidence(RuntimeError):
    pass


@lru_cache(maxsize=1)
def phase_catalog():
    return json.loads((HERE / 'data/phase_supply.json').read_text(encoding='utf-8'))


def auxiliary_supply(discs, observed_inventory=None):
    """Nearest existing phase <= supplied phase; never add all prior rows.

    A complete observed entry inventory already includes supply, so it replaces
    calculation instead of being added to it.
    """
    if observed_inventory is not None:
        if len(observed_inventory) != 13 or any(type(n) is not int or n < 0 for n in observed_inventory):
            raise ValueError('Observed inventory must contain thirteen known nonnegative integers')
        return tuple(observed_inventory)
    catalog, counts = phase_catalog(), dict.fromkeys(NOTE_IDS, 0)
    for disc in discs:
        phase = disc.get('phase')
        if type(phase) is not int or not 0 <= phase <= 8:
            raise ValueError('Each auxiliary disc requires an explicit phase 0..8')
        definition = catalog['discs'][str(disc['disc_id'])]
        rows = catalog['supply_groups'][str(definition['supply_group'])]
        candidates = [int(p) for p in rows if int(p) <= phase]
        if not candidates:
            raise MissingEvidence('No supply row at or below this phase')
        for note, quantity in rows[str(max(candidates))].items():
            counts[int(note)] += quantity
    return tuple(counts.values())


def sample_visible_menu(event_id, rng):
    """Observed menu structure; uniform pair frequency is SANDBOX ONLY."""
    if event_id not in (107, 108):
        raise ValueError('Only 107/108 have the observed two-specific-item structure')
    pair = rng.sample(range(1, 8), 2)
    ids = [event_id*100+i for i in sorted(pair)] + [event_id*100+8]
    if event_id == 108:
        ids.append(10809)
    return tuple(ids) + (0,)  # 0 is the simulator's leave control, not a server option index.


def require_sandbox(profile):
    if profile != SANDBOX_PROFILE:
        raise MissingEvidence('Full note transitions are uncalibrated. Supply the explicit sandbox profile '
                              'only for rule tests; it does not authorize training or deployment.')


def require_training_ready():
    status = json.loads((HERE / 'data/TRAINING_READINESS.json').read_text(encoding='utf-8'))
    if not status['joint_training_ready']:
        raise MissingEvidence('Joint training paused: ' + '; '.join(status['blocking_requirements']))
