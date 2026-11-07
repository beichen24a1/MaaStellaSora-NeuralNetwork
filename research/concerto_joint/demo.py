"""Constructed rule example, not an observed game or learned decision."""
import json

from note_state import (NOTE_IDS, NoteState, VisibleNoteOffer, encode_context,
                        load_catalog, outcome_vector, purchase)


def main():
    skill = next(iter(load_catalog().values()))
    counts = dict.fromkeys(NOTE_IDS, 0)
    counts.update(skill.requirements[0])
    missing = skill.requirements[0][0][0]
    counts[missing] -= 5
    state = NoteState.from_counts(counts, coins=90, floors_left=0,
                                 swaps_left=0, skills=(skill,))
    offer = VisibleNoteOffer("constructed-visible-slot", 90, ((missing, 5),))
    after, sold = purchase(state, offer)
    print(json.dumps({
        "scenario": "constructed threshold-crossing example, not a training benchmark",
        "skill_group_id": skill.group_id,
        "keep_money": outcome_vector(70, state),
        "buy_known_note": outcome_vector(70, after),
        "remaining_stock": sold.stock,
        "public_note_feature_dimension": len(encode_context(state)),
        "decision": "No learned recommendation: ordinary opportunity cost and future returns are not modeled yet.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
