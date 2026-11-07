import unittest
from dataclasses import replace

from note_state import (NOTE_IDS, NoteState, VisibleNoteOffer, encode_context,
                        load_catalog, outcome_vector, purchase)


class NoteStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()

    def make_state(self, counts, skill):
        return NoteState.from_counts(counts, coins=90, floors_left=0,
                                     swaps_left=0, skills=(skill,))

    def test_all_extracted_skill_boundaries(self):
        for skill in self.catalog.values():
            for level, requirements in enumerate(skill.requirements, 1):
                counts = dict.fromkeys(NOTE_IDS, 0)
                counts.update(requirements)
                with self.subTest(group=skill.group_id, level=level):
                    self.assertEqual(skill.level(counts), level)
                    for note, quantity in requirements:
                        below = {**counts, note: quantity - 1}
                        self.assertLess(skill.level(below), level)

    def test_threshold_crossing_wallet_stock_and_no_mutation(self):
        skill = next(iter(self.catalog.values()))
        counts = dict.fromkeys(NOTE_IDS, 0)
        counts.update(skill.requirements[0])
        missing_note = skill.requirements[0][0][0]
        counts[missing_note] -= 5
        before = self.make_state(counts, skill)
        offer = VisibleNoteOffer("observed-slot-3", 90, ((missing_note, 5),))
        after, sold = purchase(before, offer)
        self.assertEqual((before.levels(), after.levels()), ((0,), (1,)))
        self.assertEqual((before.coins, after.coins, sold.stock), (90, 0, 0))
        self.assertEqual(sum(after.counts) - sum(before.counts), 5)
        with self.assertRaises(ValueError):
            purchase(after, sold)
        with self.assertRaises(ValueError):
            purchase(before, replace(offer, price=91))

    def test_missing_inventory_is_not_zero(self):
        skill = next(iter(self.catalog.values()))
        with self.assertRaises(ValueError):
            self.make_state({NOTE_IDS[0]: 0}, skill)

    def test_different_contexts_visible_and_outcomes_separate(self):
        skill = next(iter(self.catalog.values()))
        a = self.make_state(dict.fromkeys(NOTE_IDS, 0), skill)
        b = replace(a, counts=(1,) + a.counts[1:])
        self.assertNotEqual(encode_context(a), encode_context(b))
        self.assertEqual(len(encode_context(a)), 496)
        self.assertEqual(outcome_vector(70, a)["ordinary_potential"], 70)
        self.assertNotIn("damage", outcome_vector(70, a))


if __name__ == "__main__":
    unittest.main()
