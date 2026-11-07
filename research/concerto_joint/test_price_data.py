import unittest

from price_data import normalize_note_price, repair_note_goods, validate_note_goods, validate_note_price
from infer import decide


class PriceDataTests(unittest.TestCase):
    def test_confirmed_ocr_pairs_and_legal_prices(self):
        self.assertEqual(normalize_note_price(5, 9045), 45)
        self.assertEqual(normalize_note_price(5, 9072), 72)
        for quantity, prices in ((5, (45, 72, 90)), (15, (200, 320, 400))):
            for price in prices:
                self.assertEqual(normalize_note_price(quantity, price), price)

    def test_do_not_guess_unknown_values_or_wrong_package(self):
        for quantity, price in ((5, 9090), (5, 999045), (15, 9045), (15, 90),
                                (5, 400), (5, -45), (5, 0), (True, 45), (5, "09045")):
            with self.subTest(quantity=quantity, price=price), self.assertRaises(ValueError):
                normalize_note_price(quantity, price)

    def test_runtime_rejects_unrepaired_prices(self):
        for price in (9045, 9072):
            with self.assertRaises(ValueError):
                validate_note_price(5, price)
            with self.assertRaises(ValueError):
                validate_note_goods([{"note": 90011, "quantity": 5, "price": price, "observations": 1}])

    def test_public_inference_rejects_bad_price_before_model_load(self):
        for price in (9045, 9072):
            snapshot = {"ordinary_state": {"phase": "SHOP"},
                        "shelf": [{"price": price, "contents": [[90011, 5]]}]}
            with self.assertRaisesRegex(ValueError, "quantity/price"):
                decide(snapshot, None)

    def test_mass_is_preserved_and_repair_is_idempotent(self):
        rows = [{"note": 90011, "quantity": 5, "price": price, "observations": count}
                for price, count in ((45, 535), (9045, 276), (72, 1007), (9072, 17))]
        result, audit = repair_note_goods(rows)
        self.assertEqual({row["price"]: row["observations"] for row in result}, {45: 811, 72: 1024})
        self.assertEqual(audit["corrected_observations"], 293)
        self.assertEqual(sum(row["observations"] for row in result), sum(row["observations"] for row in rows))
        again, second = repair_note_goods(result)
        self.assertEqual(result, again)
        self.assertEqual(second["corrected_observations"], 0)
