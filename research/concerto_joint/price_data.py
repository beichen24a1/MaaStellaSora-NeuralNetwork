"""Audited note-product price schema. Unknown prices fail instead of guessing."""
from collections import Counter

PRICE_SCHEMA = "note_product_prices_v2_clean"
LEGAL_PRICES = {5: frozenset((45, 72, 90)), 15: frozenset((200, 320, 400))}
KNOWN_OCR_CONCATENATIONS = {(5, 9045): 45, (5, 9072): 72}


def validate_note_price(quantity, price):
    if type(quantity) is not int or type(price) is not int:
        raise ValueError("Note quantity and price must be integers, excluding booleans")
    if quantity not in LEGAL_PRICES or price not in LEGAL_PRICES[quantity]:
        raise ValueError(f"Unverified note product quantity/price: {quantity}/{price}")
    return price


def normalize_note_price(quantity, price):
    """Only repair the two collaborator-confirmed, quantity-specific OCR errors."""
    if type(quantity) is not int or type(price) is not int:
        raise ValueError("Note quantity and price must be integers, excluding booleans")
    corrected = KNOWN_OCR_CONCATENATIONS.get((quantity, price), price)
    return validate_note_price(quantity, corrected)


def validate_note_goods(rows):
    seen = set()
    if not rows:
        raise ValueError("Empty note product catalog")
    for row in rows:
        note, quantity, price, count = (row.get(key) for key in ("note", "quantity", "price", "observations"))
        if type(note) is not int or not 90011 <= note <= 90023:
            raise ValueError("Unknown note ID in product catalog")
        if type(count) is not int or count <= 0:
            raise ValueError("Product observation count must be a positive integer")
        validate_note_price(quantity, price)
        key = note, quantity, price
        if key in seen:
            raise ValueError("Duplicate product rows must be aggregated before training")
        seen.add(key)


def repair_note_goods(rows):
    clean, corrections, before = Counter(), Counter(), Counter()
    for row in rows:
        note, quantity, price, count = (row.get(key) for key in ("note", "quantity", "price", "observations"))
        if type(note) is not int or not 90011 <= note <= 90023 or type(count) is not int or count <= 0:
            raise ValueError("Malformed note-product observation")
        corrected = normalize_note_price(quantity, price)
        clean[note, quantity, corrected] += count
        before[quantity, price] += count
        if price != corrected:
            corrections[quantity, price, corrected] += count
    result = [{"note": note, "quantity": quantity, "price": price, "observations": count}
              for (note, quantity, price), count in sorted(clean.items())]
    validate_note_goods(result)
    after = Counter()
    for row in result:
        after[row["quantity"], row["price"]] += row["observations"]
    total = sum(before.values())
    if total != sum(after.values()):
        raise RuntimeError("Price normalization changed observation mass")
    def histogram(counter):
        return [{"quantity": q, "price": p, "observations": n} for (q, p), n in sorted(counter.items())]
    audit = {"schema": PRICE_SCHEMA, "total_observations": total,
             "corrected_observations": sum(corrections.values()),
             "corrections": [{"quantity": q, "original_price": p, "corrected_price": new, "observations": n}
                             for (q, p, new), n in sorted(corrections.items())],
             "before": histogram(before), "after": histogram(after),
             "unknown_prices": 0, "observation_mass_preserved": True}
    return result, audit
