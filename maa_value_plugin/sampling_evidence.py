"""Small recording-only types; no Maa calls, model state or export transforms."""
from dataclasses import dataclass, field
import uuid


@dataclass
class ReadEvidence:
    kind: str
    links: dict = field(default_factory=dict)
    read_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    recognitions: list = field(default_factory=list)

    def matching(self, node, roi=None):
        return [r for r in self.recognitions if r["node_name"] == node
                and (roi is None or r["roi"] == list(roi))]

    @property
    def record_ids(self):
        return [r["record_id"] for r in self.recognitions]


def recognition_roi(entry, args, kwargs):
    override = kwargs.get("pipeline_override", args[1] if len(args) > 1 else {})
    node = override.get(entry, {}) if isinstance(override, dict) else {}
    return node.get("recognition", {}).get("param", {}).get("roi")


def raw_ocr(records):
    return [item for r in records for item in (r["raw_result"] or {}).get("all_results", [])]


class ObservedInventory:
    """Inventory reconstructed only from a fresh start and confirmed take actions.

    Maa's eager cache remains separate. Completeness describes this observation
    history, never an independently read bag or a server response.
    """
    def __init__(self, full_start=False):
        self.items = {}
        self.revision = 0
        self.complete = full_start
        self.issues = [] if full_start else [dict(reason="run_start_not_observed")]

    def uncertain(self, reason, **details):
        self.complete = False
        issue = dict(reason=reason, **details)
        if issue not in self.issues:
            self.issues.append(issue)

    def audit(self, candidates, offer_id):
        conflicts = []
        for p in candidates:
            identity, before = p["potential_id"], p.get("level_before", -1)
            if identity is None or before is None or before < 0:
                continue
            known = self.items.get(identity)
            expected = known["level"] if known else 0 if self.complete else None
            if expected is not None and before != expected:
                conflict = dict(potential_id=identity, history_level=expected, card_level_before=before)
                conflicts.append(conflict)
                self.uncertain("card_level_disagrees_with_history", offer_id=offer_id, **conflict)
        return conflicts

    def confirm(self, selected, record_id):
        identity = selected["potential_id"]
        level = selected.get("level_after", selected.get("new_level", -1))
        if identity is None or level is None or level < 1:
            self.uncertain("selected_potential_unresolved", confirmation_record_id=record_id)
            return
        row = dict(selected, level=level, confirmation_record_id=record_id,
                   level_basis=selected.get("level_basis", "card_level_OCR"))
        self.items[identity] = row
        self.revision += 1

    def fields(self):
        owned = [dict(p) for _, p in sorted(self.items.items())]
        counts = []
        for role in (1, 2, 3):
            members = [p for p in owned if p["role"] == role]
            noncore = [p for p in members if not p["core"]]
            counts.append(dict(role=role, core_count=len(members)-len(noncore),
                               noncore_count=len(noncore),
                               rare_count=sum(p["canonical_rarity"] == "rare" for p in noncore),
                               common_count=sum(p["canonical_rarity"] == "common" for p in noncore),
                               below_max_count=sum(p["level"] < 6 for p in noncore),
                               noncore_level_sum=sum(p["level"] for p in noncore)))
        return dict(owned=owned, owned_complete=self.complete,
                    owned_basis="fresh_start_and_UI_confirmed_selection_history",
                    owned_source="observed_selection_history", owned_revision=self.revision,
                    owned_issues=list(self.issues), owned_counts=counts,
                    owned_independently_verified=False)
