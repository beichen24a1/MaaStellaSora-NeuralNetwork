"""Value decisions using Maa's original potential recognition/click loop."""
import numpy as np
from custom.reco.climb_tower_potential.handler_preset import RecommendationHandler
from custom.reco.climb_tower_potential.state import State
from custom.reco.climb_tower_potential.context import Trekker

from .decision_log import log_forecast
from .value_model import Action, sim

class ModelPotentialHandler(RecommendationHandler):
    def __init__(self, interactor, data, runtime):
        super().__init__(interactor, data)
        self.runtime = runtime

    def read_potentials_info(self):
        tracker = self.runtime.require_tracker()
        # The explicit roster identifies role; no portrait-order inference.
        self._update_recommended_potentials()
        self._update_names()
        self._update_levels()
        portraits = self._get_adjusted_rois(self.data.trekker_rois) if tracker.portrait_roles else ()
        unresolved = []
        for i, potential in enumerate(self.data.potentials):
            role = 1 if self.data.params.potential_source == "specified_drink" else None
            items = tracker.name_candidates(potential.name, role=role, rarity=potential.rarity)
            if len(items) != 1:
                unresolved.append((i, potential, items))
                continue
            self._set_trekker(tracker, potential, items[0]["role"], portraits, i)
        if not unresolved:
            return
        search_rois = self._expand_rois(portraits) if portraits else ()
        for i, potential, items in unresolved:
            roles = {item["role"] for item in items}
            matches = [role for role in sorted(roles)
                       if role in tracker.trekkers and tracker.trekkers[role].image.size
                       and self.screen.match_trekker(tracker.trekkers[role].image, search_rois[i])]
            role = matches[0] if len(matches) == 1 else None
            item = tracker.identify(potential.name, role=role, rarity=potential.rarity)
            self._set_trekker(tracker, potential, item["role"], portraits, i)

    def _set_trekker(self, tracker, potential, role, portraits, index):
        if role not in tracker.trekkers:
            image = (self.screen.crop_screenshot(portraits[index]).copy()
                     if role in tracker.portrait_roles else np.array([]))
            tracker.trekkers[role] = Trekker(index=role - 1, image=image, is_main=role == 1)
        potential.trekker = tracker.trekkers[role]

    def choose_potential(self):
        tracker = self.runtime.require_tracker()
        tracker.open_page(self.data, State.owned_potentials)
        if self.data.core_potential:
            chosen = max(self.data.potentials, key=lambda p: tracker.identify(
                p.name, role=p.trekker.index + 1, rarity=p.rarity)["target_level"] > 0)
            # Core levels are the fixed +6 in this ordinary-score model.
            log_forecast(self.runtime.logger, tracker.expected_total,
                         [{"label": f"核心{i} {p.name}", "change": 0., "selected": p is chosen}
                          for i, p in enumerate(self.data.potentials, 1)])
            return chosen
        actions = [a for a in sim.legal_actions(tracker.state) if a.kind == "SELECT"]
        # The model has only a 40-coin refresh. Check the OCR value before
        # refreshable, which divides by it when the button is visible.
        if self.data.refresh_cost == 40 and self.data.refreshable and self.data.current_coin >= 40:
            actions.append(Action("REROLL", price=self.data.refresh_cost))
        labels = {a: f"潜能{a.index} {self.data.potentials[a.index-1].name}"
                  for a in actions if a.kind == "SELECT"}
        action = tracker.choose(tuple(actions), labels)
        return None if action.kind == "REROLL" else self.data.potentials[action.index-1]

    def cache_potential_data(self, potential):
        tracker = self.runtime.require_tracker()
        super().cache_potential_data(potential)
        tracker.selected(potential)

