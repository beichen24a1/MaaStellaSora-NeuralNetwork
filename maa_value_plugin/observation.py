"""Public observations and per-run state; independent of Maa class registration."""
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import json
import uuid

from .decision_log import forecast_rows, log_forecast
from .name_matching import clean_name as _clean, match_name
from .value_model import Action, sim, replace


class ObservationTracker:
    def __init__(self, root, preset_data, logger, model, collector=None):
        self.root, self.logger = Path(root), logger
        self.collector = collector
        self.model = model
        self.preset = self.model.preset(preset_data)
        self.preset_data = preset_data
        self.name_field = "name_zh_cn" if preset_data.get("language") == "cn" else "name_ja_jp"
        self.names, self.local_names, self.catalogue = defaultdict(list), defaultdict(list), {}
        for i, member in enumerate(preset_data["party"], 1):
            role = int(member.get("role", i))
            for item in member["targets"]:
                item = dict(item, role=role)
                item["quality"] = 1 if item["rarity_class"] == "rare" else 2
                identity = item.get("potential_id", (role, item.get("name_ja_jp")))
                self.catalogue[identity] = item
                item["identity"] = identity
                self._index_names(item)
        for item in preset_data.get("recognition_catalog", []):
            item = dict(item)
            item["quality"] = 1 if item["rarity_class"] == "rare" else 2
            item["identity"] = item.get("potential_id", (item["role"], item.get("name_ja_jp")))
            self._index_names(item)
        self.portrait_roles = {item["role"] for items in self.names.values() if len(items) > 1
                               for item in items}
        self.reset()

    def _index_names(self, item):
        for field in ("name_ja_jp", "name_zh_cn", "name_en_us", "name_zh_tw"):
            if item.get(field):
                key = _clean(item[field])
                if item not in self.names[key]:
                    self.names[key].append(item)
                if field == self.name_field:
                    self.local_names[key].append(item)

    def reset(self):
        self.run_id = uuid.uuid4().hex
        self.decision_index = 0
        self.state = self.model.initial(self.preset)
        self.acquired = {}
        self.trekkers = {}
        self.pending_event = None
        self.pending_source = None
        self.coins_uncertain = False
        self.route_estimated = False
        self.route_candidates = ((self.state.node, self.state.k, self.state.A,
                                  self.state.cBel, Fraction(1)),)
        self._current_cards = None
        initial_actions = self.model.actions(self.state)
        self.expected_total = 6 + sim.score(self.state) + float(max(
            self.model.values(self.state, initial_actions, self.preset)))
        self.logger.debug("[神经网络] 开始新一局，模型=%s，预设=%s，普通目标=%d",
                         self.model.model_id, self.preset.name, self.preset.total)

    def name_candidates(self, name, *, role=None, rarity=None):
        return list(match_name(name, self.names, role=role, rarity=rarity,
                               preferred=self.local_names).items)

    def identify(self, name, *, role=None, rarity=None):
        items = self.name_candidates(name, role=role, rarity=rarity)
        if items:
            # The handler tries cached portraits first. If none can distinguish
            # a shared name yet, keep the required deterministic forced match.
            return items[0]
        raise ValueError(f"神经网络预设中未找到潜能：{name!r}")

    def sync_owned(self, owned):
        # Also imports Maa's already observed inventory when attaching mid-run.
        pools = [list(row) for row in self.state.pools]
        acquired = self.acquired.copy()
        for potential in owned:
            if potential.core:
                continue
            item = self.identify(potential.name, role=potential.trekker.index + 1, rarity=potential.rarity)
            if item["identity"] not in acquired:
                pi = 2*(item["role"]-1)+item["quality"]-1
                index = pools[pi].index(item["target_level"])
                pools[pi][index] = pools[pi][-1]
                pools[pi].pop()
            acquired[item["identity"]] = (item, int(potential.level))
        inventory = Counter()
        for item, level in acquired.values():
            role, quality, target = item["role"], item["quality"], item["target_level"]
            inventory[(role, quality, target, level)] += 1
        self.state = replace(self.state, H=tuple((*key, n) for key, n in sorted(inventory.items())),
                             pools=tuple(tuple(row) for row in pools))
        self.acquired = acquired

    def _event_coin_deltas(self):
        if self.pending_event is None:
            return ((0, Fraction(1)),)
        eid, oid = self.pending_event
        rows = sim.EVENT_DATA[eid][2][oid][1]
        result = defaultdict(Fraction)
        for n, d, delta, _kind, _scope in rows:
            result[delta] += Fraction(n, d)
        return tuple(result.items())

    def observe_route(self, phase, coins):
        """Condition the frozen public route on the next observed screen/wallet."""
        current = self.state
        candidates = defaultdict(lambda: defaultdict(Fraction))
        for node, k, a, c_belief, route_probability in self.route_candidates:
            for g, f, first, prior in c_belief:
                for probability, desc, latent in sim._coin_paths(node, k, a, g, f, first):
                    if desc[4] != phase:
                        continue
                    for delta, event_probability in self._event_coin_deltas():
                        if self.coins_uncertain or current.coins + delta + desc[5] == coins:
                            candidates[desc][latent] += (route_probability * prior * probability
                                                         * event_probability)
        if not candidates:
            # A previously unmodeled dialogue may change the wallet. Keep the
            # route posterior, and explicitly record that unobserved coin effect.
            if self.coins_uncertain:
                raise ValueError(f"神经网络路线无法对应当前页面：{phase}，route node={current.node}")
            self.logger.info("[神经网络] 钱包观察同步：%s；先前事件金币变化未完整观测", coins)
            self.coins_uncertain = True
            self.route_estimated = True
            return self.observe_route(phase, coins)
        desc, masses = max(candidates.items(), key=lambda item: sum(item[1].values()))
        # An optional floor-1 event and the floor-2 event can have identical
        # wallets. Preserve both positions until later screens distinguish them.
        total_probability = sum(sum(row.values()) for row in candidates.values())
        self.route_candidates = tuple(
            (position[0], position[2], position[3],
             tuple((*latent, p/sum(rows.values())) for latent, p in sorted(rows.items())),
             sum(rows.values())/total_probability)
            for position, rows in sorted(candidates.items())
        )
        total = sum(masses.values())
        belief = tuple((*key, p/total) for key, p in sorted(masses.items()))
        node, floor, k, a, _phase, _delta, shop = desc
        kw = dict(node=node, floor=floor, k=k, A=a, phase=phase, coins=int(coins), cBel=belief)
        if phase == "SHOP_UNSEEN":
            kw.update(shopOrdinal=shop, firstInitial=shop == 1, paidIndex=0, restocks=2,
                      shopReturnPhase="END" if shop == 4 else "ADVANCE")
        self.state = replace(current, **kw)
        self.pending_event = None
        self.coins_uncertain = False

    def open_page(self, data, owned):
        self.sync_owned(owned)
        source_name = data.params.potential_source
        coins = int(data.current_coin)
        if coins < 0:
            raise ValueError("神经网络未读到当前金币")
        if data.core_potential:
            self.observe_route("CORE", coins)
            self.state = replace(self.state, phase="ADVANCE")
            return
        shop_page = source_name in ("drink", "specified_drink", "enhance")
        event_page = self.state.phase == "EVENT" or self.pending_source is not None
        if data.level_upped and not shop_page:
            event_page = False
            self.pending_source = None
        elif event_page and self.pending_event is not None:
            eid, oid = self.pending_event
            event_page = any(n and kind != "none" and self.state.coins + delta == coins
                             for n, _d, delta, kind, _scope in sim.EVENT_DATA[eid][2][oid][1])
            if not event_page:
                self.pending_source = None
        elif event_page and self.coins_uncertain and coins > self.state.coins:
            # Unknown events may award a card OR coins. A later battle reward
            # identifies the next route draw rather than an event card.
            event_page = False
            self.pending_source = None
        if data.refresh_count == 0 and not shop_page and not event_page:
            self.observe_route("DRAW", coins)
        else:
            self.state = replace(self.state, coins=coins)
        if source_name == "enhance":
            source = sim.Source("regular", "any", 0, "SHOP", False)
            phase = "ENHANCE_PAGE"
        elif shop_page:
            source = sim.Source("regular", "main" if source_name == "specified_drink" else "any", 0, "SHOP")
            phase = "PAGE"
        else:
            source = self.pending_source or self.state.source or sim.Source()
            phase = "PAGE"
        pools = [list(row) for row in self.state.pools]
        cards, indices, normalized = [], Counter(), []
        for potential in data.potentials:
            item = self.identify(potential.name, role=potential.trekker.index + 1, rarity=potential.rarity)
            role, quality, target = item["role"], item["quality"], item["target_level"]
            old, new = int(potential.old_level), int(potential.new_level)
            if not 0 <= old < new <= 6:
                raise ValueError(f"潜能等级识别有误：{potential.name} {old}→{new}")
            pool_index = 0
            if old == 0:
                pi = 2*(role-1)+quality-1
                offset = indices[pi]
                found = pools[pi].index(target, offset)
                pools[pi][offset], pools[pi][found] = pools[pi][found], pools[pi][offset]
                indices[pi] += 1
                pool_index = offset+1
            cards.append((int(old == 0), role, quality, target, old, new, pool_index))
            normalized.append(item)
        if source.roleMode == "bound" and cards:
            source = replace(source, boundRole=cards[0][1])
        p, q = self.state.P, self.state.Q
        for is_new, _r, _quality, _target, _old, level, _index in cards:
            if is_new:
                if level == 1:
                    p = True
                else:
                    p, q = False, level == 2 and self.state.h < 10
        if self.state.h >= 10:
            q = False
        belief = self.state.eBel
        if phase == "ENHANCE_PAGE":
            observable = [card for card in cards if card[4] < 5]
            bonus = observable[0][5] - observable[0][4] == 2 if observable else None
            belief = sim._enhance_posterior(belief, bonus)
        self.state = replace(self.state, phase=phase, source=source, offer=tuple(cards),
                             pools=tuple(tuple(row) for row in pools), P=p, Q=q, eBel=belief,
                             eventId=0, eventOptions=())
        self._current_cards = normalized
        self.pending_source = None
        self.pending_event = None

    def choose(self, actions, labels=None):
        # Snapshot exactly the public input used for this returned action.
        state = self.state
        actions = tuple(actions)
        decision = self.model.decide(state, actions, self.preset)
        action = Action(**decision["action_record"])
        self.logger.debug("[神经网络状态] phase=%s floor=%d coins=%d score=%d H=%s", state.phase, state.floor, state.coins, decision["ordinary_score"], state.H)
        for candidate in decision["values"]:
            self.logger.debug("[神经网络候选] %s | Q=%.6f", candidate["action"], candidate["expected_remaining_ordinary_return"])
        previous_total = self.expected_total
        rows = forecast_rows(decision, actions, decision["ordinary_score"], previous_total, labels)
        selected_total = 6 + decision["ordinary_score"] + decision["q"]
        log_forecast(self.logger, selected_total, rows)
        self.expected_total = selected_total
        folder = self.root / "logs"
        record = dict(decision, time=datetime.now().astimezone().isoformat(), preset=self.preset.name,
                      phase=self.state.phase, floor=self.state.floor, node=self.state.node,
                      coins=self.state.coins, wallet_route_sync_estimated=self.route_estimated)
        self.decision_index += 1
        record.update(
            log_schema_version="maa_public_decision_v3", run_id=self.run_id,
            previous_expected_total=previous_total, expected_total=selected_total,
            option_forecasts=rows, change_baseline="preceding_decision_forecast_or_initial_state",
            decision_index=self.decision_index, capture_timing="before_return_before_action_execution",
            public_state=sim.to_jsonable(state),
            original_preset=sim.to_jsonable(self.preset_data),
            original_target_pools=sim.to_jsonable(self.preset.pools),
            original_target_vector=list(self.preset.flat),
            preset_fingerprint=self.preset.fingerprint,
            legal_actions=sim.to_jsonable(actions),
            tracker_public_observation={
                "route_candidates": sim.to_jsonable(self.route_candidates),
                "coins_uncertain": self.coins_uncertain,
                "route_estimated": self.route_estimated,
                "pending_event": sim.to_jsonable(self.pending_event),
                "pending_source": sim.to_jsonable(self.pending_source),
            },
            versions=self.model.versions,
        )
        try:
            folder.mkdir(exist_ok=True)
            with (folder / f"decisions_{datetime.now():%Y%m%d}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as error:
            self.logger.warning("[神经网络] 无法写入决策日志：%s", error)
        if self.collector and self.collector.enabled:
            self.collector.model_decision(record)
        return action

    def selected(self, potential):
        item = self.identify(potential.name, role=potential.trekker.index + 1, rarity=potential.rarity)
        if item["rarity_class"] == "core":
            return
        index = next(i for i, row in enumerate(self._current_cards, 1) if row["identity"] == item["identity"])
        self.state = sim._select(self.state, index)
        self.acquired[item["identity"]] = (item, int(potential.new_level))

    def event_selected(self, eid, oid):
        self.pending_event = (eid, oid)
        outcomes = sim.EVENT_DATA[eid][2][oid][1]
        kinds = {(kind, scope) for n, _d, _delta, kind, scope in outcomes if n and kind != "none"}
        if len(kinds) == 1:
            kind, scope = next(iter(kinds))
            self.pending_source = sim.Source(kind, "main" if scope == "main" else "bound" if scope == "support_random" else "any",
                                             2 if scope == "support_random" else 0)
        self.state = replace(self.state, phase="ADVANCE", eventId=0, eventOptions=())

