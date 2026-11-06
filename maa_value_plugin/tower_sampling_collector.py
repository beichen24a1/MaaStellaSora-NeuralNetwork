"""Local adapters around official readers; all run state belongs to the collector."""
from copy import copy
from functools import partial
import uuid

from custom.action.climb_tower_shop.handler import ShopHandler
from .sampling_data import SamplingCollector, snapshot


class SamplingContext:
    """A view passed only to our callbacks; the actual Maa Context is untouched."""
    __slots__ = ("_context", "_collector", "_language")

    def __init__(self, context, collector):
        self._context = context
        self._collector = collector
        node = context.get_node_data("星塔_节点_商店_商店主流程_agent") or {}
        self._language = node.get("attach", {}).get("lang", "unknown")

    def __getattr__(self, name):
        return getattr(self._context, name)

    def run_recognition(self, entry, *args, **kwargs):
        result = self._context.run_recognition(entry, *args, **kwargs)
        if isinstance(entry, str) and "星塔" in entry:
            self._collector.record_recognition(entry, args, kwargs, result, self._language)
        return result

    def run_task(self, entry, *args, **kwargs):
        observed = isinstance(entry, str) and "星塔" in entry
        common = dict(node_name=entry, lang_type=self._language, method="run_task",
                      arguments=snapshot(args), keyword_arguments=snapshot(kwargs))
        read = self._collector.current_read
        if read:
            common.update(read_id=read.read_id, **read.links)
        if observed:
            self._collector.emit("context_run_task", phase="before", **common)
        result = self._context.run_task(entry, *args, **kwargs)
        if observed:
            self._collector.emit("context_run_task", phase="after", raw_result=snapshot(result), **common)
        return result


class ObservedPotentialHandler:
    """Delegate the official handler without adding attributes to its instance."""
    def __init__(self, handler, collector):
        self.handler = handler
        self.collector = collector
        self.offer_id = None
        self.selection_id = None
        self.page_read = None
        self.rarity_refresh_count = 0
        read = collector.current_read
        self.preload_records = list(read.recognitions) if read else []

    def __getattr__(self, name):
        return getattr(self.handler, name)

    def initialize_potentials(self):
        if not self.collector.enabled:
            self._update_refreshed_rarities()
            return self.handler.initialize_potentials()
        page = self.collector.prepare_offer(self.handler)
        self.page_read = self.collector.new_read("potential_page", offer_id=page["offer_id"],
            reward_chain_id=self.collector.reward["reward_chain_id"])
        self.page_read.recognitions.extend(self.preload_records)
        self.preload_records = []
        with self.collector.capture(read=self.page_read):
            self._update_refreshed_rarities()
            return self.handler.initialize_potentials()

    def _update_refreshed_rarities(self):
        data = self.handler.data
        if data.refresh_count != self.rarity_refresh_count:
            # Official preload runs once; every successfully refreshed page has
            # its own rarity layout, read with the same official recognizer.
            data.potential_rarities = self.handler.screen.get_potentials_rarity(data.core_potential)
            self.rarity_refresh_count = data.refresh_count

    def read_potentials_info(self):
        if not self.collector.enabled:
            return self.handler.read_potentials_info()
        with self.collector.capture(read=self.page_read):
            result = self.handler.read_potentials_info()
        self.offer_id = self.collector.observe_offer(self.handler, self.page_read)
        return result

    def choose_potential(self):
        return self._choose(self.handler.choose_potential, "policy_decision")

    def choose_fallback_potential(self):
        return self._choose(self.handler.choose_fallback_potential, "fallback_policy_decision")

    def _choose(self, function, event):
        potential = function()
        if self.collector.enabled:
            self.collector.emit(event, offer_id=self.offer_id,
                reward_chain_id=self.collector.reward["reward_chain_id"],
                selected=self.collector.potential(potential) if potential else None,
                refresh_requested=potential is None, operation_confirmed=False)
        return potential

    def pick(self, potential):
        self.selection_id = uuid.uuid4().hex
        return self.collector.operation(
            "selection_result", self.handler, self.handler.pick, potential,
            before=True, offer_id=self.offer_id, operation_id=self.selection_id,
        )

    def cache_potential_data(self, potential):
        result = self.handler.cache_potential_data(potential)
        if self.collector.enabled:
            self.collector.cache_selection(potential, self.offer_id, self.selection_id)
        return result

    def refresh(self):
        return self.collector.operation(
            "refresh_result", self.handler, self.handler.refresh, before=True, offer_id=self.offer_id
        )


class ObservedShopHandler(ShopHandler):
    def __init__(self, interactor, data, collector):
        super().__init__(interactor, data)
        self.collector = collector
        self.shop_visit_id = uuid.uuid4().hex
        self.stock_generations = {}
        self.stock_generation_id = None
        # BuyPipeline.get_plan writes target_item: keep that write local to this shop.
        self.strategies = [copy(strategy) for strategy in self.strategies]
        for strategy in self.strategies:
            strategy.buyer = partial(self._purchase, buyer=strategy.buyer)

    def _purchase(self, item, handler, *, buyer):
        if not self.collector.enabled:
            return buyer(item, handler)
        common = dict(lang_type=self.data.params.lang, item=snapshot(item),
                      operation_id=uuid.uuid4().hex, stock_scan_id=self.collector.stock_scan_id)
        self.collector.emit("purchase_attempt", phase="before", state=snapshot(self.data), **common)
        with self.collector.action_scope(**common), self.collector.capture("purchase", **common):
            result = buyer(item, handler)
        self.collector.emit(
            "purchase_result", phase="after", success=result, state=snapshot(self.data), **common
        )
        return result

    def read_goods_page_info(self):
        if not self.collector.enabled:
            return super().read_goods_page_info()
        self.collector.ensure_run("shop_page")
        previous = snapshot(self.data)
        self.collector.stock_scan_id = uuid.uuid4().hex
        with self.collector.capture("shop_stock", stock_scan_id=self.collector.stock_scan_id,
                                    shop_visit_id=self.shop_visit_id) as read:
            result = super().read_goods_page_info()
        self.collector.observe_shop(self, read, previous)
        return result

    def refresh(self):
        return self.collector.operation("shop_refresh_result", self, super().refresh, before=True)

    def _enhance(self):
        return self.collector.operation("enhancement_result", self, super()._enhance, before=True)
