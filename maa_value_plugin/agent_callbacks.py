"""Maa callbacks: reuse official flows and select the model only for its mode."""
from contextvars import ContextVar
from dataclasses import replace

from custom.action.climb_tower_preparation import AscensionPreparation, SelectParty as OfficialSelectParty
from custom.action.climb_tower_loop import AscensionLoop
from custom.action.climb_tower_record_scan import (
    LEVEL_NODE, POTENTIAL_LOCATION_NODE, POTENTIAL_RECOGNITION_NODE, RecordScanAction,
)
from custom.action.climb_tower_shop.main import ShopAction
from custom.action.climb_tower_shop.interactor import ShopInteractor
from custom.reco.climb_tower_potential.main import ChoosePotentialRecognition
from custom.reco.climb_tower_event.main import EventRecognition

from .events import analyze_model_event
from .potential import ModelPotentialHandler
from .shop import ShopNavigationFailed, run_model_shop
from .tower_sampling_collector import ObservedPotentialHandler, ObservedShopHandler


class RuntimeCallback:
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime


class Preparation(RuntimeCallback, AscensionPreparation):
    def run(self, context, argv):
        self.runtime.begin_task(context)
        context = self.runtime.collector.context(context)
        if self.runtime.active(context):
            if self.runtime.language(context) not in ("cn", "jp"):
                self.runtime.logger.error("[神经网络] 自动读取预设支持国服和日服，请选择对应的资源类型。")
                return False
        return super().run(context, argv)


class SelectParty(RuntimeCallback, OfficialSelectParty):
    def run(self, context, argv):
        context = self.runtime.collector.context(context)
        if not self.runtime.active(context):
            return super().run(context, argv)
        if self.runtime.tracker is not None:
            return True
        try:
            self.runtime.read_preset(context)
            return True
        except (ImportError, ValueError, RuntimeError) as error:
            self.runtime.logger.error("[神经网络] 预设读取失败：%s", error)
            context.tasker.post_stop()
            return False


class TowerLoop(RuntimeCallback, AscensionLoop):
    def run(self, context, argv):
        context = self.runtime.collector.context(context)
        collector = self.runtime.collector
        collector.finish_at_boundary("loop_without_settlement")
        result = super().run(context, argv)
        if self.runtime.active(context):
            # The original loop resets Maa's inventory; the preset stays cached.
            self.runtime.reset_run()
        return result


class Potential(RuntimeCallback, ChoosePotentialRecognition):
    def analyze(self, context, argv):
        context = self.runtime.collector.context(context)
        self.runtime.collector.ensure_run("potential_page_recognized")
        with self.runtime.ui_logs.scope(self.runtime.active(context)), \
                self.runtime.collector.capture("potential_callback"):
            return super().analyze(context, argv)

    def _load_handler(self, interactor, data):
        if data.params.handler == "value_model" and self.runtime.active(interactor.context):
            self.runtime.require_tracker()
            data.params = replace(data.params, environment="tower_8", max_refresh_count=100000, reserved_coin=0)
            handler = ModelPotentialHandler(interactor, data, self.runtime)
        else:
            handler = super()._load_handler(interactor, data)
        return ObservedPotentialHandler(handler, self.runtime.collector)


class Shop(RuntimeCallback, ShopAction):
    def run(self, context, argv):
        context = self.runtime.collector.context(context)
        self.runtime.collector.ensure_run("shop_page_recognized")
        self.runtime.collector.emit("lifecycle_observed", lifecycle_type="shop_entry",
                                   node_name=argv.node_name, owned=self.runtime.collector.owned())
        if self.runtime.active(context):
            with self.runtime.ui_logs.scope():
                try:
                    return run_model_shop(self, context, argv, self.runtime)
                except ShopNavigationFailed:
                    return False
        if not self.runtime.collector.enabled:
            return super().run(context, argv)
        # Same driver as ShopAction.run; only the handler records observations.
        data = self._init_data(context, argv.node_name)
        interactor = ShopInteractor(context)
        handler = ObservedShopHandler(interactor, data, self.runtime.collector)
        handler.read_shop_page_info()
        while not context.tasker.stopping:
            handler.read_goods_page_info()
            handler.execute_buy_plan()
            if not handler.can_afford_refresh():
                break
            handler.refresh()
        handler.enhance_all()
        interactor.back_to_main_page()
        return True


class Event(RuntimeCallback, EventRecognition):
    def analyze(self, context, argv):
        context = self.runtime.collector.context(context)
        if self.runtime.active(context):
            with self.runtime.ui_logs.scope():
                result = analyze_model_event(self, context, argv, self.runtime)
        else:
            result = super().analyze(context, argv)
        attach = (context.get_node_data(argv.node_name) or {}).get("attach", {})
        self.runtime.collector.event_choice(result, attach)
        return result

    def _get_question_text(self, context, image, lang):
        text = super()._get_question_text(context, image, lang)
        self.runtime.collector.event_question = text
        return text

    def _get_choice_texts(self, context, image, lang):
        choices = super()._get_choice_texts(context, image, lang)
        if choices and self.runtime.collector.enabled:
            self.runtime.collector.event_menu(choices)
        return choices


class RecordScan(RuntimeCallback, RecordScanAction):
    def __init__(self, runtime):
        super().__init__(runtime)
        self._observed = ContextVar("record_scan_observations", default=None)

    def run(self, context, argv):
        collector = self.runtime.collector
        if not collector.enabled:
            return super().run(context, argv)
        context = collector.context(context)
        observed = {}
        token = self._observed.set(observed)
        try:
            result = super().run(context, argv)
            if not observed:
                # Official RecordScan skips OCR when neither stopping target is set.
                try:
                    image = context.tasker.controller.cached_image
                    self._read_record_level(context, image, LEVEL_NODE)
                    self._read_potential_counts(context, image, POTENTIAL_LOCATION_NODE, POTENTIAL_RECOGNITION_NODE)
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    collector.logger.warning("[OCR采集] 附加结算识别失败：%s", error)
        finally:
            self._observed.reset(token)
        collector.settlement_result = observed
        collector.emit("settlement_observed", **observed, final_owned=collector.inventory.fields()["owned"],
                       **collector.owned_fields(), tracker_estimate=collector.inferred_state())
        return result

    def _read_record_level(self, context, image, node):
        level = super()._read_record_level(context, image, node)
        if (observed := self._observed.get()) is not None:
            observed["record_level"] = level
        return level

    def _read_potential_counts(self, context, image, location_node, recognition_node):
        parts = super()._read_potential_counts(context, image, location_node, recognition_node)
        if (observed := self._observed.get()) is not None:
            observed["potential_counts"] = parts
        return parts
