"""Agent runtime. Register callbacks without modifying official classes."""
from pathlib import Path
import json

from maa.agent.agent_server import AgentServer
from utils.logger import get_logger
from custom.reco.climb_tower_potential.state import State

from .log_filter import ModelLogFilter
from .observation import ObservationTracker
from .tower_sampling_collector import SamplingCollector
from .value_model import ValueModel


class ValueModelRuntime:
    def __init__(self, root, app_dir):
        self.root = Path(root)
        self.app_dir = Path(app_dir).resolve()
        self.settings = json.loads((self.root / "config.json").read_text(encoding="utf-8-sig"))
        self.logger = get_logger("custom.stella_value_model")
        self.ui_logs = ModelLogFilter()
        self.collector = SamplingCollector(
            self.app_dir, get_logger("custom.stella_value_model.sampling"), self.root,
            owned_provider=lambda: State.owned_potentials,
            counter_provider=lambda: dict(high_level_span_count=State.high_level_span_count,
                enhance_high_level_span_count=State.enhance_high_level_span_count),
        )
        self.model = None
        self.tracker = None

    def active(self, context):
        node = context.get_node_data("星塔_节点_选择潜能_agent") or {}
        return (self.settings.get("enabled", True)
                and node.get("attach", {}).get("handler") == "value_model")

    def begin_task(self, context):
        node = context.get_node_data("星塔_节点_选择潜能_agent") or {}
        self.collector.begin_task(self.language(context), node.get("attach", {}).get("handler"))
        if self.tracker is not None and not self.active(context):
            # Interrupted model runs must not seed an official strategy's inventory cache.
            State.reset()
        self.tracker = None

    @staticmethod
    def language(context):
        node = context.get_node_data("星塔_节点_进行对话选择_agent") or {}
        language = node.get("attach", {}).get("lang_type", "cn")
        return {"zh_cn": "cn", "ja_jp": "jp"}.get(language, language)

    def read_preset(self, context):
        from .jp_preset_reader import PresetReader

        language = self.language(context)
        preset = PresetReader(context, self.root / "potential_name_catalog.json", self.logger, language).read()
        if self.model is None:
            self.model = ValueModel(self.root)
        self.tracker = ObservationTracker(self.root, preset, self.logger, self.model, self.collector)
        self.collector.bind_preset(preset, self.tracker)
        self.logger.info("[神经网络] 已读取当前预设：%s；普通目标%s",
                         " / ".join(member["name_zh_cn" if language == "cn" else "name_ja_jp"]
                                    for member in preset["party"]),
                         "/".join(map(str, preset["role_ordinary_totals"])))

    def require_tracker(self):
        if self.tracker is None:
            raise ValueError("本局尚未读取当前预设，请从扫荡前开始运行神经网络策略。")
        return self.tracker

    def reset_run(self):
        if self.tracker is not None:
            self.tracker.reset()

    def register(self):
        from . import agent_callbacks
        from .sampling_events import TowerContextSink, TowerTaskSink

        self.ui_logs.attach(get_logger("custom"))
        bindings = json.loads((self.root / "agent_bindings.json").read_text(encoding="utf-8"))
        for kind, register in (
            ("actions", AgentServer.register_custom_action),
            ("recognitions", AgentServer.register_custom_recognition),
        ):
            for binding in bindings[kind].values():
                callback = getattr(agent_callbacks, binding["handler"])(self)
                if not register(binding["name"], callback):
                    raise RuntimeError(f"无法注册神经网络节点：{binding['name']}")
        self.collector.loaded()
        AgentServer.add_context_sink(TowerContextSink(self.collector))
        AgentServer.add_tasker_sink(TowerTaskSink(self.collector))
        self.logger.debug("[神经网络] 独立 Agent 节点已注册；每次任务读取一次游戏当前预设。")
