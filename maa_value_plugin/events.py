"""Event choices; unresolved menus keep the official event rules."""
import re
import unicodedata
from maa.custom_recognition import CustomRecognition
from custom.reco.climb_tower_event.main import EventRecognition
from custom.reco.climb_tower_potential.state import State

from .event_mapping import match_event
from .value_model import sim, replace

def _read_event_coins(context, image):
    node = "星塔_通用_识别当前金币_agent"
    # The expanded note bar moves the event wallet to the second HUD row.
    for roi in ((1180, 20, 65, 25), (1180, 75, 70, 35)):
        result = context.run_recognition(node, image, pipeline_override={
            node: {"recognition": {"param": {"roi": list(roi)}}}
        })
        if result and result.filtered_results:
            text = unicodedata.normalize("NFKC", result.filtered_results[0].text).strip()
            if re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)", text):
                return int(text.replace(",", ""))
    return None


def analyze_model_event(recognizer, context, argv, runtime):
    tracker = runtime.require_tracker()
    node = context.get_node_data(argv.node_name) or {}
    lang = node.get("attach", {}).get("lang_type", "cn")
    question = recognizer._get_question_text(context, argv.image, lang)
    choices = recognizer._get_choice_texts(context, argv.image, lang)
    if not choices:
        return CustomRecognition.AnalyzeResult(box=None, detail={})
    coins = _read_event_coins(context, argv.image)
    if coins is None:
        runtime.logger.info("[神经网络] 本帧未读到事件金币，等待下一次识别")
        return CustomRecognition.AnalyzeResult(box=None, detail={})
    matched = match_event(question, [{"text": c.text, "consequence": c.consequence} for c in choices], lang)
    if matched is None:
        result = EventRecognition.analyze(recognizer, context, argv)
        if result.box is None:
            return result
        tracker.sync_owned(State.owned_potentials)
        tracker.observe_route("EVENT_DRAW", coins)
        tracker.coins_uncertain = True
        tracker.route_estimated = True
        tracker.state = replace(tracker.state, phase="ADVANCE")
        # Store an unknown-source marker so a subsequent reward page does
        # not consume a second normal route instruction.
        attach = (context.get_node_data(argv.node_name) or {}).get("attach", {})
        consequence = attach.get("last_consequence", "")
        runtime.logger.info("[神经网络] 事件使用原规则：%s → %s（%s）",
                            question, attach.get("last_choice", ""), consequence)
        if re.search(r"素質|潜能|潛能|potential", consequence, re.I):
            tracker.pending_source = sim.Source()
        return result
    tracker.sync_owned(State.owned_potentials)
    tracker.observe_route("EVENT_DRAW", coins)
    eid, option_indices = matched
    tracker.state = sim.visible_event(tracker.state, eid, option_indices.values())
    legal = []
    by_option = {oid: index for index, oid in option_indices.items()}
    for action in sim.legal_actions(tracker.state):
        oid = 0 if action.kind == "SKIP_EVENT" else action.index
        if oid not in by_option:
            continue
        outcomes = sim.EVENT_DATA[eid][2][oid][1]
        if all(not n or coins + delta >= 0 for n, _d, delta, _kind, _scope in outcomes):
            legal.append(action)
    if not legal:
        raise ValueError(f"事件 {eid} 未找到可执行选项")
    labels = {a: f"选项{by_option[0 if a.kind == 'SKIP_EVENT' else a.index]+1} "
              + choices[by_option[0 if a.kind == 'SKIP_EVENT' else a.index]].text for a in legal}
    action = tracker.choose(tuple(legal), labels)
    oid = 0 if action.kind == "SKIP_EVENT" else action.index
    choice = choices[by_option[oid]]
    context.override_pipeline({argv.node_name: {"attach": {
        "last_question": question, "last_choice": choice.text, "last_consequence": choice.consequence
    }}})
    tracker.event_selected(eid, oid)
    return CustomRecognition.AnalyzeResult(box=choice.box, detail={"value_model": tracker.model.model_id,
                                                                   "event": eid, "option": oid})

