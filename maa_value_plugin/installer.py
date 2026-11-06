"""Attach the optional strategy to this official Maa installation."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import os
import shutil
import re
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent
BEGIN = "# >>> StellaSora external value model >>>"
END = "# <<< StellaSora external value model <<<"
MODE = "新版爬塔_潜能选择模式"
STRATEGY = "新版爬塔_策略"
CHOICE = "神经网络"
EXTRA_OPTIONS = (
    "新版爬塔_开始位置", "新版爬塔_潜能最大刷新次数", "新版爬塔_商店_自定义选项开关",
    "新版爬塔_商店_强化_自定义选项开关", "新版爬塔_事件选项策略文件", "新版爬塔_设置爬塔目标",
)


def read_jsonc(path):
    raw = Path(path).read_text(encoding="utf-8-sig")
    raw = re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/',
                 lambda m: m[0] if m[0].startswith('"') else "", raw, flags=re.S)
    raw = re.sub(r'"(?:\\.|[^"\\])*"|,\s*(?=[}\]])',
                 lambda m: m[0] if m[0].startswith('"') else "", raw)
    return json.loads(raw)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8-sig") == raw:
        return
    temporary = path.with_name(path.name + ".value-model.tmp")
    temporary.write_text(raw, encoding="utf-8")
    temporary.replace(path)


def remove_loader(source):
    return re.sub(r"(?m)^" + re.escape(BEGIN) + r"\r?\n.*?^" + re.escape(END) + r"\r?\n?",
                  "", source, flags=re.S)


def _is_model_case(case):
    handler = case.get("pipeline_override", {}).get(
        "星塔_节点_选择潜能_agent", {}).get("attach", {}).get("handler")
    return case.get("name") in (CHOICE, "value_model") or handler == "value_model"


def callback_overrides():
    overrides = {}
    bindings = read_jsonc(ROOT / "agent_bindings.json")
    for kind, field, custom in (("actions", "action", "custom_action"),
                                ("recognitions", "recognition", "custom_recognition")):
        for node, binding in bindings[kind].items():
            overrides[node] = {field: {"type": "Custom", "param": {custom: binding["name"]}}}
    return overrides


def patch_interface(ui, defaults, model_id):
    ui = deepcopy(ui)
    options = ui["option"]
    mode = options[MODE]
    original = [c for c in mode["cases"] if not _is_model_case(c)]
    if not original:
        original = deepcopy(defaults["option"][MODE]["cases"])
    strategy = options.get(STRATEGY, {})
    if not any(not _is_model_case(c) for c in strategy.get("cases", [])):
        options[STRATEGY] = deepcopy(defaults["option"][STRATEGY])
    # Original controls belong to the original modes; the model ignores them.
    referenced = {name for task in defaults["task"] for name in task.get("option", [])}
    extras = [name for name in EXTRA_OPTIONS
              if name in options or name in defaults["option"] or name in referenced]
    for name in extras:
        if name in defaults["option"]:
            options.setdefault(name, deepcopy(defaults["option"][name]))
    for case in original:
        children = [name for name in case.get("option", []) if name not in extras]
        case["option"] = children + extras
    model = {
        "name": CHOICE, "label": f"神经网络{model_id}",
        "description": "每次启动任务，首次扫荡后读取游戏当前使用的完整预设，连续爬塔复用。潜能、刷新、商店和已建模事件由神经网络决策。",
        "pipeline_override": {
            "星塔_节点_选择潜能_agent": {"attach": {"handler": "value_model", "environment": "tower_8"}},
            "星塔_入口_agent": {"attach": {"reset_state": True}},
            "星塔_跳过进入星塔流程_agent": {"enabled": False},
        },
    }
    # MFAAvalonia reads task option names with Index = 0, so a newly
    # created task can skip default_case and select the first case.
    mode["cases"] = [model, *original]
    mode["default_case"] = CHOICE
    for task in ui["task"]:
        if task["name"] != "新版爬塔":
            continue
        task["description"] = "国服和日服神经网络可自动读取当前游戏预设；也可选择官方原策略。"
        task["option"] = [name for name in task["option"] if name not in extras]
        # Existing nodes are overridden at task execution, never redefined in
        # another file within the same resource bundle (Maa rejects duplicates).
        overrides = task.setdefault("pipeline_override", {})
        for node, binding in callback_overrides().items():
            node_data = overrides.setdefault(node, {})
            for field, value in binding.items():
                custom = node_data.setdefault(field, {})
                custom["type"] = value["type"]
                custom.setdefault("param", {}).update(value["param"])
    return ui


def saved_model_selections(app, default_index, index_map):
    for path in (app / "config/instances").glob("*.json"):
        instance = read_jsonc(path)
        changed = False
        for task in instance.get("TaskItems", []):
            if task.get("entry") != "星塔_入口_agent":
                continue
            options = task.get("option") or []
            mode = next((option for option in options if option.get("name") == MODE), None)
            index = default_index
            if index is None and mode is not None:
                index = index_map.get(mode.get("index"))
            if index is None:
                continue
            if mode is None:
                options.append({"name": MODE, "index": index})
                task["option"] = options
            elif mode.get("index") != index:
                mode["index"] = index
            else:
                continue
            changed = True
        if changed:
            yield path, instance


def install(app, *, select_default=True, staged=False):
    app = Path(app).resolve()
    destination = app / "maa_value_plugin" if staged else ROOT
    main = app / "agent/main.py"
    source = main.read_text(encoding="utf-8-sig")
    clean = remove_loader(source)
    anchor = re.search(r"(?m)^import custom[^\n]*\n", clean)
    if anchor is None:
        raise RuntimeError("官方 agent/main.py 结构已变化，未找到 import custom 加载点。")
    ui_path = app / "interface/tasks/climb_tower.json"
    ui = read_jsonc(ui_path)
    defaults = read_jsonc(ROOT / "official_climb_tower.json")
    model_id = read_jsonc(ROOT / "model.json")["model_id"]
    patched = patch_interface(ui, defaults, model_id)
    indices = {case["name"]: i for i, case in enumerate(patched["option"][MODE]["cases"])}
    index_map = {}
    for old_index, case in enumerate(ui["option"][MODE]["cases"]):
        name = CHOICE if _is_model_case(case) else case["name"]
        if name in indices:
            index_map[old_index] = indices[name]
    selections = list(saved_model_selections(app, indices[CHOICE] if select_default else None, index_map))
    try:
        relative = os.path.relpath(destination, app)
    except ValueError:
        relative = str(destination)
    loader = (f"{BEGIN}\n"
              "def _load_stella_value_plugin():\n"
              "    import json\n"
              "    import runpy\n"
              "    from pathlib import Path\n"
              "    app_dir = Path(__file__).resolve().parents[1]\n"
              "    link = json.loads((app_dir / 'agent/stella_value_plugin.json').read_text(encoding='utf-8-sig'))\n"
              "    bootstrap = (app_dir / link['plugin_dir'] / 'bootstrap.py').resolve()\n"
              "    runpy.run_path(str(bootstrap))['install'](app_dir)\n"
              "_load_stella_value_plugin()\n"
              "del _load_stella_value_plugin\n"
              f"{END}\n")
    back = read_jsonc(app / "resource/base/pipeline/base.json")["通用_返回按钮"]
    swipe = read_jsonc(app / "resource/base/pipeline/climb_tower/climb_tower_agent_sub_bag.json")["星塔_背包_向下滑动_agent"]
    # These are direct-coordinate actions, so they have no recognition box
    # for wait_freezes' default target=True. Keep the official wait timings.
    click_wait = dict(back["post_wait_freezes"], target=[0, 0, 1280, 720])
    swipe_wait = swipe["post_wait_freezes"]
    swipe["post_wait_freezes"] = dict(
        swipe_wait if isinstance(swipe_wait, dict) else {"time": swipe_wait},
        target=[350, 95, 593, 591])
    pipeline = {
        "星塔_神经网络_预设OCR": {"recognition": {
            "type": "OCR", "param": {"expected": [".+"], "threshold": .3}}},
        "星塔_神经网络_预设卡片边界": {"recognition": {
            "type": "ColorMatch", "param": {"lower": [[255, 255, 255]],
                "upper": [[255, 255, 255]], "roi": [350, 95, 593, 591],
                "connected": True, "count": 100}}},
        "星塔_神经网络_预设点击": {
            "action": {"type": "Click"}, "post_wait_freezes": click_wait, "timeout": 60000},
        "星塔_神经网络_预设列表": {"recognition": {
            "type": "OCR", "param": {"expected": ["プリセット", "预设"], "roi": [120, 0, 260, 80], "threshold": .3}}},
        "星塔_神经网络_预设详情": {"recognition": {
            "type": "OCR", "param": {"expected": ["素質プリセット", "(?:潜能|素质)预设"], "roi": [120, 0, 260, 80], "threshold": .3}},
            "post_wait_freezes": back["post_wait_freezes"]},
        "星塔_神经网络_预设滑动": swipe,
        "星塔_神经网络_返回编队": {
            "timeout": 60000,
            "next": ["星塔_神经网络_编队页", "[JumpBack]通用_返回按钮"]},
        "星塔_神经网络_编队页": {"recognition": {
            "type": "And", "param": {"all_of": ["星塔_编队角色_agent"]}}},
    }
    # Complete resource reads and planning before modifying the live app.
    if staged:
        retained = {"config.json", "upload_settings.json", "preset.json", "preset_targets.json"}
        for source_file in ROOT.rglob("*"):
            if not source_file.is_file():
                continue
            relative_file = source_file.relative_to(ROOT)
            target = destination / relative_file
            if relative_file.as_posix() in retained and target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
    write_json(app / "agent/stella_value_plugin.json", {"plugin_dir": relative})
    updated = clean[:anchor.end()] + loader + clean[anchor.end():]
    if updated != source:
        main.write_text(updated, encoding="utf-8", newline="\n")
    write_json(ui_path, patched)
    write_json(app / "resource/base/pipeline/stella_value_model.json", pipeline)
    for path, instance in selections:
        write_json(path, instance)
    sampling = app / "agent/agent_config/tower_sampling.json"
    if not sampling.exists():
        write_json(sampling, {"enabled": True, "capture_screenshots": False})
    try:
        app_relative = os.path.relpath(app, destination)
    except ValueError:
        app_relative = str(app)
    write_json(destination / "installation.json", {"app_dir": app_relative, "module_dir": ".",
               "loader_file": "agent/main.py", "entry_point": "after import custom", "installer_version": 26,
               "python_platform": sysconfig.get_platform(), "controllers": ["Win32", "Adb"]})
    return model_id


def main():
    app = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent
    model = install(app, staged="--staged" in sys.argv[2:])
    print(f"神经网络{model} 安装完成。重启 MFAAvalonia 后，「新版爬塔」默认选中神经网络（国服/日服）。")
    print("扫荡后的编队页会自动读取当前使用的预设。OCR与模型日志已启用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
