"""Read the active CN/JP in-game preset using the same party-screen flow."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
import json
import re

import numpy as np
from maa.buffer import ImageBuffer

from .preset_tool import compile_preset
from .name_matching import clean_name as clean

OCR_NODE = "星塔_神经网络_预设OCR"
CLICK_NODE = "星塔_神经网络_预设点击"
SWIPE_NODE = "星塔_神经网络_预设滑动"
RETURN_NODE = "星塔_神经网络_返回编队"
LIST_NODE = "星塔_神经网络_预设列表"
DETAIL_NODE = "星塔_神经网络_预设详情"
CARD_NODE = "星塔_神经网络_预设卡片边界"

LOCALES = {
    "jp": {"name_field": "name_ja_jp", "button": "プリセット", "selected": r"現在設定中"},
    "cn": {"name_field": "name_zh_cn", "button": "预设",
           "selected": r"(?:当前|正在|现在)(?:设[置定]|使用|应用|生效)|(?:使用|设[置定])中"},
}


def resize(image, width, height):
    buffer = ImageBuffer()
    if not buffer.set(image) or not buffer.resize(width, height):
        raise ValueError("Maa 图像缩放失败")
    return buffer.get()


def grayscale(image):
    channels = image[:, :, :3].astype(np.uint16)
    return (29 * channels[:, :, 0] + 150 * channels[:, :, 1] + 77 * channels[:, :, 2]) >> 8


def same_view(previous, image):
    """Compare the scroll area itself; OCR boxes can jitter at the bottom."""
    return (previous is not None
            and np.mean(np.abs(previous.astype(np.int16) - image.astype(np.int16)) > 20) < .02)


@dataclass(frozen=True)
class Text:
    text: str
    box: tuple[int, int, int, int]
    score: float = 1.0

    @property
    def cx(self):
        return self.box[0] + self.box[2] / 2

    @property
    def cy(self):
        return self.box[1] + self.box[3] / 2


def as_text(rows):
    return [Text(str(r.text), tuple(map(int, r.box)), float(r.score)) for r in rows]


def joined_text(words):
    """Join OCR fragments left-to-right, preserving wrapped name lines."""
    lines = []
    for word in sorted(words, key=lambda row: row.cy):
        if not lines or word.cy - lines[-1][0].cy > 12:
            lines.append([word])
        else:
            lines[-1].append(word)
    return "".join("".join(row.text for row in sorted(line, key=lambda row: row.box[0]))
                   for line in lines)


class Catalog:
    def __init__(self, path, language="jp"):
        self.language = language
        self.name_field = LOCALES[language]["name_field"]
        self.data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        merged = {}
        for aliases in self.data["by_normalized_name"].values():
            for row in aliases:
                key = (row["potential_id"], tuple(row["roles"]))
                merged.setdefault(key, {}).update({k: v for k, v in row.items() if v is not None})
        self.rows = tuple(merged.values())
        self.match = lru_cache(maxsize=4096)(self._match)

    def _match(self, text, role=0, character=0, roster=(), force=False):
        key = clean(text)
        if len(key) < 2:
            return None
        candidates = []
        for row in self.rows:
            if character and row["character_id"] != character:
                continue
            if role and ("master" if role == 1 else "assist") not in row["roles"]:
                continue
            if roster and not any(row["character_id"] == member
                                  and ("master" if slot == 1 else "assist") in row["roles"]
                                  for slot, member in roster):
                continue
            name = clean(row.get(self.name_field, ""))
            if not name:
                continue
            ratio = 1.0 if key == name else SequenceMatcher(None, key, name).ratio()
            # Once the character is known, tolerate a single wrong glyph in a
            # short name such as 触む暗色 / 蝕む暗色. Never guess across the party.
            short_name = 4 <= min(len(key), len(name)) <= 5 and abs(len(key) - len(name)) <= 1
            threshold = .75 if character and short_name else (.82 if len(key) > 4 else .91)
            if force or ratio >= threshold:
                candidates.append((ratio, row))
        if not candidates:
            return None
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        if (not force and len(candidates) > 1 and candidates[0][1]["potential_id"] != candidates[1][1]["potential_id"]
                and candidates[0][0] - candidates[1][0] < (.1 if candidates[0][0] < .82 else .035)):
            return None
        return candidates[0][1]


class Accumulator:
    """Join overlapping scroll frames using names, IDs and displayed totals."""
    def __init__(self, catalog):
        self.catalog = catalog
        self.names = {}
        self.totals = {}
        self.total_sources = {}
        self.characters = {}
        self.targets = {1: {}, 2: {}, 3: {}}
        self.frames = 0
        self.forced_matches = []

    def read_sidebar(self, texts):
        for role, y in ((1, 190), (2, 299), (3, 407)):
            words = [r for r in texts if 110 < r.cx < 270 and abs(r.cy - y) < 30
                     and r.score >= .65 and not re.fullmatch(r"\d+", r.text)]
            name = joined_text(words)
            if len(clean(name)) > len(clean(self.names.get(role, ""))):
                self.names[role] = name
        for role, y in ((1, 230), (2, 336), (3, 444)):
            values = [r for r in texts if 155 < r.cx < 212 and abs(r.cy - y) < 22
                      and re.fullmatch(r"(?:总等级|等级|[Ll][Vv])?\d{1,3}", clean(r.text)) and r.score >= .7]
            if values:
                best = min(values, key=lambda r: abs(r.cy - y))
                self.totals[role] = int(re.search(r"\d+$", clean(best.text)).group())
                self.total_sources[role] = "sidebar"

    def _headers(self, texts):
        headers = []
        for row in texts:
            if not 295 < row.box[0] < 355 or not 95 <= row.cy < 686:
                continue
            # Names such as 尘沙（源初） may be split into adjacent OCR boxes.
            parts = [r for r in texts if 295 < r.box[0] and r.cx < 850
                     and abs(r.cy - row.cy) < 14 and r.score >= .5]
            name = joined_text(parts)
            key = clean(name)
            scores = [(SequenceMatcher(None, key, clean(name)).ratio(), role)
                      for role, name in self.names.items()]
            if scores:
                ratio, role = max(scores)
                sidebar = clean(self.names[role])
                if ratio >= .8 or (len(sidebar) >= 2 and sidebar in key):
                    if len(key) > len(sidebar):
                        self.names[role] = name
                    if any(abs(y - row.cy) < 14 for y, _ in headers):
                        continue
                    headers.append((row.cy, role))
                    if role not in self.totals:
                        values = [r for r in texts if 860 < r.cx < 945
                                  and abs(r.cy - row.cy) < 18 and r.score >= .5
                                  and re.fullmatch(r"\d{1,3}", clean(r.text))]
                        if values:
                            best = min(values, key=lambda r: abs(r.cy - row.cy))
                            self.totals[role] = int(clean(best.text))
                            self.total_sources[role] = "character_header"
        return sorted(headers)

    def feed(self, image, texts, read_level, rectangles):
        self.frames += 1
        if len(self.names) < 3 or len(self.totals) < 3:
            self.read_sidebar(texts)
        headers = self._headers(texts)
        # Other recognized card labels anchor their row, keeping section titles
        # out of forced matching when the OCR damages a potential name.
        card_rows = [row.cy for row in texts if 354 <= row.box[0] and row.cx < 942
                     and self.catalog.match(row.text)]
        columns = {i: [] for i in range(5)}
        for row in texts:
            if row.score < .55 or row.box[0] < 354 or row.cx >= 942 or not 125 < row.cy < 686:
                continue
            col = int((row.cx - 363) // 117.3)
            if col in columns and len(clean(row.text)) >= 2:
                columns[col].append(row)
        for col, words in columns.items():
            words.sort(key=lambda r: r.cy)
            for i, word in enumerate(words):
                previous_headers = [role for y, role in headers if y < word.cy]
                role = previous_headers[-1] if previous_headers else 0
                character = self.characters.get(role, 0)
                variants = [(word.text, word.box)]
                if i + 1 < len(words) and 0 < words[i + 1].cy - word.cy < 22:
                    other = words[i + 1]
                    variants.insert(0, (word.text + other.text,
                                       (word.box[0], word.box[1], word.box[2],
                                        other.box[1] + other.box[3] - word.box[1])))
                item = None
                roster = tuple(sorted(self.characters.items())) if not role and len(self.characters) == 3 else ()
                for text, box in variants:
                    item = self.catalog.match(text, role, character, roster)
                    if item:
                        break
                forced_text = None
                if not item and (character or roster) and any(abs(word.cy - y) < 22 for y in card_rows):
                    text, box = variants[0]
                    item = self.catalog.match(text, role, character, roster, force=True)
                    forced_text = text
                if not item:
                    continue
                if not role:
                    role = next((r for r, c in self.characters.items() if c == item["character_id"]), 0)
                if not role:
                    continue
                existing = self.characters.get(role)
                if existing is not None and existing != item["character_id"]:
                    continue
                self.characters[role] = item["character_id"]
                pid = item["potential_id"]
                if forced_text is not None:
                    match = {"role": role, "ocr_text": forced_text, "potential_id": pid}
                    if match not in self.forced_matches:
                        self.forced_matches.append(match)
                if item["item_stype"] == 42:
                    self.targets[role][pid] = (item, 1)
                    continue
                # Ignore a card clipped at the top. Its full level was read on
                # the preceding overlapping frame, or will be read after scroll.
                row_tops = [(abs((b[1] + b[3]) - (box[1] + box[3] + 8)), b[1])
                            for b in rectangles if abs((b[1] + b[3]) - (box[1] + box[3] + 8)) < 25]
                top = min(row_tops)[1] if row_tops else box[1] - 109
                if top < 100 or top > 653:
                    continue
                x = round(363 + 117.3 * col)
                grade_image = image[max(97, top - 1):top + 27, x + 3:x + 36]
                level = read_level(grade_image)
                if level is not None:
                    self.targets[role][pid] = (item, level)

    def observed_totals(self):
        return {role: sum(level for _, level in values.values()) for role, values in self.targets.items()}

    @property
    def complete(self):
        return (len(self.characters) == len(self.totals) == len(self.names) == 3
                and self.observed_totals() == self.totals)

    def compile(self, attribute_tower=""):
        party = []
        for role in (1, 2, 3):
            targets = self.targets[role]
            party.append({"character_id": self.characters[role], self.catalog.name_field: self.names[role],
                          "core": {str(pid): level for pid, (row, level) in targets.items() if row["item_stype"] == 42},
                          "ordinary": {str(pid): level for pid, (row, level) in targets.items() if row["item_stype"] == 41}})
        preset = compile_preset({"name": "游戏当前预设", "party": party,
                                 "attribute_tower": attribute_tower}, self.catalog.data)
        preset["language"] = self.catalog.language
        preset["source"] = {"target_authority": f"{self.catalog.language}_in_game_current_preset",
                            "method": "Maa OCR, card levels and scrolling",
                            "displayed_role_totals": self.totals, "frames_read": self.frames,
                            "displayed_role_total_sources": self.total_sources,
                            "observed_role_totals": self.observed_totals(), "totals_match": self.complete,
                            "forced_name_matches": self.forced_matches}
        return preset


class PresetReader:
    def __init__(self, context, catalog_path, logger, language="jp"):
        self.context, self.logger = context, logger
        self.controller = context.tasker.controller
        self.locale = LOCALES[language]
        self.catalog = Catalog(catalog_path, language)

    def screenshot(self):
        image = self.controller.post_screencap().wait().get()
        if image is None or not image.size:
            raise ValueError("未取得游戏截图")
        if image.shape[:2] != (720, 1280):
            image = resize(image, 1280, 720)
        return image

    def ocr(self, image, *, roi=(0, 0, 0, 0), digits=False):
        result = self.context.run_recognition(OCR_NODE, image, {OCR_NODE: {"recognition": {
            "type": "OCR", "param": {"roi": list(roi), "expected": [r"[1-6]" if digits else ".+"],
                                      "only_rec": digits, "threshold": .3,
                                      "replace": [], "order_by": "Vertical"}}}})
        return as_text(result.all_results) if result else []

    def level(self, crop):
        if not crop.size:
            return None
        image = resize(crop, crop.shape[1] * 3, crop.shape[0] * 3)
        rows = self.ocr(image, digits=True)
        valid = [(r.score, int(clean(r.text))) for r in rows if re.fullmatch(r"[1-6]", clean(r.text))]
        return max(valid)[1] if valid else None

    def card_rects(self, image):
        # Close 1-pixel gaps with NumPy; Maa locates connected card outlines.
        windows = np.lib.stride_tricks.sliding_window_view
        mask = grayscale(image) < 90
        dilated = windows(np.pad(mask, 1), (3, 3)).any(axis=(-2, -1))
        closed = windows(np.pad(dilated, 1, constant_values=True), (3, 3)).all(axis=(-2, -1))
        binary = np.repeat((closed.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
        result = self.context.run_recognition(CARD_NODE, binary)
        boxes = [tuple(row.box) for row in result.all_results] if result else []
        return [box for box in boxes if 80 < box[2] < 130 and 105 < box[3] < 162
                and 350 < box[0] < 940]

    def click(self, x, y, wait_for=None):
        width, height = self.controller.cached_image.shape[1], self.controller.cached_image.shape[0]
        result = self.context.run_task(CLICK_NODE, {CLICK_NODE: {
            "action": {"param": {
                "target": [round(x * width / 1280), round(y * height / 720), 1, 1]}},
            "next": [wait_for] if wait_for else []}})
        if not (result and result.status.succeeded):
            raise ValueError("预设页面点击未执行")

    def swipe(self, upward=True):
        width, height = self.controller.cached_image.shape[1], self.controller.cached_image.shape[0]
        # About one card row, with overlap to retain both its name and level.
        y1, y2 = (626, 455) if upward else (455, 626)
        result = self.context.run_task(SWIPE_NODE, {SWIPE_NODE: {"action": {"param": {
            "begin": [round(926 * width / 1280), round(y1 * height / 720)],
            "end": [round(926 * width / 1280), round(y2 * height / 720)]}}}})
        if not (result and result.status.succeeded):
            raise ValueError("预设页面滚动未执行")

    def return_to_party(self):
        result = self.context.run_task(RETURN_NODE)
        if not (result and result.status.succeeded):
            raise ValueError("预设已读取，但未返回编队页面")

    def read(self):
        image = self.screenshot()
        buttons = [r for r in self.ocr(image, roi=(1080, 0, 200, 78))
                   if self.locale["button"] in clean(r.text)]
        if not buttons:
            raise ValueError(f"编队页未找到右上角的{self.locale['button']}")
        self.click(buttons[0].cx, buttons[0].cy, wait_for=LIST_NODE)
        previous = None
        for _ in range(12):
            if self.context.tasker.stopping:
                raise ValueError("任务已停止")
            image = self.screenshot()
            words = self.ocr(image)
            selected = [r for r in words if re.search(self.locale["selected"], clean(r.text))]
            if selected:
                self.click(selected[0].cx, selected[0].cy + 42, wait_for=DETAIL_NODE)
                break
            view = grayscale(image[95:615:3, 50:1250:3])
            if same_view(previous, view):
                raise ValueError("未设置当前使用的预设，请先在游戏内选择预设并使用，再重新开始任务")
            previous = view
            self.swipe()
        else:
            raise ValueError("未设置当前使用的预设，请先在游戏内选择预设并使用，再重新开始任务")
        image = self.screenshot()
        # The switch is off when its white knob is on the left.
        switch = grayscale(image[95:121, 1208:1257])
        yy, xx = np.where(switch > 235)
        toggled = bool(len(xx) and float(np.median(xx)) > 26)
        if toggled and not self.context.tasker.stopping:
            self.click(1231, 108)
        try:
            preset = self.read_detail()
        except Exception:
            try:
                self.finish_detail(toggled)
            except Exception as cleanup_error:
                self.logger.debug("预设读取失败后未能恢复界面：%s", cleanup_error)
            raise
        self.finish_detail(toggled)
        return preset

    def finish_detail(self, toggled):
        if self.context.tasker.stopping:
            return
        if toggled:
            self.click(1231, 108)
        if not self.context.tasker.stopping:
            self.return_to_party()

    def read_detail(self):
        accumulator = Accumulator(self.catalog)
        # A newly opened detail view is normally already at the top.
        previous = None
        for _ in range(12):
            image = self.screenshot()
            words = self.ocr(image)
            accumulator.read_sidebar(words)
            if any(role == 1 for _, role in accumulator._headers(words)):
                break
            view = grayscale(image[95:686:3, 295:942:3])
            if same_view(previous, view):
                break
            previous = view
            self.swipe(upward=False)
        last_view = None
        upward = True
        for _ in range(28):
            if self.context.tasker.stopping:
                raise ValueError("任务已停止")
            image = self.screenshot()
            words = self.ocr(image)
            accumulator.feed(image, words, self.level, self.card_rects(image))
            self.logger.debug("[神经网络预设] 第%d帧，已读取%s，目标%s", accumulator.frames,
                              accumulator.observed_totals(), accumulator.totals)
            if accumulator.complete:
                break
            view = grayscale(image[95:686:3, 295:942:3])
            if same_view(last_view, view):
                if not upward:
                    break
                # Revisit earlier rows once if a name or level was missed.
                upward = False
                last_view = None
                self.swipe(upward=False)
                continue
            last_view = view
            self.swipe(upward=upward)
        return accumulator.compile()
