"""Match visible economic event consequences to the frozen event action IDs.

The complete visible menu must have one unambiguous match. Quiz answers remain
with Maa's existing answer table. No question title or choice order is assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import re
import unicodedata


@dataclass(frozen=True)
class Effect:
    kind: str
    amount: int = 0
    quality: str = ""
    scope: str = ""
    chance: int = 0


def _text(value):
    return unicodedata.normalize("NFKC", str(value or "")).lower().replace("，", ",")


def _has(text, pattern):
    return re.search(pattern, text, re.I) is not None


def _effect(text: str) -> Effect | None:
    text = _text(text)
    if not text or _has(text, r"正确回答|正解すると|correct answer"):
        return None
    potential = _has(text, r"潜能|潛能|素質|potential")
    note = _has(text, r"音符|musical note|random note")
    hp = _has(text, r"生命|hp|体力|生命値")
    quality = ("rare" if _has(text, r"稀有|虹|rare") else
               "common" if _has(text, r"普通|common|normal potential|素質[（(](?:金|青)[）)]") else "")
    scope = ("support" if _has(text, r"援护|援護|支援|support|assist") else
             "main" if _has(text, r"主控|主角|主役|操作中|main|active trekker") else "")
    chance = (33 if _has(text, r"33%") else
              50 if _has(text, r"50%\s*(?:概率|機率|几率|chance|の確率|获得|獲得|失去|恢复|回復|[)）])|概率[^,]*50%|[（(]50%") else 0)

    if _has(text, r"随机变化|隨機變化|ランダムで変動|randomly change"):
        return Effect("change", int(potential), "", "hp" if hp else "")
    if _has(text, r"650") and _has(text, r"200"):
        return Effect("coin_gamble", 650)
    if _has(text, r"200") and _has(text, r"100") and chance:
        return Effect("coin_gamble", 200)

    # A coin glyph is often omitted by OCR (Japanese: を100枚消費).
    paid_coin = re.search(r"(?:消耗|消費|spend)\s*(?:辉光币|輝光幣|coins?)?\s*[x×]?\s*(\d+)\s*(?:枚|金币|金幣|coins?|to\b|,)", text)
    jp_coin = re.search(r"(?:を|×|x)\s*(\d+)\s*枚?\s*消費", text)
    paid_coin = jp_coin or paid_coin
    # Chinese strings may place the coin icon between the verb and quantity.
    if paid_coin is None and not note and not hp:
        paid_coin = re.search(r"(?:消耗|消費|spend)\s*[^\d]{0,8}(\d+)(?!\d|\s*%)", text)
    cost = int(paid_coin.group(1)) if paid_coin else 0

    hp_cost = re.search(r"(?:消耗|消費|spend)\s*(?:最大|max\s*)?(?:hp)?\s*(\d+)\s*%", text)
    hp_cost = hp_cost or re.search(r"(?:最大)?hp(?:を)?\s*(\d+)\s*%\s*(?:消費|消耗)", text)
    if potential and chance == 33:
        return Effect("chance_potential", 0, quality, scope, 33)
    if hp_cost and potential:
        return Effect("hp_potential", int(hp_cost.group(1)), quality, scope, chance)
    if potential and chance == 50:
        if re.search(r"100", text) and not hp:
            return Effect("coin_or_potential", 100, quality, scope, 50)
        if re.search(r"120", text) and not hp:
            return Effect("coin_or_potential", 120, quality, scope, 50)
        return Effect("hp_or_potential", 0, quality, scope, 50)
    if potential and cost:
        return Effect("buy_potential", cost, quality, scope)
    if potential and note and _has(text, r"消耗|消費|spend"):
        return Effect("notes_potential", 10, quality, scope)
    if potential and not _has(text, r"消耗|消費|spend"):
        return Effect("free_potential", 0, quality, scope)

    if hp and _has(text, r"恢复|恢復|回復|restore|recover"):
        if cost:
            return Effect("buy_hp", cost)
        if chance and note:
            return Effect("hp_or_notes", 0, "", "", 50)
        return Effect("heal", 20)
    if note and cost:
        return Effect("buy_notes", cost)
    if note and _has(text, r"150") and _has(text, r"消耗|消費|spend"):
        return Effect("notes_coin", 150)
    if note and _has(text, r"獲得|获得|獲得|obtain|gain"):
        return Effect("free_notes", 5)
    if _has(text, r"(?:获得|獲得|obtain|gain)[^\d]{0,12}30|(?:を|×|x)\s*30\s*枚?\s*獲得"):
        return Effect("coin", 30)
    return None


def E(kind, amount=0, quality="", scope="", chance=0):
    return Effect(kind, amount, quality, scope, chance)


MENUS = {
    101: {10101:E("buy_potential",100,"common"),10102:E("buy_potential",120,"regular"),10103:E("coin",30)},
    102: {10201:E("buy_potential",120,"regular","support"),10202:E("buy_potential",160,"regular","main"),10203:E("buy_potential",200,"rare"),10204:E("coin",30)},
    103: {10301:E("notes_coin",150),10302:E("coin",30)},
    104: {10401:E("notes_potential",10,"rare"),10402:E("buy_potential",200,"rare"),10403:E("coin",30)},
    105: {10501:E("coin_gamble",200),10502:E("coin_gamble",650),10503:E("coin",30)},
    106: {10601:E("hp_potential",20,"rare","",50),10602:E("hp_potential",40,"common"),10603:E("coin",30)},
    109: {10901:E("coin_or_potential",100,"common","",50),10902:E("buy_potential",120,"regular")},
    110: {11001:E("coin_or_potential",120,"rare","",50),11002:E("buy_potential",160,"rare")},
    111: {11101:E("heal",20),11102:E("hp_potential",20,"common")},
    112: {11201:E("heal",20),11202:E("hp_potential",30,"regular")},
    113: {11301:E("heal",20),11302:E("hp_potential",30,"regular"),11303:E("hp_potential",50,"rare")},
    117: {11701:E("change",0,"","hp"),11702:E("coin",30)},
    118: {11801:E("change",1,"","hp"),11802:E("coin",30)},
    119: {11901:E("change",1),11902:E("coin",30)},
    120: {12001:E("buy_notes",50),12002:E("hp_or_notes",0,"","",50)},
    121: {12101:E("buy_potential",100,"regular"),12102:E("hp_or_potential",0,"regular","",50)},
    122: {12201:E("buy_potential",180,"rare"),12202:E("hp_or_potential",0,"rare","",50)},
    123: {12301:E("buy_hp",50),12302:E("hp_potential",30,"regular")},
    124: {12401:E("chance_potential",0,"regular","",33),12402:E("coin",30)},
    125: {12501:E("chance_potential",0,"common","",33),12502:E("chance_potential",0,"rare","",33),12503:E("coin",30)},
    126: {12601:E("free_potential",0,"common","support"),12602:E("heal",20)},
    127: {12701:E("free_potential",0,"regular","support"),12702:E("free_notes",5)},
    128: {12801:E("free_potential",0,"rare","support"),12802:E("coin",30)},
}


def _score(observed, expected):
    if (observed.kind, observed.amount, observed.chance) != (expected.kind, expected.amount, expected.chance):
        return None
    if observed.scope and observed.scope != expected.scope:
        return None
    if observed.quality and observed.quality != expected.quality:
        return None
    return 2 + int(observed.scope == expected.scope) + int(observed.quality == expected.quality)


def match_event(question: str, choices: list, lang: str = "jp"):
    """Return (event ID, zero-based visible choice -> option ID), or None."""
    effects = []
    for choice in choices:
        consequence = choice.get("consequence", "") if isinstance(choice, dict) else choice.consequence
        effect = _effect(consequence)
        if effect is None:
            return None
        effects.append(effect)
    if len(effects) < 2:
        return None
    matches = []
    for eid, options in MENUS.items():
        if len(effects) != len(options):
            continue
        for ids in permutations(options):
            scores = [_score(actual, options[oid]) for actual, oid in zip(effects, ids)]
            if None not in scores:
                matches.append((sum(scores), eid, dict(enumerate(ids))))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        return None
    return matches[0][1], matches[0][2]
