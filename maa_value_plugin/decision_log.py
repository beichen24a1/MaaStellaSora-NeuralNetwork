"""Human-readable terminal forecasts and changes from the previous forecast."""
from __future__ import annotations

def action_name(action):
    kind = action.kind
    if kind == "SELECT":
        return f"潜能{action.index}"
    if kind == "REROLL":
        return f"刷新潜能（{action.price}金币）"
    if kind == "BUY":
        return f"购买{'指定' if action.type == 2 else '普通'}饮品（{action.price}金币）"
    if kind == "ENHANCE":
        return "免费强化" if not action.price else f"强化（{action.price}金币）"
    if kind == "RESTOCK":
        return f"刷新商店（{action.price}金币）"
    if kind == "LEAVE":
        return "离开商店"
    if kind == "EVENT_OPTION":
        return f"事件选项{action.index % 100}"
    if kind == "SKIP_EVENT":
        return "跳过事件"
    return {"FREE_ENHANCE": "免费强化", "ADVANCE": "继续爬塔"}.get(kind, kind)


def forecast_rows(decision, actions, ordinary_score, previous_total, labels=None):
    labels = labels or {}
    rows = []
    for action, item in zip(actions, decision["values"]):
        total = 6 + ordinary_score + item["expected_remaining_ordinary_return"]
        rows.append({"action": item["action"], "label": labels.get(action, action_name(action)),
                     "expected_total": float(total), "change": float(total - previous_total),
                     "selected": item["action_record"] == decision["action_record"]})
    return rows


def log_forecast(logger, selected_total, rows):
    options = [f"选中后整局预期 {selected_total:.2f}"]
    for row in rows:
        change = row["change"] if abs(row["change"]) >= .005 else 0.0
        label = " ".join(row["label"].split())
        options.append(f"{'→ ' if row['selected'] else ''}{label} {change:+.2f}")
    # MFA decodes Unicode escapes after framing stdout records. Keep one
    # physical record, then render Markdown hard breaks inside its log item.
    logger.info("[神经网络] %s", r"  \u000a".join(options))
