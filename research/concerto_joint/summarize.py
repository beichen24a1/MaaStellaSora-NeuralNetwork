"""Make a short user-readable result from the completed machine report."""
import argparse
import json

from bootstrap import HERE


def summarize(run="joint_v4_env_fixed"):
    directory = HERE / "runs" / run
    report = json.loads((directory / "evaluation/report.json").read_text(encoding="utf-8"))
    model = json.loads((directory / "joint_critic.json").read_text(encoding="utf-8"))
    primary = report["conditions"]["primary"]
    labels = {"beta_only": "beta 不买音符", "greedy": "beta 加音符规则", "joint": "新联合模型"}
    lines = ["音符／协奏联合模型：离线实验结果", "",
             "已完成整局模拟、联合标签、训练、独立推理和留出评测。",
             "本报告中的分数是带明确假设的模拟结果，不代表实机成绩或伤害。", "",
             f"训练根局面 {model['train_roots']} 个；开发根局面 {model['dev_roots']} 个。",
             f"最终评测 {report['config_count']} 种未训练配置，每种 {report['rounds_per_primary_config']} 局、三个策略。",
             "偏好分 = 普通潜能 + 3×协奏等级和；3 只是本次实验选择。", "",
             "策略                 含核心潜能    协奏等级和    偏好分    音符花费"]
    for key in ("beta_only", "greedy", "joint"):
        p = primary["policies"][key]
        lines.append(f"{labels[key]:<16} {p['potential_total']['mean']:>9.2f} {p['concerto']['mean']:>11.2f} "
                     f"{p['objective']['mean']:>10.2f} {p['note_spent']['mean']:>10.2f}")
    pair = primary["joint_vs_greedy"]
    delta = pair["objective"]["paired_mean_delta"]
    ci = pair["objective"]["paired_seed_bootstrap_ci95"]
    lines.extend(("", f"相对音符规则，新模型偏好分变化 {delta:+.2f}，配对 95% 区间 [{ci[0]:+.2f}, {ci[1]:+.2f}]。",
                  "区间只描述本次固定配置面板的种子波动，不覆盖真实游戏规则误差。", ""))
    if report["promotion"]["recommended_over_greedy_in_primary_simulator"]:
        lines.append("主模拟环境的留出比较支持使用新联合模型；仍需结合下列规则敏感性判断适用范围。")
    else:
        lines.append("本轮证据不足以推荐替换音符规则。权重可用于继续实验，不能宣称已经提升。")
    lines.extend(("", "规则敏感性：新模型相对音符规则的偏好分变化"))
    for name, condition in report["conditions"].items():
        if name != "primary":
            p = condition["joint_vs_greedy"]
            if p["pairs"]:
                lines.append(f"  {name}: {p['objective']['paired_mean_delta']:+.2f}（{p['pairs']} 对）")
    lines.extend(("", f"模拟未覆盖异常：{len(report['failures'])} 条，详细记录保留在 report.json。",
                  "学习范围：商店买音符／潜能／强化／刷新／离开，以及事件取舍。",
                  "普通选牌仍由冻结 beta 完成，终局音符调整由所有策略共用的固定算法完成。",
                  "覆盖 61 件已精确匹配的双协奏秘纹；尚未建模实战伤害与技能触发价值。", "",
                  "使用入口见本目录上两级 README.md。",
                  "逐局数据见 evaluation/episodes.csv；规则、置信区间和对照结果见 evaluation/report.json。"))
    text = "\n".join(lines) + "\n"
    (directory / "结果说明.txt").write_text(text, encoding="utf-8")
    print(text)
    return directory / "结果说明.txt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="joint_v4_env_fixed")
    args = parser.parse_args()
    summarize(args.run)
