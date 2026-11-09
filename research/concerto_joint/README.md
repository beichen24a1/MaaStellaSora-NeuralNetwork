# 音符／协奏联合模型研究

**当前阶段：规则对齐与真实转移数据准备。没有活动联合检查点，暂不训练或交付模型。**

依据协作者的新研究资料，修正已确认的事件、菜单、大包与阶段供给规则。v2、v3、v4权重及报告保留为历史记录，当前加载器拒绝旧合同；v4的+2.29不作为真实游戏或本次修正规则后的提升证据。

- [本轮对齐、数据复核与限制](ALIGNMENT_20261108.md)
- [实机录像标注、库存对账及新差异（2026-11-10）](VIDEO_AUDIT_20261110.md)
- [另外两座塔补录：类别覆盖、隐藏音符及单独购买](TOWER_CAPTURE_20261110.md)
- [采集者的简单步骤](CAPTURE_GUIDE.md)
- [下一阶段完整联合价值合同（设计）](TRAINING_CONTRACT.md)
- [训练准备状态](data/TRAINING_READINESS.json)
- [最新47.6万条记录的结构覆盖扫描](data/real_note_coverage_20261108.json)
- [修正规则的沙盒诊断](data/environment_alignment_v5.json)
- [输入来源](DATA_SOURCES.md)

## 已完成

- 114错误答案不给音符，保留事件并禁用失败项；12002区分回血和音符分支。
- 公共快照不扩大可见菜单；沙盒107/108只生成两种不同指定音符项、随机项，以及108的金币项。
- 删除最小损失音符扣除。随机扣除核尚未知；测试用按存量加权、不放回抽样明确标注为未校准。
- 15枚大包按整架生成、最多一个。约80%是观察支持的近似频率，不把精确4/5或均匀位置当成已确认规则。
- 按秘纹ID与显式阶段查询1044条供给记录；完整入塔库存已含供给时不再叠加。
- 训练、标签生成、fit、整套实验、评测及打包均有准备状态门槛；隐式完整模拟被拒绝。

## 可运行的检查

在仓库根目录执行，只需NumPy：

```bash
python -m unittest discover -s research/concerto_joint -v
python research/concerto_joint/audit_environment.py
python research/concerto_joint/audit_real_note_coverage.py /local/real-game-data.zip --output /local/coverage.json
python research/concerto_joint/capture_records.py /local/note_transitions.jsonl --output /local/capture_audit.json
```

规则检查通过不等于训练准备通过。`train.py`、`run_experiment.py`与打包当前应明确拒绝执行，不会创建新标签或改写历史运行。

完整模拟只有显式选择 `rule_alignment.SANDBOX_PROFILE` 才能运行，供局部回归检查使用。该沙盒仍使用未校准的战斗掉落、随机音符分配、扣除、菜单组合和商品条件关系；回血只记分支，不模拟生命和生存，终局修改入口也未对齐。不能据此生成新一批号称已校准的训练样本。

公共快照保留可见菜单和实际库存；未知不填0。旧公共推理示例属于旧合同，当前拒绝加载。未来统一模型必须把普通选牌、购买、强化、刷新与事件一起纳入同一续局策略的价值判断；本轮没有冒充完成该训练器。

## 历史证据

- [价格污染审计](PRICE_AUDIT.md)：v2作废，确认的价格修复继续保留。
- [旧环境缺陷](ENVIRONMENT_AUDIT.md) 与 [上轮实现修复](ENVIRONMENT_FIX.md)：记录v3/v4的演变，不能代表完整状态空间已验证。
- [v4已知问题标记](runs/joint_v4_env_fixed/KNOWN_ISSUES.json)：旧权重和数值不改写为新环境成绩。

原潜能内核和冻结beta文件未修改。本模块仍专注模型与数据规范，不注册Maa回调，不实现采集Agent。原始玩家包、日志和视频不提交到公开仓库。
