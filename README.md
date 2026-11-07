# MaaStellaSora NeuralNetwork 插件

MaaStellaSora 项目额外开发的神经网络数值模型插件。

- `maa_value_plugin/`：可编辑源码、模型和相关数据。
- `NeuralNetwork.cmd`：Windows 自解压安装包。
- `build_installer.py`：从 `maa_value_plugin/` 重新生成安装包。

## 安装

将 `NeuralNetwork.cmd` 放到 `MFAAvalonia.exe` 所在目录，并确保该目录下有官方 Maa 安装包提供的 `python/python.exe`，然后双击运行。

## 修改后重新打包

在本机安装 Python 3 后，在仓库根目录运行：

```bash
python3 build_installer.py
```

公开版本的 `maa_value_plugin/upload_settings.json` 不包含上传令牌。需要使用数据上传功能时，请在本地配置令牌，不要提交到仓库。

## 离线联合模型研究

[`research/concerto_joint/`](research/concerto_joint/README.md) 提供音符／协奏与普通潜能联合决策的独立实验模块，包含训练、冻结权重、公共 JSON 推理接口和完整留出评测。它不更改插件当前默认模型；成绩只适用于文档列明的模拟规则与实验目标。

原 `joint_v2` 因音符价格 OCR 污染、`joint_v3_prices_clean` 因环境缺陷作废。当前活动运行是 `joint_v4_env_fixed`，使用新规则重新生成标签、训练和评测；推荐状态以当前报告为准。

已修复音符事件动作过滤、普通目标剪枝和整架折扣约束，见 [环境修复](research/concerto_joint/ENVIRONMENT_FIX.md)；实际掉落和调整规则仍待实测验证。

## 许可

第三方许可和来源说明见 `maa_value_plugin/licenses.json` 与 `maa_value_plugin/critic/vendor/SOURCE.json`。
