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

## 许可

第三方许可和来源说明见 `maa_value_plugin/licenses.json` 与 `maa_value_plugin/critic/vendor/SOURCE.json`。
