# ComfyUI-AccelLog

ComfyUI 加速日志补丁：**每次采样结束，自动在日志里汇报这一刀实际用的是哪个注意力加速后端、用了多少次、以及是谁决定的**。

- 不注册任何工作流节点
- 不依赖任何第三方库（只用 Python 标准库 + ComfyUI 内部接口）
- 不改动 ComfyUI 核心文件——**删掉整个文件夹即完全移除**

## 效果

每次采样（CFGGuider.sample）结束时输出一行：

```
[加速日志] SageAttention×160 | 来源: XB sage 节点
[加速日志] Kitchen INT8×622 | 来源: T8 节点(仅块缓存)
[加速日志] Kitchen INT8×280 | 来源: Comfy Kitchen INT8 (--use-ck-attention)
[加速日志] PyTorch注意力×560 | 来源: 顶栏切换(PyTorch (SDPA))
[加速日志] 未检测到已知加速（核心默认路径，未计数） | 来源: 未识别
```

- **×N**：本次采样中该内核被调用的次数
- **来源**：这次为什么用了它——启动参数默认 / 工作流里的 XB sage 节点 / BlockCache-T8 节点 / ModelAttentionBackend 节点 / 顶栏切换（配合 ComfyUI-AccelSwitch 时识别为「顶栏切换」）
- 覆盖 sageattention、comfy-kitchen、flash、xformers、pytorch、sub_quad、split、basic 及注意力注册表内的全部后端

## 安装

把本文件夹放到 ComfyUI 的 `custom_nodes/` 目录下，重启 ComfyUI：

```bash
cd ComfyUI/custom_nodes
git clone <本仓库地址> ComfyUI-AccelLog
```

或者用 ComfyUI-Manager 的「Install via Git URL」粘贴仓库地址。

## 使用

- 日志输出在 ComfyUI 控制台 / `user/comfyui.log`，过滤关键词：**`加速日志`**
- 启动时会打印一行确认：`注意力加速日志已启用 | 采样钩子: CFGGuider.sample | 启动默认: ...`
- 想关闭：设置环境变量 `ACCEL_LOG=0` 后启动

## 兼容性

- 开发验证环境：ComfyUI 0.43
- 依赖的钩子点（attention 模块函数 / REGISTERED_ATTENTION_FUNCTIONS / CFGGuider.sample）多年未变，其他 0.3x~0.4x 版本通常也能用
- 接口对不上时**不会报错崩溃**：对应项静默跳过，启动日志里会有「跳过项」说明

## 卸载

删除本文件夹，重启 ComfyUI。

## License

MIT
