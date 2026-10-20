# 开发、测试与构建

## 环境

运行插件需要 Blender 内置的 Python、NumPy、`bpy`、`bmesh`、`mathutils`。没有额外的 pip 运行时依赖或联网服务。发布验证环境为 Windows / Blender 5.2.2 LTS。

纯数值测试可以使用装有 NumPy 的 Python；烘焙、保存和插件注册测试必须在独立 Blender 进程中运行。以下命令在仓库根目录执行，`blender` 和 `python` 请换成本机相应程序。

## 构建

```text
python tools/build_release.py
```

生成：

- `dist/arm_clearance-v0.2.3.zip`：Blender 插件安装包，包含完整插件模块和许可证。
- `dist/arm-clearance-v0.2.3-source.zip`：仓库源码归档，可解压后建立 Git 仓库。
- `dist/SHA256SUMS.txt`：两份 ZIP 的 SHA-256。
- `dist/release-manifest.json`：模块、许可证与归档的哈希及版本。

构建只读取明确列出的源码、文档、测试和工具，不打包 `build`、私有素材、`.git`、缓存或已生成的 ZIP。文件顺序和 ZIP 时间戳固定，输入字节相同会产生相同归档。

## 合成回归测试

```text
python tests/test_body_field.py
python tests/test_pair_weights.py
blender --background --factory-startup --python-exit-code 1 --python tests/test_baking.py
blender --background --factory-startup --python-exit-code 1 --python tests/test_persistence.py
blender --background --factory-startup --python-exit-code 1 --python tests/test_distribution.py
```

先构建，再执行最后一项，以确保实际测试的是 ZIP 内的插件。各检查失败时使用非零退出码；生成文件位于 `build/test-results`，该目录不纳入源码归档或 Git。

| 测试 | 范围 |
| --- | --- |
| `test_body_field.py` | 身体距离场、导数、曲率与三角覆盖的合成数值检查 |
| `test_pair_weights.py` | 配对接触能量、力与力矩的数值差分检查 |
| `test_baking.py` | 合成骨架烘焙、原动作及非目标曲线、范围边界、异常回滚 |
| `test_persistence.py` | 七种生命周期场景、十三次独立重开；真实 Action 烘焙与保存，碰撞模型使用合成替身 |
| `test_distribution.py` | 从 ZIP 导入全部内部依赖，注册、对比、六骨合成烘焙和卸载检查；当前测试限制 Blender 5.2 |

公开测试不依赖私有角色工程。真实模型的完整动作与性能验证来自开发期案例，证据范围见 [验证说明](VALIDATION.md)，不能用这些合成测试代替真实动画预览验收。

## 模块

UI 入口位于 `arm_clearance/__init__.py`，当前流程位于 `pipeline_v2.py`。`solver_v2.py`、`solver_v2_contacts.py`、`solver_v2_derivatives.py` 执行求解；`body_field.py`、`arm_geometry.py`、`hand_geometry.py` 建立碰撞代理；`full_surface.py`、`surface_repair.py` 负责真实表面复核与修复；`baking.py` 处理 Action。

`pipeline.py` 和 `solver.py` 仍包含当前流程使用的基础函数，必须一并保留。请勿根据文件名中的版本标记删除它们。

本地发布的仓库名称确定为 `arm-clearance`，作者署名为 `BlankWX`。这次整理仅更新 `arm_clearance/__init__.py` 的作者元数据，求解和烘焙逻辑保持已验证的 v0.2.3 原样。后续更改应更新版本、验证记录并重新构建发布包。
