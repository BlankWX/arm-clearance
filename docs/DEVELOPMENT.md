# 开发、测试与构建

## 环境

运行插件需要 Blender 内置的 Python、NumPy、`bpy`、`bmesh`、`mathutils`。没有额外的 pip 运行时依赖或联网服务。v0.2.7 的验证环境为 Windows / Blender 5.2.2 LTS；元数据最低版本为 Blender 4.4，但旧版本兼容性尚未完成验证。完整的历史测试与片段回归范围见 [验证记录](VALIDATION.md)。

纯数值测试可以使用装有 NumPy 的 Python；烘焙、保存和插件注册测试必须在独立 Blender 进程中运行。以下命令在仓库根目录执行，`blender` 和 `python` 请换成本机相应程序。

## 构建

```text
python tools/build_release.py
```

生成：

- `dist/arm_clearance-v0.2.7.zip`：Blender 插件安装包，包含完整插件模块和许可证。
- `dist/arm-clearance-v0.2.7-source.zip`：仓库源码归档，可解压后建立 Git 仓库。
- `dist/SHA256SUMS.txt`：两份 ZIP 的 SHA-256。
- `dist/release-manifest.json`：模块、许可证与归档的哈希及版本。

上述文件名对应 v0.2.7。构建应使用已核验的源码，并再次验证实际安装 ZIP；旧版本归档不代表新构建包已通过验证。

构建只读取明确列出的源码、文档、测试和工具，不打包 `build`、私有素材、`.git`、缓存或已生成的 ZIP。文件顺序和 ZIP 时间戳固定，输入字节相同会产生相同归档。

## 合成回归测试

```text
python tests/test_temporal.py
python tests/test_console_progress.py
python tests/test_body_field.py
python tests/test_pair_weights.py
blender --background --factory-startup --python-exit-code 1 --python tests/test_progress_solver.py
blender --background --factory-startup --python-exit-code 1 --python tests/test_stationary.py
blender --background --factory-startup --python-exit-code 1 --python tests/test_modal_progress.py
blender --background --factory-startup --python-exit-code 1 --python tests/test_baking.py
blender --background --factory-startup --python-exit-code 1 --python tests/test_persistence.py
blender --background --factory-startup --python-exit-code 1 --python tests/test_distribution.py
```

先构建，再执行最后一项，以确保实际测试的是 ZIP 内的插件。各检查失败时使用非零退出码；生成文件位于 `build/test-results`，该目录不纳入源码归档或 Git。

| 测试 | 范围 |
| --- | --- |
| `test_temporal.py` | 时间矩阵五对角投影的等价性、边界、紧支撑、特征分解和禁止稠密帧单位阵 |
| `test_console_progress.py` | 8 项确定性日志回归：节流、立即刷新、准确终态、输出容错、无后台线程 |
| `test_stationary.py` | 20 项完整静止、符号等价、慢动作、局部关节活动、控制点投影与 warm 修复回归 |
| `test_progress_solver.py` | 80 帧真实优化器的阶段、线搜索、三种控制范围、warm/缓存与分组 |
| `test_modal_progress.py` | 7 项假 context 操作器测试：早取消、启动失败、关闭异常、幂等；不调用 GUI |
| `test_body_field.py` | 身体距离场、导数、曲率与三角覆盖的合成数值检查 |
| `test_pair_weights.py` | 配对接触能量、力与力矩的数值差分检查 |
| `test_baking.py` | 合成骨架烘焙、原动作及非目标曲线、范围边界、异常回滚 |
| `test_persistence.py` | 七种生命周期场景、十三次独立重开；真实 Action 烘焙与保存，碰撞模型使用合成替身 |
| `test_distribution.py` | 从 ZIP 导入全部内部依赖，注册、对比、六骨合成烘焙和卸载检查；当前测试限制 Blender 5.2 |

公开测试不依赖私有角色工程。真实模型的完整动作与性能验证来自开发期案例，证据范围见 [验证说明](VALIDATION.md)，不能用这些合成测试代替真实动画预览验收。

## 控制台日志约定（v0.2.4）

日志直接从 Blender 主线程写入标准输出并刷新，不创建后台线程或定时心跳。Windows 系统控制台只负责展示这些输出；侧栏提供手动显示按钮，高级设置控制烘焙启动时是否自动显示。非 Windows 环境不依赖 Windows 控制台操作。

保持所有求解数值、分组、线搜索接受条件、目标函数调用顺序和现有 generator 的产出位置不变。额外进度仅报告已经发生或即将执行的阶段；单次底层计算尚未返回时不推算虚假百分比。普通步骤约一秒节流，阶段、组数、修复轮次和终态立即输出，包含已用时间和适用的当前步。

每次烘焙独立记录状态。正常完成须区分 Action 的 `passed` 与 `needs_review`，取消与异常分别记录；日志不能把 generator 正常结束或完整烘焙已形成当作复核通过。检查首次验证失败、采样/求解/复核取消、烘焙已保留后的异常及重复运行，确保清理路径只输出一次准确终态。输出不可用时，日志故障不得改变原任务的求解或结果。

GitHub 上传、标签和 Release 单独执行；本地构建不会自动发布。v0.2.7 发布说明、附件清单和 v0.2.3 历史记录见 [发布说明](RELEASE.md)。

## 模块

UI 入口位于 `arm_clearance/__init__.py`，当前流程位于 `pipeline_v2.py`。`solver_v2.py`、`solver_v2_contacts.py`、`solver_v2_derivatives.py` 执行求解；`body_field.py`、`arm_geometry.py`、`hand_geometry.py` 建立碰撞代理；`full_surface.py`、`surface_repair.py` 负责真实表面复核与修复；`baking.py` 处理 Action。

`pipeline.py` 和 `solver.py` 仍包含当前流程使用的基础函数，必须一并保留。请勿根据文件名中的版本标记删除它们。

仓库名称为 `arm-clearance`，作者署名为 `BlankWX`。当前 checkout 的目录名 `github_release_v0.2.3` 保持不变，目录名称不代表当前源码版本；构建版本从 `bl_info` 读取。当前代码为 v0.2.7，新增静止段共享控制约束；默认参数及碰撞、烘焙规则保持既有实现；未采用的实验不进入发布包。后续代码更改应更新版本、验证记录并重新构建发布包。

时间预条件矩阵由 temporal.py 计算。必须保留 (B.T @ C) @ B 的左结合顺序与原浮点非对称性；改变求和次序即使矩阵仅差 1e-13，也可能使非线性求解轨迹分岔。当前环境需检查逐位一致及零启动真实目标轨迹，不能只检查 allclose 或 warm 起点。


静止段识别与控制投影位于 `stationary.py`，详见 [设计说明](STATIONARY.md)。无静止段时保留旧求解路径；不能为通过检测而跳过实际表面复核。

## v0.2.6 发布整理边界（历史）

发布整理仅更新文档和归档，15 个插件模块及许可证必须与已验证并安装的 v0.2.6 校验清单完全一致。安装包应保持相同 SHA-256；源码包因说明文档更新可产生新校验和，必须同步更新 `release-manifest.json` 与 `SHA256SUMS.txt`。不要覆盖历史验证目录中的原归档与清单。

本轮无需重新烘焙动画。可使用 `tests/test_distribution.py -- --zip <安装包路径> --report <检查结果路径> --skip-synthetic` 在独立 `--factory-startup` Blender 中核对实际 ZIP 的导入、注册及卸载，不打开用户工程。不得为发布整理重新运行 5772 帧；私有片段、缓存与实验源码不进入公开仓库或归档。
