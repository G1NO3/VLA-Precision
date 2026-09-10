# 本机训练环境配置记录

配置日期：2026-09-09。代码基点：`75eb56a35ca6e8c3ee0a0a3d41b99e88be469b81`。
用途：本机 GPU 上的 Stage I SFT、Stage II ACoB 和离线数据预处理。

## 日常使用

在 Bash 中运行：

```bash
cd ~/pipette/VLA-Precision
source scripts/activate_training_env.sh
python main.py --help
```

激活脚本加载 `.venv`，并将 `.runtime/lib` 加入当前 shell 的
`LD_LIBRARY_PATH`，让 TorchCodec 找到 FFmpeg。它还默认设置
`XLA_PYTHON_CLIENT_PREALLOCATE=false`，避免单 GPU 上多个进程各自提前占满显存。
这是内存分配策略，不保证完整模型和指定 batch 能装入显存。

使用项目文档中的 uv 命令时保留 `uv run --no-sync`；
重新同步使用 `bash scripts/setup_training_env.sh`，或者
`uv sync --frozen --group stage2`。
不要直接运行不带依赖组的 `uv sync`，因为上游 `default-groups=[]`，
它会移除未选中的训练依赖。

## 已安装版本和目录

| 项目 | 本机设置 |
|---|---|
| Python | 3.11.16，`.venv` |
| uv | 0.12.5，已有用户级安装 |
| 依赖组 | `stage2`，已包含 `stage1` |
| GPU | NVIDIA RTX PRO 5000 Blackwell，sm_120，约 48GB |
| 驱动 | 610.43.02 |
| PyTorch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| JAX / jaxlib / CUDA plugin | 0.5.3 |
| Flax / NumPy | 0.10.2 / 1.26.4 |
| TorchCodec / PyAV / OpenCV | 0.5 / 15.1.0 / 4.11.0 |
| FFmpeg | 7.1.1，独立的 `.runtime` conda prefix |
| Python 包缓存 | `../.cache/uv` |
| uv 管理的 Python | `../.cache/uv-python` |
| FFmpeg conda 包缓存 | `../.conda-pkgs` |

`.venv` 约 8.9GB，`.runtime` 约 586MB；包缓存另外占空间。
`.runtime` 只提供原生运行库，不要将它作为 Python 训练环境激活。

OpenPI、LeRobot、AgentLace 保持上游锁定的 Git commit：
- OpenPI：`2d70d966582e711128ad8358d8dbf23d2cc3d658`
- LeRobot：`d6ea3bbce0fc8c75138f02639ac4ceaf12a67829`
- AgentLace：`cf2c337c5e3694cdbfc14831b239bd657bc4894d`

## 相对上游的配置修正

1. 上游锁文件安装 PyTorch 2.7.1 CUDA 12.6，实测在本机报
   `no kernel image is available for execution on the device`。
   在 `pyproject.toml` 指定官方 CUDA 12.8 index，固定同一版本的
   torch / torchvision，并更新 `uv.lock` 中相应 CUDA 依赖。
   JAX、Flax、OpenPI 等算法依赖版本保持不变。
   [PyTorch 官方 Blackwell / CUDA 12.8 说明](https://pytorch.org/blog/pytorch-2-7/)。

2. 本机没有系统 C 编译器和 Linux 输入头文件。安装脚本使用已有
   `../.conda-envs/starVLA` 的 GCC 和 sysroot 编译 LeRobot 间接依赖的 evdev。
   只读取工具链，不激活或修改原有 Python 环境。
   另一台机器可使用系统编译工具与头文件，或通过
   `VLA_BUILD_TOOLCHAIN` 指定等效 conda 工具链目录。

3. TorchCodec 初次导入缺少 FFmpeg 动态库。安装了本地
   `.runtime`，完整原生包清单在
   [runtime-linux-64.lock](../scripts/runtime-linux-64.lock)。
   [TorchCodec 官方 FFmpeg 安装说明](https://github.com/meta-pytorch/torchcodec/tree/v0.5.0#installing-torchcodec)。
   保留了上游 TorchCodec 0.5；已实测导入和三路视频解码。

## 重建和自检

```bash
cd ~/pipette/VLA-Precision
bash scripts/setup_training_env.sh
source scripts/activate_training_env.sh

python scripts/check_training_env.py \
  --video-root ../datasets/g1-pipette-3view-hgdagger-20260904-bc-30hz/videos/chunk-000
```

没有本地示范数据时可省略 `--video-root`，其余检查仍会执行。
重建脚本需要 uv 和 conda；默认使用 `../.miniconda3/bin/conda`，
可通过 `VLA_CONDA_EXE` 指定其它 conda 可执行文件。
原生包清单针对 Linux x86_64。

本次完整自检通过，日志为 `.cache/environment-smoke.log`：

- Stage I 训练、ACoB agent、Actor、Learner、数据预处理实际模块导入。
- JAX GPU 卷积、JIT、梯度和 Optax optimizer 更新。
- PyTorch GPU 矩阵乘、卷积与反向传播，以及 torchvision CUDA NMS。
- Stage I / II 示例配置解析、CLI 帮助入口。
- LeRobot 默认 TorchCodec 后端读取真实 episode 0 三路视频，每路三个时间点。
  ego 输出 `(3,3,720,1280)`，两路其它视角输出 `(3,3,1080,1080)`。
  此检查仅验证解码；没有改变或验证训练时的相机重命名。

## 保留的上游元数据告警

`uv pip check --python .venv/bin/python` 仍报告两项：

- AgentLace 依赖旧的 `typing` backport，其声明的 Python 范围不包含 3.11。
  Python 3.11 使用标准库 typing，当前实际训练模块导入通过。
- LeRobot 声明需要 `opencv-python-headless`，而本仓库已有
  `opencv-python-headless; sys_platform == 'never'` override，
  实际安装的是 OpenCV GUI wheel。没有同时安装两个提供同名 cv2 文件的 wheel；
  本次图像与视频检查通过。

此外，部分上游 Pydantic Field 属性在导入时产生 warning。
这些是保留并记录的告警，不能把此次安装说成 `uv pip check` 零告警。

## 验证范围

本次完成训练环境安装和 smoke checks，没有下载 PI0.5 基座权重或启动完整训练。
没有安装完整 `real-robot` 依赖组，也没有连接机器人、相机或启动控制服务。
现有 VLAPolicyBridge、TWIST2 和其它 Python 环境没有修改。

上游示例仍指向 UR 机器人和多 GPU 配置；本机只有 GPU 0。
G1 的模型输入、动作语义、数据转换、单 GPU batch / 内存预算和机器人适配
需在下一阶段配置，当前示例 YAML 不能视为已完成的 pipette 真机任务配置。

## Pipette native JAX SFT

See [PIPETTE_JAX_SFT.md](PIPETTE_JAX_SFT.md) for the prepared data, native pi05 configuration, validation results, and measured full-SFT memory limitation.
