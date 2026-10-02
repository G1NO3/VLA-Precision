# π0.5 五阶段任务远程服务器训练指南

本文面向在另一台 Linux GPU 服务器上操作的实验人员或 agent，从获取代码开始，说明如何用 237 条五阶段示教训练五个原生 OpenPI JAX π0.5 模型。按顺序完成环境安装、数据缓存、统计、预检、短训练、正式训练和备份。无需连接机器人，也无需运行 GUI、Redis、TWIST2、RLPD actor 或 learner。

代码仓库是 `G1NO3/VLA-Precision`，分支是 `jy-full-pi05`。训练实现基点为 `015f7b8`；本指南与该实现配套。它不是 starVLA 的 PyTorch π0.5，不使用 GR00T checkpoint 初始化，也不是之前的单手 tip attachment 模型。每个阶段独立从官方 `pi05_base` 初始化。

当前验证边界：本地已完成真实数值数据准备、五阶段统计、真实视频抽样解码、原生参数形状与抽象前向检查，以及 30 项针对性测试。尚未完成此五阶段配方的远程多 GPU 数值训练。下文的 GPU、内存和存储建议是容量规划，不是实测吞吐或显存保证；远程短训练是必须的验收步骤。

动作公式、划分依据和与 GR00T 的差异另见 [五阶段训练配方](PI05_FULLTASK_JAX_TRAINING_ZH.md)。本文专门讲如何在新服务器执行。

## 1 训练目标和默认配置

| 项目 | 当前实现 |
| --- | --- |
| 数据集 | `jren313/g1-pipette-2view-teleop0925-5task-eerel` |
| 固定数据 revision | `340ebabb48e6d1fb42f92178fba0f6ded87322df` |
| OpenPI commit | `2d70d966582e711128ad8358d8dbf23d2cc3d658`，由 `uv.lock` 固定 |
| 模型 | 原生 JAX/Flax π0.5，全参数微调，无 LoRA，无 EMA |
| 视觉 | 头部 RGB 与左腕；resize270，训练随机 crop256，验证中心 crop256，再 resize224 |
| 状态 | 59 维；训练时 0.8 概率整段归一化状态置零，再作为文本 token 输入 |
| 动作 | 30 × 32，数据 60 Hz，名义动作块长度 0.5 秒 |
| 优化 | 原生 AdamW，统一学习率 1e-5，warmup100，cosine 到 1e-6 |
| 默认设备 | 单节点，单 JAX 进程，4 GPU，FSDP4 |
| batch | 全局 128，不是每卡 128；没有配置梯度累积 |
| 预算 | 各阶段约 10 epoch 的样本数，不早停 |
| 验证 | 每阶段固定 480 个验证 chunks，约每 epoch 和最终一步评估 |
| 模型选择 | 最低验证 normalized MSE；同时记录物理单位误差与保持不动基线 |

| 阶段 | 任务 | 训练 episodes | 验证 episodes | 训练 anchors | batch128 时总 updates |
| --- | --- | ---: | ---: | ---: | ---: |
| p1 | 右手拿移液枪 | 54 | 9 | 36558 | 2857 |
| p2 | 左手拿试管 | 26 | 5 | 17290 | 1351 |
| p3 | 移液枪瞄准试管口 | 42 | 7 | 28149 | 2200 |
| p4 | 左手放回试管 | 26 | 5 | 17176 | 1342 |
| p5 | 右手放回移液枪 | 54 | 9 | 38568 | 3014 |

237 条是已切分的阶段 episodes，不是 237 条完整五阶段任务。p3 不是旧数据中的绿色枪头安装任务。数据按原录制划分，训练过滤静止腕部 chunks，验证不做此过滤；各阶段仅用自己的训练集计算统计。

不要随意改变以下数据契约：状态 59 维；动作 32 维；chunk30；头部加左腕两视图；腕部动作相对 chunk 起点的指令位姿；右手五通道是绝对寄存器；左拇指 p1 为绝对值、p2–p5 为相对寄存器差；14 维手臂关节角是绝对指令。状态不是动作标签，腕部标签来自 commanded joints 的 FK。

## 2 服务器和权限要求

建议起点是单节点 4 张 80 GB 级 GPU、至少 128 GB 主机 RAM，条件允许时使用 256 GB，以及 600 GB 到 1 TB 可用 SSD。此建议包含训练状态、数据和保存时临时空间，不保证 batch128 一定可运行。GPU 初始化、反向传播、验证都必须实测。48 GB 单卡不能因为小 batch 就假定能全参数训练。

本流程针对 Linux x86_64、Bash 和 Python 3.11。仓库锁定 JAX 0.5.3、Flax 0.10.2、Orbax 0.11.13、NumPy 1.26.4；Torch 2.7.1 CUDA12.8 仍作为数据依赖安装，但训练反向传播是 JAX。不要直接升级到最新版 JAX 或照搬最新版 OpenPI 的安装命令。

GPU 驱动由服务器管理员维护。CUDA wheel 不包含内核驱动；驱动必须同时适合 GPU 型号与锁定的软件。JAX 官方还提示系统 `LD_LIBRARY_PATH` 可能覆盖 wheel 的 CUDA 库；这里的 `.runtime/lib` 用于 FFmpeg，不应混入不兼容 CUDA 路径。见 [JAX 安装说明](https://docs.jax.dev/en/latest/installation.html)。

先检查服务器，GPU 命令应在实际分配到的计算节点执行：

```bash
ssh YOUR_USER@YOUR_SERVER
uname -m
nvidia-smi
nvidia-smi topo -m
free -h
df -h
command -v git uv conda gcc
```

需要 Git、uv、conda（仅构建 FFmpeg runtime）、C 编译工具与 Linux 输入头文件。Ubuntu 上若 evdev 编译缺失头文件，可请管理员安装 `build-essential` 和 `linux-libc-dev`。不需要 `twist2_deploy`、Isaac Sim、VR SDK 或机器人 DDS 环境。

未安装 uv 时可按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装，再重新登录以加载 PATH。conda 可使用集群提供的 Miniforge/Miniconda module，不要修改其他项目的环境。

网络需要能访问 GitHub、PyPI、PyTorch wheel 源、conda-forge、Hugging Face、Google Cloud Storage；W&B 在线监控另需访问 W&B。受限网络先在联网节点下载到共享磁盘，不要把令牌写进 YAML、脚本或 Git。

### 存储预算

| 内容 | 规划大小 |
| --- | --- |
| 原始视频和数值数据 | 约 10.6 GB 视频，另有 metadata/parquet |
| 共享 270 RGB 图像缓存 | 约 66 GiB，五阶段共用，不复制五份 |
| 共享数值缓存 | 本地约 61 MB |
| 原生基础权重 | 约十几 GB，以下载结果为准 |
| 每阶段最近完整训练状态 | 估计约 50 GB，包含 AdamW 状态 |
| 每阶段最佳推理权重 | 估计约 16 GB |
| 额外空间 | 环境、包缓存、日志、JAX 编译缓存，以及新旧 checkpoint 重叠保存空间 |

五阶段 last+best 再加数据可能需要数百 GB；额外的 smoke run 也会保存全尺寸模型。不要把这些写到小容量系统盘或未确认保留期限的临时 scratch。

## 3 目录和代码

除 SSH 和最终下载命令外，下文都在远程服务器执行。将 `/data/YOUR_USER/pipette` 改为你有写权限且容量足够的绝对路径。不要直接使用本机 `/home/jwang3617/...` 的 YAML。

```bash
export PIPETTE_ROOT=/data/YOUR_USER/pipette
mkdir -p "$PIPETTE_ROOT"
cd "$PIPETTE_ROOT"
git clone --branch jy-full-pi05 --single-branch \
  https://github.com/G1NO3/VLA-Precision.git VLA-Precision
git clone https://github.com/birbirll/VLAPolicyBridge.git VLAPolicyBridge
git -C VLAPolicyBridge checkout --detach e8d44200f05283c9530fcee1c120bd56fce14fb4
cd VLA-Precision
git status --short --branch
git rev-parse HEAD
git -C ../VLAPolicyBridge rev-parse HEAD
```

Bridge 固定为本地数据准备时核对的版本，仅用它的 `UrdfKinematics`、关节顺序和 vendored URDF。不要安装整个 Bridge GUI/机器人依赖。若该仓库需要权限，请用有读取权限的账号，或传输该 commit 的源码；不要从不明版本取另一个 URDF。

已有仓库时，不要覆盖有改动的工作区；先 `git status`，再在干净工作区 fetch 并切到相应分支，或新建独立 clone。新 clone 的远程名是 `origin`；原开发机曾使用 `fork`，远程名不是固定接口。

建议目录布局：

```text
pipette/
  VLA-Precision/                 训练代码、.venv、.runtime
  VLAPolicyBridge/               只用于 FK 和 URDF
  datasets/
    g1-pipette-2view-teleop0925-5task-eerel/   原始数据
    g1-pipette-fulltask-jax-v1/              共享缓存及 p1 到 p5 manifest
  models/openpi/                官方基础权重与 tokenizer
  assets/pipette-fulltask-jax/    五阶段各自的归一化统计
  outputs/pi05-fulltask-jax/     配置、预检、日志、W&B 本地文件
  checkpoints/pipette-fulltask-jax/
  .cache/                       uv、HF、JAX 等缓存
```

## 4 安装独立训练环境

以下只安装 Stage I 所需依赖组。当前 `setup_training_env.sh` 安装的是包含 Stage I 的 Stage II 全套；也可以用它，但远程纯 BC 不需要额外的 RL 包。两种方式选一种，不要在运行训练时反复同步环境。

```bash
cd "$PIPETTE_ROOT/VLA-Precision"
export UV_CACHE_DIR="$PIPETTE_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PIPETTE_ROOT/.cache/uv-python"
export CONDA_PKGS_DIRS="$PIPETTE_ROOT/.conda-pkgs"
export GIT_LFS_SKIP_SMUDGE=1

uv sync --frozen --group stage1 --python 3.11
conda create --prefix "$PIPETTE_ROOT/VLA-Precision/.runtime" \
  --file scripts/runtime-linux-64.lock --yes
source scripts/activate_training_env.sh

export HF_HOME="$PIPETTE_ROOT/.cache/huggingface"
export OPENPI_DATA_HOME="$PIPETTE_ROOT/models/openpi"
export JAX_COMPILATION_CACHE_DIR="$PIPETTE_ROOT/.cache/jax"
export PYTHONUNBUFFERED=1
mkdir -p "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/logs"
```

`.runtime` 只提供共享库，不要 `conda activate .runtime`；真正训练用 `.venv/bin/python`。已有正确的 `.runtime/conda-meta/history` 时不用重复创建。不要执行不带 `--group stage1` 或 `--group stage2` 的 `uv sync`，否则可能移除训练包。日常可直接使用已激活的 `python`，或者 `uv run --no-sync`。

若无系统编译器而使用项目现有 setup 脚本，可通过 `VLA_BUILD_TOOLCHAIN` 指定带 gcc/sysroot 的工具链目录，通过 `VLA_CONDA_EXE` 指定 conda 可执行文件；详情见 [环境记录](LOCAL_TRAINING_ENV.md)。

确认解释器和关键导入，兼容补丁应在直接导入 OpenPI training 模块前安装：

```bash
python - <<'PY'
import sys
import importlib.metadata as m
from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat
install_lerobot_import_compat()
import cv2, pyarrow, scipy, wandb
from vla_precision.integrations.openpi.configs import get_config
print(sys.executable)
for name in ("jax", "jaxlib", "jax-cuda12-plugin", "flax", "orbax-checkpoint", "numpy"):
    print(name, m.version(name))
print(get_config("pi05_full_finetune_pipette_fulltask").name)
PY
```

新 shell、tmux 或作业环境需要重新设置 `PIPETTE_ROOT` 并 `source scripts/activate_training_env.sh`，以及上面的 HF/OpenPI/JAX 缓存环境变量。不要复制其他机器的 `.venv`。

## 5 下载原生基础模型和 tokenizer

先独立下载，避免正式 GPU 作业在初始化时才等待网络。使用锁定 OpenPI 自己的下载工具，不需要另装 gsutil：

```bash
JAX_PLATFORMS=cpu python - <<'PY'
from openpi.shared import download
base = download.maybe_download(
    "gs://openpi-assets/checkpoints/pi05_base", gs={"token": "anon"})
tokenizer = download.maybe_download(
    "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
assert (base / "params").is_dir(), base
assert tokenizer.is_file(), tokenizer
print("params:", base / "params")
print("tokenizer:", tokenizer)
PY
```

预期路径：

```text
$PIPETTE_ROOT/models/openpi/openpi-assets/checkpoints/pi05_base/params/
$PIPETTE_ROOT/models/openpi/big_vision/paligemma_tokenizer.model
```

`params/` 是原生 Orbax 目录，不是 `.pt`、`.safetensors` 或单个文件。不要选择 `pi05-HIL293`、旧 XYZ-only 模型或 starVLA 的 PyTorch 转换权重。

离线计算节点可以预先传输以上两个下载结果，保留目录结构并在配置中指定新的 `params/` 路径。传完后必须做第 9 节的 `--model-shapes` 检查，目录存在不代表内容完整。OpenPI 下载缓存可能使用较宽松文件权限；共享服务器上将其放在你自己受限的父目录下，不要放令牌进去。

## 6 下载数据并生成所有图像缓存

如果 Hugging Face 要求身份认证，使用环境中的 `hf auth login` 交互登录；不要把 token 写在命令参数里。原始数据下载和准备必须使用固定 revision，脚本已内置，不必手动选择最新版本。

```bash
set -o pipefail
python scripts/prepare_pipette_fulltask.py \
  --source "$PIPETTE_ROOT/datasets/g1-pipette-2view-teleop0925-5task-eerel" \
  --output "$PIPETTE_ROOT/datasets/g1-pipette-fulltask-jax-v1" \
  --bridge-root "$PIPETTE_ROOT/VLAPolicyBridge" \
  --download --cache-images \
  2>&1 | tee "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/logs/prepare.log"
```

脚本会检查 237 episodes、60 Hz、关节顺序、录制级划分、批量 FK 与 Bridge FK，以及源数据右腕指令 XYZ；然后生成 59 维状态、32 维 command 缓存和两路 RGB `.npy`。正式训练要求所有图像缓存存在，而非每个 batch 随机 seek MP4。

成功时最后打印 p1 到 p5 的 episode/anchor 数，应该与第 1 节表格一致，且打印 `Images: cached`。完整数据应有 237 个 `numeric.npz` 和 474 个图像 `.npy`，随后预检会逐一检查形状和类型。

准备步骤通常不需要 GPU。`--numeric-only` 可用于先看数值链路，但不能代替正式训练的 `--cache-images`，两个参数互斥。图片缓存不足或磁盘将剩不到 10 GiB 时会拒绝继续。

下载或图像缓存中断后可重跑同一命令：已完成缓存会复用；图像的 `.partial.npy` 在成功后才改为最终文件名。若数值缓存/receipt 损坏、URDF/准备脚本/输入路径变化引起契约报错，不要绕过哈希校验，先调查，必要时用新的输出目录重新准备。不要两个进程同时写同一个准备目录。

跨机器迁移推荐传原始数据后在目标机重新准备。prepared manifest 含源数据绝对路径，直接复制整套旧 manifest 再改一两个 YAML 容易造成契约不一致。

## 7 GPU 分配和实际计算检查

以下假设你已获得同一节点的四张 GPU。在独占服务器可以选择 `0,1,2,3`；在 Slurm、容器或共享服务器，必须使用管理员或调度器授权的可见设备列表，不要扩大到别人的 GPU。

```bash
export PIPETTE_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CUDA_VISIBLE_DEVICES="$PIPETTE_GPUS" JAX_PLATFORMS=cuda python - <<'PY'
import jax
import jax.numpy as jnp
print("backend:", jax.default_backend())
print("devices:", jax.devices())
assert jax.device_count() == 4, "本例需要四张 GPU；否则先调整设备和配置"
for device in jax.devices():
    x = jax.device_put(jnp.ones((64, 64)), device)
    y = jax.jit(lambda a: a @ a)(x)
    y.block_until_ready()
    assert bool(jnp.isfinite(y).all())
    print("matrix multiply OK:", device)
PY
```

这只证明每张 GPU 可以进行 JAX 数值计算，不验证 FSDP collective、完整模型训练或显存峰值；第 10 节才验证这些链路。

重要：`main.py` 会用 YAML 的顶层 `cuda_visible_devices` 设置进程环境。因此只在 shell 改 `CUDA_VISIBLE_DEVICES`，却保留 YAML 的 `0,1,2,3`，可能选错卡。应在获得 GPU 分配后生成配置；若下次分配变化，先更新 YAML 的设备列表。Slurm 采用单节点单 task（如 `--nodes=1 --ntasks=1 --gres=gpu:4`，具体以集群规则为准），不要 `--ntasks=4`，也不要在登录节点开训练。当前入口没有配置多节点 JAX distributed 初始化。

## 8 生成正式训练配置

从准备好的 manifest 自动生成五份 YAML，保持 global batch128、10 epoch。路径全部在当前服务器解析为绝对路径。

```bash
python scripts/configure_pipette_fulltask.py \
  --data-root "$PIPETTE_ROOT/datasets/g1-pipette-fulltask-jax-v1" \
  --output-dir "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/configs" \
  --init-params "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params" \
  --checkpoint-root "$PIPETTE_ROOT/checkpoints/pipette-fulltask-jax" \
  --assets-root "$PIPETTE_ROOT/assets/pipette-fulltask-jax" \
  --cache-root "$PIPETTE_ROOT/.cache/pipette-fulltask-jax" \
  --gpus "$PIPETTE_GPUS" --fsdp-devices 4 \
  --batch-size 128 --epochs 10 \
  --project pipette-fulltask-pi05-jax --run-prefix pi05_fulltask_v1
```

生成文件为 `configs/pi05_fulltask_p1.yaml` 到 `pi05_fulltask_p5.yaml`。`openpi.name` 必须是 `pi05_full_finetune_pipette_fulltask`；`resume` 和 `overwrite` 默认均为 false。不要套用旧 tip 数据的配置。

若改变卡数或 batch，使用新 `--output-dir` 和新 `--run-prefix` 重生成，脚本会拒绝覆盖内容不同的已有配置。全局 batch 必须能被可见设备数整除，设备数必须能被 `fsdp_devices` 整除；这些算术条件满足仍不保证显存足够。

例如四卡显存不足时，可先尝试 global64/FSDP4 并使用新实验名。训练步数会自动按样本预算增加，但 batch、更新次数和每样本优化轨迹已经改变，不能称为完全相同的实验。不要只手改 batch 后继续使用原来的总步数。

## 9 统计和 CPU 预检

先计算每个阶段的训练集归一化统计，再检查输入。此步骤不会启动 GPU 训练。

```bash
for phase in 1 2 3 4 5; do
  bash scripts/run_pipette_fulltask_jax.sh norm \
    "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/configs/pi05_fulltask_p${phase}.yaml" || break
done
```

如果中途报错，先解决并补齐五阶段统计，再继续：

```bash
for phase in 1 2 3 4 5; do
  JAX_PLATFORMS=cpu python scripts/check_pipette_fulltask.py \
    --config "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/configs/pi05_fulltask_p${phase}.yaml" \
    --require-cached --images \
    --output "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/preflight_p${phase}.json" || break
done

JAX_PLATFORMS=cpu python scripts/check_pipette_fulltask.py \
  --config "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/configs/pi05_fulltask_p1.yaml" \
  --require-cached --model-shapes \
  --output "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/model_shapes_p1.json"

JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  -p no:cacheprovider tests/test_pipette_fulltask.py \
  tests/test_pipette_preview.py tests/test_correction_data.py
```

逐份检查报告，不能只看最后一个命令成功。应满足：

- `missing_image_caches: 0`，episode/anchor 数与第 1 节一致。
- state59、actions30×32、两张实际图像224×224，右腕占位图像 mask=false。
- `max_sampled_token_length < 320`，没有状态或标签 NaN/Inf。
- `model_check` 的 `missing` 和 `mismatched_shapes` 都为 0，`loss_shape` 为 `[1,30]`。
- 本地基点上的针对性测试结果为 30 passed。不要把它描述成整个仓库所有测试通过。

`check` 默认使用合成图像；上面显式加 `--images` 才抽样读取实际图像缓存。`--model-shapes` 只做抽象前向与基础权重元数据匹配，不分配完整模型和优化器，也不会发现训练时 OOM。

统计目录如下，每个阶段必须匹配自己的 stats：

```text
assets/pipette-fulltask-jax/pi05_full_finetune_pipette_fulltask/
  local/g1-pipette-fulltask-jax-v1-p1/norm_stats.json
  local/g1-pipette-fulltask-jax-v1-p1/norm_contract.json
  ... p2 到 p5 ...
```

## 10 短训练验收

先用 p1 生成独立 smoke 配置，保留正式训练的 global batch 和 FSDP 设置，只跑 4 次更新、8 个验证样本。这样能尽早发现完整训练路径问题；它不会证明正式 480 样本验证的总耗时或整个 run 的稳定性。

下面的 Python 仅创建新的 smoke YAML，不覆盖正式 YAML：

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml
root = Path(os.environ["PIPETTE_ROOT"])
base = root / "outputs/pi05-fulltask-jax"
cfg = yaml.safe_load((base / "configs/pi05_fulltask_p1.yaml").read_text())
cfg["openpi"].update(
    exp_name="pi05_fulltask_smoke_p1", num_train_steps=4,
    log_interval=1, save_interval=4, eval_interval=4,
    eval_samples=8, eval_batch_size=4, resume=False, overwrite=False)
cfg["openpi"]["lr_schedule"].update(warmup_steps=1, decay_steps=4)
cfg["paths"]["checkpoint_root"] = str(root / "checkpoints/pipette-fulltask-jax-smoke")
path = base / "smoke_p1.yaml"
with path.open("x") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(path)
PY
```

W&B 登录可通过 `wandb login` 完成。需要团队空间时先设置 `WANDB_ENTITY` 为你有权限的团队；不要把 API key 提交到 Git。设置本地 W&B 目录后启动：

```bash
export WANDB_DIR="$PIPETTE_ROOT/outputs/pi05-fulltask-jax/wandb"
mkdir -p "$WANDB_DIR"
wandb login
set -o pipefail
WANDB_MODE=online bash scripts/run_pipette_fulltask_jax.sh train \
  "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/smoke_p1.yaml" \
  2>&1 | tee "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/logs/smoke_p1.log"
```

无外网时把 `WANDB_MODE=online` 改为 `offline`；不要因此停用全部本地日志。之后可在联网节点用 `wandb sync` 同步具体的 `offline-run-*` 目录。离线恢复与在线 run 的关联不应想当然，见第 13 节。

验收标准：4 个 optimizer updates 完成；loss 和 grad_norm 有限；出现最终 holdout 指标；`4/params`、`4/train_state`、`4/assets` 和 `best/<step>` 实际写完；进程正常退出。第一次 JIT 编译和 checkpoint 保存可能显著慢于 update，不要只凭暂时没有新 loss 就中断。

smoke 目录为：

```text
checkpoints/pipette-fulltask-jax-smoke/
  pi05_full_finetune_pipette_fulltask/pi05_fulltask_smoke_p1/
```

如需重复 smoke，使用新的 smoke 配置文件名和 `exp_name`，保留失败日志；不要用 `overwrite: true` 清空旧目录。正式训练不加载 smoke 权重，仍从官方基础模型开始。需要验证断点恢复时，可对 smoke 做专门的后续恢复测试，但修改总步数和 schedule 应记录为测试，不应用到正式实验。

## 11 正式训练和断开 SSH

在独占或已获许可的计算节点上，可用 tmux 保持进程。Slurm 下 tmux 不能替代作业分配；作业到期仍会被调度器结束，应把同样命令放入单 task 作业，按集群规范提交。

```bash
tmux new -s pi05-fulltask
```

在 tmux 内重新设置实际根目录并激活：

```bash
export PIPETTE_ROOT=/data/YOUR_USER/pipette
cd "$PIPETTE_ROOT/VLA-Precision"
source scripts/activate_training_env.sh
export HF_HOME="$PIPETTE_ROOT/.cache/huggingface"
export OPENPI_DATA_HOME="$PIPETTE_ROOT/models/openpi"
export JAX_COMPILATION_CACHE_DIR="$PIPETTE_ROOT/.cache/jax"
export WANDB_DIR="$PIPETTE_ROOT/outputs/pi05-fulltask-jax/wandb"
export PYTHONUNBUFFERED=1
```

确认 YAML 的 GPU 列表与当前分配一致，然后顺序跑五阶段。不要在同四张 GPU 上同时启动五个 run。

```bash
(
  set -euo pipefail
  for phase in 1 2 3 4 5; do
    WANDB_MODE=online bash scripts/run_pipette_fulltask_jax.sh train \
      "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/configs/pi05_fulltask_p${phase}.yaml" \
      2>&1 | tee "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/logs/train_p${phase}.log"
  done
)
```

任一阶段报错就停止循环，避免带着错误继续跑后面的阶段。括号在子 shell 中启用 `set -e`，失败后仍保留 tmux shell。按 `Ctrl-b` 然后 `d` 断开 tmux；重连用 `tmux attach -t pi05-fulltask`。不要用 `Ctrl-c` 作为 detach。

也可以只跑一阶段：

```bash
bash scripts/run_pipette_fulltask_jax.sh train \
  "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/configs/pi05_fulltask_p3.yaml"
```

每次 `train` 都先做要求图像缓存齐全的 CPU 预检，再用 CUDA 启动训练。不要加 `torchrun`、`accelerate launch` 或为每张卡开一个 Python；这套入口由一个 JAX 进程使用全部配置 GPU。

运行中不要改源数据、归一化、YAML、Python 环境或分支。JAX 编译缓存不要和不可信用户共用可写目录。尚无此配方的可靠每 epoch 耗时；应记录目标节点实际 update、评估和保存耗时后再估算剩余时间。

## 12 进度和训练结果

W&B 项目是 `pipette-fulltask-pi05-jax`，正式 run 名分别是 `pi05_fulltask_v1_p1` 到 `p5`。应查看：

| 指标 | 含义 |
| --- | --- |
| `loss` | 训练 flow matching loss，不是末端位置 RMSE |
| `grad_norm` | 梯度范数，用于发现异常或数值发散 |
| `camera_views` | 首 batch 图像核查，含右腕黑色占位图 |
| `eval/mse` | 固定验证样本上的归一化动作预测 MSE，选 best 用这个 |
| `eval/hold_mse` | 保持 chunk 第 0 行动作不变的基线 |
| `eval/mse_ratio` | 模型 MSE / 基线 MSE，小于 1 表示此指标优于保持不动 |
| `eval/left_xyz_rmse_mm` 和 `eval/right_xyz_rmse_mm` | XYZ 坐标分量 RMSE，mm，不是三维距离的直接均值 |
| `eval/left_rotation_rms_deg` 和 `eval/right_rotation_rms_deg` | SO(3) 旋转差角度 RMS，度 |
| `eval/left_thumb_rmse_registers` 和 `eval/right_hand_rmse_registers` | 对应动作语义的寄存器 RMSE |
| `eval/arm_joints_rmse_deg` | 14 维关节角 RMSE，度 |

物理指标还有 `_hold` 后缀的保持不动基线。验证使用固定 flow noise、10 步求解、完整状态，不启用训练 state dropout。训练 loss 与 eval/mse 定义不同，不能要求两条曲线数值相等。

```bash
nvidia-smi
tail -n 40 "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/logs/train_p1.log"
tail -n 2 "$PIPETTE_ROOT/checkpoints/pipette-fulltask-jax/pi05_full_finetune_pipette_fulltask/pi05_fulltask_v1_p1/holdout_metrics.jsonl"
df -h "$PIPETTE_ROOT"
```

默认约每 epoch 验证，不是每次训练 log 都有验证点。正式 run 的验证间隔 p1 到 p5 分别为 286、136、220、135、302 个 updates，最后一步额外保证验证。

验证 episodes 与 GR00T 的划分一致，但不是同一组 480 个 frame anchors；不要将两套 normalized MSE 直接比较。严谨比较要用同一组输入样本、相同动作语义和物理单位。这里的验证误差也不等于真机任务成功率；反复选模型的 holdout 不是独立最终测试集。

## 13 Checkpoint 和断点恢复

每阶段目录如下：

```text
checkpoints/pipette-fulltask-jax/
  pi05_full_finetune_pipette_fulltask/pi05_fulltask_v1_p1/
    resolved_config.yaml
    wandb_id.txt
    holdout_samples.json
    holdout_metrics.jsonl
    best_selection.json
    <completed_step>/
      params/
      train_state/
      assets/
    best/<best_step>/
      params/
      assets/
```

主目录的数字是已经完成的 updates。主目录默认只保留最近一次完整训练检查点，含 optimizer 和 step；`best/` 独立保留验证 MSE 最低的推理权重和对应统计，不含 optimizer。它不是 `best.pt`。

正常结束必须等日志中的 `waiting for checkpoint manager to finish` 后进程退出。突然退出只会保留上次完成的检查点，并不保证自动保存 Ctrl-c 时的最新状态。默认约每 epoch 保存一次，早期中断可能连第一个完整检查点都没有。调度时为验证和保存预留时间。

恢复前确认原训练进程已停止。保留原数据、统计、实验名、模型定义、总训练预算与 checkpoint root，仅把该阶段原 YAML 的 `openpi.resume` 改为 true；`overwrite` 保持 false。可在编辑器中修改：

```yaml
openpi:
  resume: true
  overwrite: false
```

这是要修改的字段，不是完整 YAML，不能覆盖整个配置。原 `num_train_steps` 是累计目标，恢复时不要改成“剩余步数”。保持 `wandb_id.txt`，并继续使用相同 W&B project/entity，在线恢复会要求原 run 存在。

然后重新运行同一阶段：

```bash
set -o pipefail
bash scripts/run_pipette_fulltask_jax.sh train \
  "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/configs/pi05_fulltask_p1.yaml" \
  2>&1 | tee -a "$PIPETTE_ROOT/outputs/pi05-fulltask-jax/logs/train_p1.log"
```

确认日志恢复到预期 update，而非从 0 开始。若第一次保存前就失败，底层 OpenPI 可能识别为没有可恢复状态并重新初始化；这不是成功恢复。不要把 `initialization_checkpoint` 指向 `best/` 当成恢复：那会丢失优化器和进度。不要设置 `overwrite: true`，该选项会删除整个现有 run 目录。

恢复不是逐位重放：DataLoader 位置和 NumPy augmentation RNG 没有完整存入 checkpoint。数据/统计/holdout 哈希变化会被校验拒绝；确需换数据应创建新实验，而不是修改旧实验的契约文件。

W&B 故障不必意味着模型丢失。离线 run 尚未上传时不要直接改在线并依赖 `resume="must"`，先同步原 run 或持续离线；必要时在 YAML 设置 `wandb_enabled: false` 继续模型恢复，仍会记录本地 holdout JSONL，但不再提供新的在线训练曲线。不要手工删除 `wandb_id.txt` 来冒充新 run。

五阶段循环中途失败时，只恢复失败阶段，并从后续尚未开始的阶段继续，不要重新从 p1 跑且使用 overwrite。已完成阶段不需要再训练。

## 14 备份和传回模型

先确认训练和异步保存已经结束。查看 best 的真实位置：

```bash
export PIPETTE_RUN_DIR="$PIPETTE_ROOT/checkpoints/pipette-fulltask-jax/pi05_full_finetune_pipette_fulltask/pi05_fulltask_v1_p1"
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["PIPETTE_RUN_DIR"])
choice = json.loads((root / "best_selection.json").read_text())
print("best step:", choice["step"], "mse:", choice["mse"])
print("inference directory:", root / choice["checkpoint"])
PY
```

最稳妥是保留完整 run 文件夹、对应生成 YAML、prepared manifests/contract、`norm_stats.json`/`norm_contract.json`、数据 revision、VLA-Precision 和 Bridge commit，以及日志。如果只导出推理模型，至少带 `best/<step>/params`、其 `assets`、`best_selection.json`、`resolved_config.yaml`、`holdout_samples.json` 与动作定义文档。

在自己的电脑执行下列命令，从远程复制完整 p1 run；先替换用户名、服务器和实际路径。此命令不删除源文件，也不使用 `--delete`：

```bash
mkdir -p ./pi05-fulltask-models
rsync -avh --partial \
  YOUR_USER@YOUR_SERVER:/data/YOUR_USER/pipette/checkpoints/pipette-fulltask-jax/pi05_full_finetune_pipette_fulltask/pi05_fulltask_v1_p1/ \
  ./pi05-fulltask-models/pi05_fulltask_v1_p1/
```

其余阶段同理。Orbax checkpoint 是目录树，不能只复制一个元数据文件。归档后先验证可以读取，再考虑按实验保留策略清理旧完整状态；删除 `train_state` 后就不能精确恢复 AdamW。本文不提供自动删除命令。

不要把新权重直接放进旧的 XYZ-only GUI lane。新的双腕 32 维动作需要匹配状态构造、相机输入、相对位姿还原、左拇指阶段语义、IK 和安全控制。当前文档到离线训练与模型导出为止，不声称真机部署已经完成。

## 15 常见问题

| 症状 | 检查与处理 |
| --- | --- |
| `No module named vla_precision` | 检查是否在仓库目录，是否激活 `.venv`，是否执行 `uv sync --frozen --group stage1` |
| Bridge 导入或 URDF 不存在 | 核对固定 commit、`--bridge-root` 和 vendored URDF；不要指向 TWIST2 根目录 |
| evdev 编译失败 | 补齐 C 编译工具与 Linux 输入头文件；不要靠安装整个机器人软件栈解决 |
| `libGL`、FFmpeg 或 TorchCodec 动态库报错 | 检查 `.runtime` 安装和 activation；不要混用其他项目的 conda Python |
| JAX 只显示 CPU 或 CUDA 初始化失败 | 在实际 GPU 节点检查驱动、设备授权、JAX plugin 和库冲突；`JAX_PLATFORMS=cuda` 应失败而不是静默退 CPU |
| GPU 数量或 FSDP 整除错误 | 核对 YAML 的 `cuda_visible_devices`、batch 和 `fsdp_devices`，不要只改 shell 变量 |
| 初始化就 OOM | 全参数权重和 AdamW 状态可能已经超出单卡容量；增加合适的 FSDP 设备和显存，不能只依赖减 batch |
| 训练或验证 OOM | 新实验降低 global batch；验证单独降低 `eval_batch_size` 且必须整除 480；重新短测。不要把这些改变误称为原配方 |
| 模型或 tokenizer 下载卡住 | 检查 Google Cloud Storage 网络；在联网节点下载完整缓存后传输，不要用随机初始化替代 |
| Hugging Face 401 或 403 | 检查当前账号读权限、网络和登录；不要把个人 token 放到日志 |
| `Preparation contract changed` | 路径、URDF、脚本或 metadata 发生变化；调查后用新目录重新准备，不删除契约绕过检查 |
| `Normalization contract mismatch` | 检查阶段和 manifest；在正式训练开始前为对应数据重新算统计，运行中的实验不要换 stats |
| missing image caches | 完整跑完 `--download --cache-images`；只有 metadata/parquet 不够 |
| 视频 truncated 或帧数不一致 | 检查 pinned 数据下载完整性；不要裁掉错误帧以通过检查 |
| run directory already exists | 确认是恢复还是新实验；恢复用 resume，新实验用新名字，勿 overwrite |
| W&B 无曲线 | 检查 project/entity、online/offline、账号权限；查看本地日志，评估约每 epoch 才产生 |
| 没有 `best.pt` | 正常，查看 `best_selection.json` 指向的原生 Orbax `best/<step>/` |
| 长时间没有新 loss | 区分首次 JIT、480 样本评估、磁盘保存和真正挂起；检查 GPU、CPU、磁盘及日志，不盲目 kill |

## 16 交接验收清单

- 记录服务器 GPU/驱动、节点、CPU/RAM、可用磁盘、两个仓库 commit 和依赖版本。
- 五阶段 metadata/数值准备成功，474 个图像缓存通过预检；原始 revision 和 URDF 契约保留。
- 五份正式 YAML、五份训练 stats 和五份真实图像预检报告齐全，模型形状检查通过。
- 目标 GPU 上短训练完成更新、评估、last 和 best 保存，不仅是 CPU 抽象前向通过。
- 正式五个 run 正常完成，最终 updates 与各自生成配置一致；未把 smoke 权重作为正式初始化。
- 五个 best 的路径、step、验证指标与保持不动基线可读取，归一化和配置随模型一同备份。
- 明确剩余工作：共享验证集上的 GR00T 对照、独立测试，以及双腕真机推理和安全验证。

## 实现入口

- [数据准备与 FK](../scripts/prepare_pipette_fulltask.py)
- [标签与划分](../src/vla_precision/data/pipette_fulltask.py)
- [五阶段配置生成](../scripts/configure_pipette_fulltask.py)
- [CPU 预检](../scripts/check_pipette_fulltask.py)
- [启动入口](../scripts/run_pipette_fulltask_jax.sh)
- [JAX 训练与恢复](../src/vla_precision/integrations/openpi/train.py)
- [固定 holdout 和 best 选择](../src/vla_precision/integrations/openpi/fulltask_eval.py)
