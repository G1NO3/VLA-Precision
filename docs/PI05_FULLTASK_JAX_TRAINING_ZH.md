# π0.5 五阶段 JAX 训练说明

这套流程沿用本项目原有的 Physical Intelligence OpenPI JAX/Flax 实现，从官方 `pi05_base/params` 初始化，并在五个阶段上分别全参数微调。它不使用 starVLA 的 PyTorch π0.5，也不从旧的 tip attachment checkpoint 接着训练。新增代码位于 VLA-Precision；不修改机器人控制、安全层或现有 GUI lane。

目前已完成数值数据准备、五份配置生成、训练集归一化和 CPU 输入检查；尚未启动完整训练。启动前必须在训练机补齐视频缓存，并验证 GPU 和磁盘容量。

## 数据与任务

来源为 [237 条五阶段示教](https://huggingface.co/datasets/jren313/g1-pipette-2view-teleop0925-5task-eerel)，固定 revision `340ebabb48e6d1fb42f92178fba0f6ded87322df`。237 条是切分后的阶段 episodes，不是 237 次完整五阶段任务。该数据包含拿移液枪、拿试管、瞄准管口、放回试管和放回移液枪；不要把它的第三阶段等同于旧数据中的绿色 tip attachment。

每个阶段都独立从基础 π0.5 初始化，不是 p1 权重接着训练 p2，也不是一个混合五任务模型。完整任务的阶段切换与真机部署不属于本次训练入口。

| 阶段 | 任务 | 训练 episodes | 验证 episodes | 过滤后训练 anchors | 10 epoch 等效步数 |
| --- | --- | ---: | ---: | ---: | ---: |
| p1 | 右手拿移液枪 | 54 | 9 | 36,558 | 2,857 |
| p2 | 左手拿试管 | 26 | 5 | 17,290 | 1,351 |
| p3 | 移液枪瞄准试管口 | 42 | 7 | 28,149 | 2,200 |
| p4 | 左手放回试管 | 26 | 5 | 17,176 | 1,342 |
| p5 | 右手放回移液枪 | 54 | 9 | 38,568 | 3,014 |

训练步数按 `ceil(训练 anchors × epochs / global_batch)` 计算，表中 global batch 为 128。因此部分阶段比原 GR00T 配方多一步，是取整差异。五阶段的训练 anchors 数已在真实 parquet 上复现，与 GR00T 配方一致。

验证 episode ID 固定为发布模型卡的划分。准备脚本按 `source_session + 原 source_episode` 检查，同一段录制切出来的 pick 和 return 不得跨训练集和验证集。语言使用数据集每个阶段的完整 goal sentence，而不是旧的插 tip 指令。

## 状态和动作定义

状态 59 维，严格按下列顺序拼接：

| 区间 | 含义 |
| --- | --- |
| 0–28 | 实测 29 关节角 |
| 29–34 | 左腕实测 XYZ 和 rotation vector |
| 35–40 | 右腕实测 XYZ 和 rotation vector |
| 41–46 | 左腕指令 XYZ 和 rotation vector |
| 47–52 | 右腕指令 XYZ 和 rotation vector |
| 53 | 左手 thumb bend 指令 |
| 54–58 | 右手 pinky、ring、middle、index、thumb bend 指令 |

腕部 FK 使用 VLAPolicyBridge 中 vendored TWIST2 URDF 的 `left_rubber_hand` / `right_rubber_hand` link，参考系为 pelvis；不是移液枪 nozzle TCP。脚本对批量 FK 与 bridge FK 做数值核对，并核对源数据现成的右腕指令 XYZ。

动作 32 维，每个 action chunk 为 30 个 60 Hz 样本，对应 0.5 秒的名义执行块；第一个到最后一个样本的时间间隔为 29/60 秒。

| 区间 | 监督目标 |
| --- | --- |
| 0–5 | 左腕相对指令位姿 |
| 6–11 | 右腕相对指令位姿 |
| 12 | 左拇指：p1 为绝对寄存器，p2–p5 为 chunk-relative 寄存器差 |
| 13–17 | 右手五通道绝对寄存器 |
| 18–24 | 左臂七关节绝对指令角 |
| 25–31 | 右臂七关节绝对指令角 |

对每个腕部，标签是：

```text
translation[k] = p_cmd[t+k] - p_cmd[t]
rotation[k]    = Log(R_cmd[t].T @ R_cmd[t+k])
```

平移的轴是 pelvis 轴；旋转差表达在 anchor 的局部坐标系。不能直接相减 rotation vector，也不能逐步累加 chunk-relative 平移。标签来自 commanded joints 的 FK，不是 measured joints 的运动。第 0 行腕部相对动作严格为零；绝对手部和关节通道第 0 行并非零。

尾部不足 30 行时重复最后一个 command，然后做相对变换，不跨 episode。14 个关节通道在将来的部署中仍应只作为 IK 初值，不能绕过 IK 和安全层直接执行。

## 过滤与归一化

训练过滤复现 starVLA 的实际代码：拼接左右腕 XYZ，用 `t+29` 与 `t` 的六维欧氏距离判断是否达到 1 mm。它不是逐只手判断，也不是整个 chunk 的最大运动幅度。验证集不应用此过滤。

这个规则可能删除“手腕静止但正在抓握”的样本。当前预检中，首尾手部指令变化超过 1 寄存器但被过滤的训练 anchors 为 p1=9、p2=14、p3=0、p4=8、p5=6。为保持第一轮的数据口径，目前没有修改过滤规则；后续可把保留纯抓握动作作为单独对照实验。

每个阶段单独计算训练集 state/action 的 OpenPI quantile statistics，并供该阶段训练、验证和推理共同使用。使用固定的实际 chunk 标签，禁止对相对动作再次做 delta transform。`norm_contract.json` 记录阶段 manifest、归一化文件哈希和样本数，预检会拒绝过期或错阶段的统计。

统计实现保留 OpenPI 的原生 RunningStats 和 Normalize，而不是复制 starVLA 的归一化文件。因此虽然动作的物理定义一致，归一化数值不承诺逐位一致。跨模型应比较物理误差，不能直接比较各自 normalized MSE。

## JAX 配方

| 设置 | 新配置 |
| --- | --- |
| 实现 | 官方 OpenPI 的 JAX/Flax π0.5，通过 VLA-Precision 接入 |
| 基础权重 | 官方 `pi05_base/params`，不是 `.pt` 或 starVLA 转换权重 |
| 微调 | 全参数；无 LoRA；无冻结视觉骨干 |
| 图像 | 头部 RGB 和左腕；resize270，训练随机 crop256，验证中心 crop256，再 resize224 |
| 缺失相机 | 右腕填零并设 image mask=false，不复制左腕图像 |
| 图像增强 | 上述裁剪之外，保留原生 OpenPI 训练增强；不声称与 GR00T 的 rotation/color jitter 完全相同 |
| 状态输入 | 完整 59 维 quantile-normalized state 编码为文本 |
| 文本长度 | 320 tokens；真实数值抽样检查最大 243–246 tokens |
| 状态 dropout | 仅训练时，以 0.8 概率将整段归一化状态置零，再 tokenize；验证和推理不丢弃 |
| 输出 | 30 × 32；保留预训练动作投影层形状 |
| 优化器 | 原生 JAX Optax AdamW，非 bitsandbytes 8-bit AdamW |
| 学习率 | 统一 1e-5，warmup100，cosine 到 1e-6 |
| 其他优化参数 | betas 0.9/0.95，eps 1e-8，weight decay 1e-10，梯度裁剪 1.0，无 EMA |
| batch | 默认 global128，单 JAX 进程使用四 GPU，FSDP4 |
| 训练时长 | 10 epoch 等效样本预算，不早停 |
| 推理求解 | 原生 π0.5 flow matching，10 步 |
| 监控 | W&B 项目 `pipette-fulltask-pi05-jax`，五阶段分别为独立 run |

统一 1e-5 是延续本项目 JAX π0.5 全参数微调的保守起点，不是 GR00T 的骨干 2e-5 / 动作头 2e-4，也不是已经实验证明最优的学习率。推理默认 10 步而不是 GR00T 的 4 步。

59 维状态不需要把 action_dim 改成 59。OpenPI π0.5 在 tokenize 时读取完整状态，之后的 PadStatesAndActions 只补齐不足 32 维的输入，不截断更长的状态。其连续状态投影只用于 π0，π0.5 的状态条件来自文本。原生基础权重的全部 51 个参数叶子已通过形状核对。

## 操作次序

在包含已配置 JAX 环境的 VLA-Precision 目录执行。新机器需先按仓库说明安装环境；数据准备同时需要相邻的 VLAPolicyBridge 仓库，或显式提供 `--bridge-root`。

```bash
cd /path/to/pipette/VLA-Precision
source scripts/activate_training_env.sh
```

先下载和准备数据。纯数值预检可用 `--numeric-only` 替代 `--cache-images`；正式训练必须补齐图像缓存。

```bash
python scripts/prepare_pipette_fulltask.py \
  --source ../datasets/g1-pipette-2view-teleop0925-5task-eerel \
  --output ../datasets/g1-pipette-fulltask-jax-v1 \
  --download --cache-images
```

原始视频约 10.6 GB；全部 270×270 RGB mmap 图像约 66 GiB，五个阶段共用一份缓存，不重复五份。准备程序不覆盖已改变的数据契约或数值缓存。迁移到另一台机器时，重新准备或重新生成带新路径的配置；不要直接复制含旧绝对路径的 manifest 后继续训练。

生成五份训练配置。下面默认四卡、global batch128；更换卡数或 batch 后请重新生成到新配置目录，步数会随之调整。

```bash
python scripts/configure_pipette_fulltask.py \
  --data-root ../datasets/g1-pipette-fulltask-jax-v1 \
  --output-dir ../outputs/pi05-fulltask-jax/configs \
  --init-params ../models/openpi/openpi-assets/checkpoints/pi05_base/params \
  --gpus 0,1,2,3 --fsdp-devices 4 --batch-size 128 --epochs 10
```

可通过 `--checkpoint-root`、`--assets-root`、`--cache-root` 指向训练机的大容量磁盘。这里只接受原生 JAX `params/` 文件夹。五份 YAML 都是完整配置，不依赖动态继承，也不会覆盖不同内容的现有配置。

先计算所有阶段的统计，再预检：

```bash
for phase in 1 2 3 4 5; do
  bash scripts/run_pipette_fulltask_jax.sh norm \
    ../outputs/pi05-fulltask-jax/configs/pi05_fulltask_p${phase}.yaml
  bash scripts/run_pipette_fulltask_jax.sh check \
    ../outputs/pi05-fulltask-jax/configs/pi05_fulltask_p${phase}.yaml
done
```

`check` 默认用真实数值加合成图像检查输入，不会误报视频已准备好。额外检查原生权重和抽象前向：

```bash
JAX_PLATFORMS=cpu python scripts/check_pipette_fulltask.py \
  --config ../outputs/pi05-fulltask-jax/configs/pi05_fulltask_p1.yaml \
  --model-shapes --require-cached
```

加 `--images` 可验证实际解码图像；未建立缓存时会较慢。检查通过后，在 GPU 训练机启动一个阶段：

```bash
WANDB_MODE=online bash scripts/run_pipette_fulltask_jax.sh train \
  ../outputs/pi05-fulltask-jax/configs/pi05_fulltask_p1.yaml
```

这是一个 JAX 进程，不要用 `accelerate launch`、`torchrun` 或启动四个独立进程。`train` 会先做包括图像缓存完整性的 CPU 预检，然后强制使用 CUDA，不允许悄悄退回 CPU 训练。GPU 可见列表和 FSDP 数由 YAML 决定。

本机只有一张约 48 GB 显卡，生成的默认四卡配置不是本机可直接启动的配置；不能因为数据预检通过就认定 full fine-tune 的显存足够。还未执行四卡 GPU smoke test，batch128 的实际显存需在目标训练机验证。建议先用独立 smoke 配置及新 run 名完成少量 update，再跑完整预算。

## 验证与模型选择

约每个 epoch 和最后一步执行一次验证。每阶段固定 480 个验证 chunks，按 episode 均衡抽样并记录 `holdout_samples.json`；每次评估固定 flow noise，不消耗训练 RNG，不启用 state dropout 或图像随机增强。

注意：使用的是与 GR00T 相同的验证 episodes，但这 480 个 frame anchors 不是 starVLA 的原始随机 480 样本。不能把新指标直接接在旧 GR00T loss curve 上。严格对照应让两个模型在同一份 `holdout_samples.json` 上重新评估。

W&B 和本地 `holdout_metrics.jsonl` 记录 normalized MSE、保持第 0 行不动的基线、左右腕 XYZ 误差、旋转误差、左右手寄存器误差及关节误差。XYZ 指标是坐标分量 RMSE，以 mm 表示；旋转指标用 SO(3) 测地角 RMS，以度表示。两种手部语义都在各自正确的物理单位中比较。

最低验证 MSE 对应的权重保存为原生 Orbax checkpoint，不会生成 `best.pt`。这只是 open-loop prediction quality，不是真机任务成功率。用于反复选模型的 holdout 应称为验证集；若写论文，还需要独立的最终测试。

## 保存和恢复

默认目录结构如下：

```text
checkpoints/pipette-fulltask-jax/
  pi05_full_finetune_pipette_fulltask/
    pi05_fulltask_v1_p1/
      resolved_config.yaml
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

主目录保留最近完整检查点，包含参数、AdamW 状态和训练步数，供断点恢复。`best/` 单独保留最优推理权重与该阶段的训练统计，不含优化器。`best_selection.json` 给出实际路径。新流程的数字目录名是已完成 update 数。

恢复时在原 YAML 设置 `openpi.resume: true`，保持 run 名、数据与状态/动作定义不变，仍使用原 checkpoint_root。不要设置 overwrite，也不要从 `best/` 恢复优化器。数据读取器和 NumPy augmentation RNG 的进度未完整 checkpoint，因此恢复不是逐位重放中断前的样本顺序。

JAX AdamW 的磁盘占用显著大于 PyTorch 8-bit 优化器。一个全参数训练状态可能约 50 GB，最佳推理参数另约 16 GB，异步保存时还需要新旧快照的临时空间。五阶段一起保留 last+best，加图像和原始数据，应准备数百 GB，建议约 500 GB 以上可用容量。这里只是容量估计，以实际 checkpoint 大小为准；本机约 175 GiB 空闲不适合不加规划地保留五个完整 run。

## 本地完成范围

目前下载了完整 metadata 和 237 个 parquet，生成共享数值缓存、五阶段 manifest、五份 YAML 和五份训练集统计。30 项定向测试通过，包括动作旋转组合、尾部处理、录制级划分、state dropout、最优模型评估和原生 Orbax 保存恢复。完整模型的抽象前向通过，59 维状态得到 `[1,30]` 的 loss shape，且基础参数形状全部兼容；这不是 GPU 数值前向或训练验证。

另下载了每阶段第一条训练 episode 的两路视频，验证真实图像解码后的输入链路。没有下载全部视频，没有生成全部图像 mmap，没有启动完整训练或机器人。部署仍需后续把新的 JAX 推理输入输出接入双腕 IK lane；不要选择旧的 XYZ-only π0.5 GUI 配置来执行这套 32 维模型。
