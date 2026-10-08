# GR00T 五阶段 RLToken 配置与接管边界

当前默认配置使用 GUI 已部署的 **单个五任务 GR00T v2 checkpoint**，沿用 `jm_pipette` 当前 QwenOFT actor-critic 的奖励。五阶段共用冻结 GR00T，但初期各自训练 token/actor/critic，用户已选择先做 P1（拿移液枪）。`gr00t_rlt.py` 是纯离线准备/训练入口；新增 `gr00t_rlt_live.py` 提供模型服务、SQLite 导出和有限轮次训练，GUI 的独立 RLToken lane 使用顺序 collector。实现已通过离线测试，**尚未进行真机闭环验收**。

默认配置位于 [gr00t_fulltask_v2_rlt.yaml](../configs/pipette_rl/gr00t_fulltask_v2_rlt.yaml)，入口是 [gr00t_rlt.py](../scripts/gr00t_rlt.py)。旧五 checkpoint 配方保留在 [gr00t_fulltask_rlt.yaml](../configs/pipette_rl/gr00t_fulltask_rlt.yaml)，必须显式用 `--config configs/pipette_rl/gr00t_fulltask_rlt.yaml` 选择；不会自动下载旧模型。代码保存在 `VLA-Precision` 的 `jy-full-gr00t` 分支工作区；没有修改另一个 `jm_pipette` 实验，也没有切换 starVLA、Bridge 或 RLinf 分支。

P1 token 和 BC 权重已训练并通过实际 GR00T GPU 服务推理检查；P2–P5 尚未训练各自初始权重。PICO 相对双腕接管为独立可选输入，硬件标定仍需操作员验收。配置保留 `live_enabled: false`，表示离线工具不隐式启动机器人；改为 true 仍报错。在线功能通过新服务和 Bridge 的显式 `--rlt_run` 参数启用，不以该标志代替安全验收。

## 模型与下载位置

2026-10-08 发布说明：四个相关仓库统一发布到各自的 `jy-full-gr00t` 分支。starVLA 的 `2c52f5f10fe232bd875bd0bdad8454775154bd58` 只将此前训练/推理已经使用的五任务 dataset registry 补丁提交；模型计算代码未变。版本检查精确兼容这个提交及配方原始基线 `217eeb95bd359e31254e1c09ddfb04216711ec26`，不放行任意后续提交。保留原配方 contract hash，因此现有 P1 token/BC/cache 无需重写或重训。模型权重、数据集及 replay 不通过 Git 推送。

当前默认模型：

- 仓库：`jren313/starvla-pipette-5task-v2`，P1–P5 共用。
- revision：`72be4b8c7ea7454ae4d665c4b2c7eba0aa4c51e2`。
- checkpoint：`models/starvla-pipette-5task-v2/gr00t/checkpoints/steps_9000_pytorch_model.pt`。
- 权重 SHA256：`a4934bf17217ea7cd7786be4632401ad905fdd0c2779a698ee5fcf8d054e90c5`。
- 五阶段共用该 checkpoint 自己的归一化统计，左拇指在**所有阶段**都使用 chunk-relative 标签。

`download --phase p1` 即可登记/核验共享模型，已有正确权重会复用。`check --verify-weights` 可重新核验完整权重；普通 `check` 核验元数据哈希、权重大小和下载收据。

以下仅为**旧配方历史对照**，不是当前默认模型：

| 阶段 | Hugging Face 仓库 | 选定 checkpoint |
| --- | --- | --- |
| p1 拿移液枪 | `jren313/starvla-pipette-pick-r3` | `steps_2000_pytorch_model.pt` |
| p2 拿试管 | `jren313/starvla-pipette-tube-pick-r2` | `steps_1000_pytorch_model.pt` |
| p3 对准管口 | `jren313/starvla-pipette-aim-r1` | `steps_1350_pytorch_model.pt` |
| p4 放回试管 | `jren313/starvla-pipette-tube-return-r1` | `steps_500_pytorch_model.pt` |
| p5 放回移液枪 | `jren313/starvla-pipette-return-r1` | `steps_3000_pytorch_model.pt` |

旧配方每阶段固定 Hugging Face revision，具体 SHA 在旧 YAML 中。旧五权重合计约 44.38 GiB；新配方只需一份约 8.88 GiB 的权重。旧路径示例：

```text
models/starvla-pipette-pick-r3/gr00t/
  config.yaml
  config.full.yaml
  dataset_statistics.json
  checkpoints/steps_2000_pytorch_model.pt
  download_receipt.json
```

`download` 命令将完整权重 SHA256 与 Hugging Face LFS SHA256 核对，记录文件大小、哈希和 revision；不会反序列化模型来验证下载。

这些模型基于 Cosmos-Reason2-2B 的第 16 层特征和 32 层 DiT，参考动作仍走原来的 4 步 flow sampler。加载依赖原 starVLA 环境及 Cosmos backbone/processor 缓存。上游构造器可能先打印“随机初始化 head”，随后 `from_pretrained` 会严格加载整个选定 `.pt`；必须以完整严格加载成功为准，不能忽略缺失参数错误。

## 奖励使用实际实验配置

奖励来源是 `/home/jwang3617/jm_pipette/runs/qwenoft_ac/config.json`，以及该工作区 `VLAPolicyBridge/vla_policy_bridge/hil/grpo_session.py`、`ac_model.py`、`ac_rewards.py`。检查时 Bridge commit 是 `ac8e7cd9b8003d574d5ea95b675c5733719fb9d3`。它与旧说明中的失败 −10 已不同：

| 字段 | 本次配置 |
| --- | ---: |
| 奖励时钟 | 单调时间，60 Hz 计时基准 |
| 每步时间惩罚 | −0.005 |
| 成功终止 | +10 |
| 失败终止 | 0 |
| 单步折扣 | √0.999，约 0.999499875 |
| 终止 bootstrap | 0 |

按实际持续时间计算 `ticks = max(1, round(elapsed_seconds × 60))`。若控制窗口已结束但操作员稍后才标记结果，终止窗口的结束时间扩展到结果时间，等待仍计费。终止奖励替换最后一个 tick 的时间惩罚，不是额外再加一次。失败奖励 0 不表示整个失败回合回报为 0：前面的时间惩罚仍然存在。

discard、aborted、未完成回合不作为失败样本训练。旧 QwenOFT replay 不会被迁移或重写。单元测试包含与 `jm_pipette` 原函数直接比较；无该工作区的机器会跳过这一个来源核对测试，其余数值公式测试仍可运行。

当前按每阶段独立 episode 配置，成功是完成该阶段，不是把同一条完整五阶段轨迹在五处都标成功。连续五阶段任务的阶段切换、中间奖励和整回合成功条件尚未实现，需在上线前明确。

## 两阶段 RLToken 和 BC 预热

本实现使用本地 `ref/RLToken.pdf` / `ref/RLToken.txt` 的思路，以及固定 commit `c9e80673265b88a3eaf97d8dd3726d081ad3a889` 的 RLinf `RLTTokenTransformer`。它是 GR00T 适配，不是声称论文在 GR00T 上已有验证结果，也不是 RLinf 原生分布式 worker 的完整复刻。

流程为：

```text
两视图与状态59
    → 冻结的 GR00T
    → 第16层 VL tokens + 原 DiT 参考动作30×32
    → 学习 token encoder/decoder 的特征重建
    → 冻结 token encoder
    → 使用示教动作预热小型 actor
    → 用新采集的双腕 replay 更新 actor 与双 Q critic
```

第 1 阶段只学习读出器，GR00T 不更新。读出器为两层 encoder 和两层自回归 decoder，将 2048 维 VL 特征压缩到一个 256 维 token，最多接收 1024 个位置；输入更长会报错，不截断。按原录制划分训练与验证，重建目标只作用于有效 token。特征来自模型产生参考动作的同一次 VL forward，保留原相机裁剪、状态归一化和固定采样 seed。

随后 BC 只训练小 actor，让它从 token、normalized state59 和 frozen reference 学习示教动作。actor 输出是直接预测的动作块，不是“原始动作加一个限制很小的 XYZ residual”。开始离线 Q 更新时必须加载 BC checkpoint；没有 BC 文件就拒绝开始。

第 2 阶段冻结 GR00T 和 token，使用本地 TD3-style actor-critic：双 Q、target network、目标动作平滑、延迟 actor 更新，加人工 BC 与 frozen-reference 约束。默认 actor LR=1e-4、critic LR=3e-4、batch128、每轮600个采样更新步骤、actor 每2步更新、前20次实际 critic 更新不使用 Q actor loss、Polyak=0.005。纯人工 batch 不更新 critic，因此采样步骤数不一定等于 critic 更新数。actor LR 是新小 MLP 的起点，不是把 QwenOFT 原动作头的学习率照搬；“相同奖励”不等于全部超参数和执行方式相同。

人工通道使用人工动作 BC；非人工且有监督资格的通道使用 frozen-reference BC。当前 PICO 的 IK 路径与策略 IK 不同，因此 **凡出现 PICO 覆盖的窗口，整窗 `td_eligible=false`**，不更新 critic，也不对该窗口施加负 Q actor loss；有效人工动作仍用于 BC。无有效运动的接管保持不作为人工 BC。reference dropout=0.5 只作用于 actor 输入，不删除 reference 监督目标。没有独立 demo rehearsal batch；离线示教仅用于初始 token/BC 预热，之后的人类监督来自实际 replay。

没有在线熵调节、自动探索噪声、reward classifier 或五阶段自动串联。actor 只在新 episode 首次请求时检查并加载已发布的新 checkpoint，同一 episode 不换权重。现在微调的是依托 GR00T 的 RL policy，不是用 Q 梯度直接更新 GR00T 的几十亿参数。

## 动作与归一化

模型仍输出 32 维；RL actor 只学习前 18 维：

| 通道 | 语义 | RL 行为 |
| --- | --- | --- |
| 0–5 | 左腕 chunk-relative XYZ 与旋转向量 | 学习 |
| 6–11 | 右腕 chunk-relative XYZ 与旋转向量 | 学习 |
| 12 | 左拇指；v2 五阶段均为 chunk-relative（旧配方仅 p1 absolute） | 学习未被保持的可变通道 |
| 13–17 | 右手五通道绝对寄存器 | 学习可变通道 |
| 18–31 | 左右臂各7个绝对关节角 | 保留 frozen GR00T 输出，只作 IK 初值 |

18 是名义输出宽度，不意味着每阶段都有18个有效探索通道。`q01 == q99` 的通道无法通过归一化逆变换改变物理动作，保持 frozen reference，并从学习 mask 排除。v2 还排除 GUI 明确保持的手指通道：P1/P5 为 `[12,17]`，P2/P4 为 `[13,14,15,16,17]`，P3 为 `[12,13,14,15,16,17]`（18维 actor 索引）。测试将这些 mask 与 Bridge 的 `gr00t_phase_hand_holds` 对照。不能让 critic 利用不实际执行的数值变化获取虚假 Q 增益。

注意：mask 只定义学习资格；小 actor 在无效通道回填 frozen reference，并不直接执行物理抓握保持。真正部署时必须由 Bridge 保持当前指令值，并记录执行 mask。不能把 `physical_actions()` 的输出直接当作已经施加保持约束的机器人指令。

Critic 输入除了 token、state59、动作和逐通道有效 mask，还包括 frozen 14 维关节 seed，因为 seed 会影响 IK 姿态。不同物理量的归一化使用 checkpoint 自己的 `PolicyNormProcessor`，没有混用 π0.5 统计。最终恢复物理动作仍保留米、弧度、寄存器的不同语义。它只是提案，不是安全处理后的执行指令。

这套离线适配没有扩展安全范围或修改控制器。实际上线必须保留两腕工作空间、姿态限速、关节限制、双臂互碰、手指夹持和失联保护；归一化范围不是安全运动范围。

## 命令与环境

在 `/home/jwang3617/pipette/VLA-Precision` 执行。配置中的路径相对包含这些仓库的工作区解析。下载、元数据检查和小网络测试可使用 VLA-Precision 的 `.venv`；真实 GR00T 推理应使用已有 starVLA Python，而不是 JAX 的依赖环境。

```bash
cd /home/jwang3617/pipette/VLA-Precision
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/gr00t_rlt.py download --phase p1
for phase in p1 p2 p3 p4 p5; do
  .venv/bin/python scripts/gr00t_rlt.py check --phase "$phase" || break
done
.venv/bin/python scripts/gr00t_rlt.py status --phase p1
```

`status` 是只读清单，区分离线素材与在线 readiness；文件存在不代表其来源、训练质量或真机安全已验证。即使所有 checkpoint 文件都存在，当前 `online_ready` 仍为 false。

完整数据下载及数值准备（237 episodes；5条 frame-0 相机参考片段不能用作完整训练集）：

```bash
.venv/bin/python scripts/prepare_pipette_fulltask.py \
  --source ../datasets/g1-pipette-2view-teleop0925-5task-eerel \
  --output ../datasets/g1-pipette-fulltask-rlt-v2 --download
```

真实图像 probe，只执行模型推理，不发动作：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
../starVLA/.venv/bin/python scripts/gr00t_rlt.py probe --phase p1 --device cuda
```

默认从已准备的五阶段 dataset 读取当前阶段第一条训练样本。没有全部图像缓存时 probe 可读取已有单条视频；`cache --allow-video-decode` 可逐帧从原始视频读取，速度较慢，但不必额外生成约66 GiB的图像数组。若不传该开关，仍要求按 [数据准备指南](PI05_FULLTASK_REMOTE_TRAINING_ZH.md) 使用 `--cache-images` 准备完整图像缓存。不需要运行 JAX 训练。

以下是重现特征提取、token 与 BC 的命令。P1 已执行完成，不要直接覆盖已有输出；新实验需使用新的 output_root：

```bash
../starVLA/.venv/bin/python scripts/gr00t_rlt.py cache --phase p1 --max-observations 2048 --allow-video-decode
../starVLA/.venv/bin/python scripts/gr00t_rlt.py token --phase p1 --updates 2000
../starVLA/.venv/bin/python scripts/gr00t_rlt.py bc --phase p1 --updates 2000
```

默认训练特征上限2048，另取约20%的验证样本，不是使用全部视频帧；完整遍历可以提高上限，但先规划磁盘和推理耗时。前缀长度可变，只在 batch 内补齐。五个阶段各自训练和保存，不共用未经验证的 token/actor。

输出默认位于 `outputs/gr00t-fulltask-v2-rlt/p1/`，包含 `features/manifest.json`、`token/best.pt`、`token/metrics.jsonl`、`bc/best.pt` 和 `bc/metrics.jsonl`。新旧模型 contract hash 不同，缓存、token、BC 和 replay 不可混用。这里的 PyTorch小网络确实使用 `best.pt`，与 JAX Orbax 权重不同。已有 features/token/bc 目录不会被覆盖；需要新实验时复制配置并改变 output_root。当前不支持中途恢复 token/BC optimizer；中断后的结果保留，用新输出目录重跑。

## 新 replay 格式和离线 RL

离线 `train` 接受 NPZ 和同名 JSON manifest，格式为 `gr00t_rlt_replay.v1`。新 collector 把回合、窗口和发布审计写入 `RLT_RUN/replay.sqlite3`，新 live 工具从中导出兼容 NPZ。它不是旧的3维 SQLite，也不接受旧异步 replay。

NPZ 每行是一段无历史动作融合的 chunk window。`valid` 只证明 Bridge 发布了对应行，不代表电机已到位或底层逐帧 ACK。包含：

| 字段 | 形状或含义 |
| --- | --- |
| `token`、`next_token` | N×256，使用同一冻结 readout 的 encode_flat 输出 |
| `state`、`next_state` | N×59，checkpoint 的归一化状态 |
| `reference`、`next_reference` | N×30×32，固定 seed 的 frozen GR00T normalized reference；14维 seed 与实际传入 IK 的值一致 |
| `action` | N×30×18，真实接受的同一相对锚点下的 normalized 提案/人工动作 |
| `valid` | N×30×18 bool，有效执行通道；提前结束未执行部分和常量通道为false |
| `human` | N×30×18 bool，控制权按手臂/通道区分 |
| `bc_eligible` | N×30×18 bool，监督资格，不等于 human |
| `td_eligible` | N bool；当前任何 PICO 覆盖窗口均为 false，仅作监督学习 |
| `start_ns`、`end_ns` | N，单调时间窗口边界 |
| `terminal_ns` | N，终止结果时间；普通窗口可填0 |
| `outcome` | N，`continuing` / `success` / `failure` |
| `episode_outcome` | N，完整回合的success或failure |

JSON 必须包含 `schema`、`phase`、`contract_sha256`、`token_sha256`、`execution_contract: sequential_chunk_no_ensemble_v1` 和 `completed_episodes_only: true`。collector 仅在实际 Bridge 发布后递增行数；提前终止后的剩余行无效。SQLite 另存逐行 `published_q29`、`measured_q29`、`publication_ns` 用于审计。自主 `action` 是经过执行控制器映射的原策略提案，不伪称底层安全层后的 TCP 指令；人工 action 则从已发布关节 FK 和手部指令重新编码。当前没有电机逐条确认，也没有解决所有接触动力学/底层控制器不可观测性。

人工窗口需要重新设定命令姿态锚点，用 SO(3) 组合计算旋转差，而非相减 rotation vector。窗口第0行的相对位姿只是锚点，不能因此把该行自动标成有效的人类运动示范；绝对手指通道要另行判断。不能把安全层裁剪后的关节角直接塞进这18维动作标签。

没有异步历史队列和执行归因字段的当前 schema **拒绝** `bc_async` replay，不能用它直接复用 `jm_pipette` 缓冲区。若后续选择沿用当前 Bridge 的异步融合，应加入对应双腕调度状态，升级 schema 并明确 Q 的 action 是整块提案还是已执行前缀，而不是悄悄删除这些约束。

获得合规新 replay 后，以下命令训练的是小型 actor/critic，不是GR00T主干：

```bash
../starVLA/.venv/bin/python scripts/gr00t_rlt.py train --phase p1 \
  --replay /absolute/path/to/replay.npz --updates 600
../starVLA/.venv/bin/python scripts/gr00t_rlt.py train --phase p1 \
  --replay /absolute/path/to/replay.npz --updates 600 \
  --resume ../outputs/gr00t-fulltask-v2-rlt/p1/learner/step-00000600.pt
```

`updates` 表示本轮新增采样更新步骤数。checkpoint 包含 actor、双Q、target、两个optimizer、累计step、实际critic更新计数和随机状态；整轮完成才发布 `latest.json`。中途停止不会发布半成品，下次从上一份完整 `.pt` 恢复。当前采样是同阶段 replay 均匀抽样，没有独立 rehearsal，也不冒充复刻 `jm_pipette` 的版本平衡采样。监控使用逐更新 JSONL；本入口尚未接 W&B 或 GUI learner 卡片。

## PICO 接管实现

2026-10-02 按用户确认的相对位姿方式新增了 TWIST2 sensor-only 输入和 Bridge `--hg_input pico`。左Y、右A分别按住接管左右腕，首次按下以当前机器人指令FK锚定，旋转使用SO(3)组合；右侧握/扳机也使用相对输入，不突然张手。沿用18维动作范围，左手只开放拇指弯曲、右手开放五通道。

第一版进入任一侧接管就暂停双臂自主动作、清空整个策略队列并递增epoch，未接管侧保持。松开后等待新观测再限速交还；输入停更/突跳/暂停/复位等触发保持，恢复需要新的clutch周期。原全身teleop和默认GUI lane未修改。操作顺序和按键见 [PICO相对接管与检查](../../VLAPolicyBridge/docs/PICO_RELATIVE_INTERVENTION_ZH.md)。

新 RL collector 已接入 PICO 接管接口并记录其发布动作，但不会把旧 HG-DAgger 日志自动转成 RL replay。本机无 PICO SDK，示例配置默认拒绝真实控制，必须先验证轴向及逐控制器追踪失效行为；旧 SDK 可能无法报告单个控制器失联。GUI 新 lane 默认未启用 PICO；未完成这些检查之前只能验收自主采集与终止控制，不能宣称已具备可用的人工接管。

## 验证范围

2026-10-04 v2 迁移验证：完整237条示教已下载并准备数值数据（训练202条、验证35条；P1为54/9条）。五阶段模型元数据、学习 mask 和左拇指相对标签检查通过。P1 使用真实数据图像运行 v2 GPU probe 成功，得到 prefix `[156,2048]`、mask `[156]`、state `[59]`、reference/demo action `[30,32]`，均为有限值。`test_gr00t_rlt.py`、`test_gr00t_rlt_v2.py`、`test_pipette_fulltask.py` 合计59项通过。没有启动或控制机器人。

旧五模型适配曾用 p1、p3 真实视频执行 GPU 模型加载和推理，分别得到 VL前缀 `[156,2048]`、`[159,2048]`，以及state `[59]`、reference `[30,32]` 和demo action `[30,32]`。这是历史验证，不能代替 v2 的特征入口测试或真机验收。

定向测试覆盖奖励与来源函数一致性、终止标记等待、不同阶段和token哈希拒绝、旧XYZ replay拒绝、逐通道人工BC、常量通道冻结、critic的mask与joint-seed输入、真实小网络梯度、optimizer恢复，以及小型测试数据上的 token→BC→RL→resume 命令链路。测试数据不作为真机训练数据，也不发布可控制机器人的模型。

2026-10-04 实际数据训练：P1 使用 2048 个训练观测、409 个验证观测（按原录制划分，54/9条），token 训练2000步，最佳验证 reconstruction=3.86312635；actor BC训练2000步，最佳验证 normalized action MSE=0.04018651。两者最佳均在第2000步，不代表真机成功率。实际 unified GR00T + 新 token/actor GPU 推理检查通过，未连接机器人。尚无真实新 RL episodes，未执行真实 replay 的 actor-critic 训练。

## 真机 RL 操作流程（先 P1；需要操作员验收）

1. GUI 选择 **pipette FULL TASK · GR00T RLToken · per-phase RL**，不是原 GR00T IL lane。变量 `RLT_PHASE=p1`，`RLT_RUN=/home/jwang3617/pipette/runs/gr00t_rlt`。使用原来的全栈启动入口。不要同时运行 IL bridge 或旧策略服务。
2. 先确认台面、抓握、阶段起始姿态、kill switch、物理停机手段与相机。初始 ready/reset 仍可能移动机器人，不是只加载网络。保持 pelvis 固定。新模块没有扩大安全范围，也没有新增经过验证的双臂碰撞检测。
3. 按 X PLAY 开始一个回合。顺序执行最多30行，每行60Hz；完成后保持最后指令，等待下一次观测与推理，不做 ACT 融合/提前覆盖。`inference_hz=10` 是后台检查频率，不意味着每秒生成10块；实际推理期间机器人保持。
4. **R1+A 持续0.5秒成功；R1+X 持续0.5秒失败；F1丢弃**。按下终止组合即停止继续发送策略行，达到时长后提交标签；两种组合同时按下不会提交。R1修饰键拦截普通 X PLAY。Unitree遥控器F1和GUI聚焦时键盘F1均可丢弃；如果该型号没有F1物理键，用键盘/GUI按钮。PICO自己的A键仍是右腕clutch，不能与Unitree A混淆。
5. 成功/失败后原位保持，不自动张手、复位或切换阶段。F1丢弃当前回合；没有当前回合时丢弃最近回合。数据留在SQLite供审计，但之后的训练导出排除它。**B暂停、通信失联、reset或控制异常会将未结束回合标成aborted，不当作失败训练**；需要有效标签时应先按成功/失败，而不是先按B。
6. 回合结束后安全恢复场景，再显式PLAY。Y仍沿用原全任务语义：暂停时复位到P1起始姿态/指令；L1左右仍是带姿态运动的阶段选择，不只是改文本。其他阶段服务和collector必须与 `RLT_PHASE` 一致，不匹配时禁止RL动作。跨阶段需结束当前服务并选择对应phase，不能在P1服务里在线混训五阶段。
7. 收集一小批包含成功与失败的代表性回合后保持暂停，运行下面的有限轮次learner。首轮先少量更新验收，再决定是否使用默认600步；不要在仅一条回合上反复大量训练。更新期间不要PLAY（工具会检查采集状态并拒绝继续），同一phase禁止两个learner同时写权重。

```bash
cd /home/jwang3617/pipette/VLA-Precision
PYTHONPATH=src ../starVLA/.venv/bin/python scripts/gr00t_rlt_live.py train \
  --phase p1 --run ../runs/gr00t_rlt --updates 100
```

训练前从SQLite创建只含同phase、同contract/token、已完成且未丢弃回合的不可覆盖快照。以后重复相同命令自动恢复 `learner/latest.json`。训练过程中若发现输入回合被F1撤销，会中止本轮；**已发布权重不能通过删数据自动反向消除影响**，要回退到合适的旧checkpoint重新训练。下一次PLAY的新episode才加载新actor；失败加载不退回随机策略或IL。

离线服务检查（无监听端口、无机器人动作）与单独导出：

```bash
PYTHONPATH=src ../starVLA/.venv/bin/python scripts/gr00t_rlt_live.py serve \
  --phase p1 --run ../runs/gr00t_rlt --smoke-test
PYTHONPATH=src .venv/bin/python scripts/gr00t_rlt_live.py export \
  --phase p1 --run ../runs/gr00t_rlt
```

PICO启用前按 [PICO检查文档](../../VLAPolicyBridge/docs/PICO_RELATIVE_INTERVENTION_ZH.md) 启动sensor-only输入并完成本地配置，绝不能同时让全身teleop写机器人命令。通过检查后，在RL bridge行增加 `--hg_dagger --hg_input pico --hg_groups left_arm,right_arm --pico_config /absolute/path/to/verified.local.json --pico_file /absolute/path/to/input.json`（实际路径与输入生产者一致）。推理间隙保持，PICO也不会在该间隙驱动新运动；此保守顺序方案可能有顿挫，需真机验收后再优化。

### Loss 与文件位置

actor 总损失：`L_actor = 1.0 L_human_BC + 1.0 L_reference - w Q1`，critic预热后 `w=0.1`，预热时0；人工覆盖窗口不参与Q项。没有 `rehearsal_loss`。critic为双Q对含时间奖励的bootstrap目标的MSE，仅作用于TD合格窗口。

- `runs/gr00t_rlt/replay.sqlite3`：回合、压缩窗口、发布审计、丢弃事件。异常退出后残留running回合在collector重启时标aborted。
- `runs/gr00t_rlt/snapshots/`：每轮训练的只读NPZ/JSON快照，不能与别的实验混用。
- `outputs/gr00t-fulltask-v2-rlt/p1/learner/metrics.jsonl`：**每次采样更新**记录critic_loss、q1_loss、q2_loss、td_samples、critic_updates、target_q；actor更新步额外记录actor_loss、bc_loss/human_bc_loss、ref_loss/reference_loss、rl_loss、三个weighted loss、监督元素计数、q_pi和q_weight。非actor更新步不伪造actor loss。
- `outputs/gr00t-fulltask-v2-rlt/p1/learner/step-XXXXXXXX.pt` 与 `latest.json`：完整可恢复训练状态与原子发布指针。
- `/tmp/policy_bridge/rlt_status.json`：当前session、phase、回合、暂停原因、结果计数；GUI结果命令绑定新鲜session和episode，拒绝旧请求。

首轮验收要在低风险场景逐项检查：开始时不突跳；R1+X不触发PLAY；标签/丢弃后保持且不复位；未执行尾行不入有效mask；PICO clutch/松开/输入失联保持；进程中断不生成成功或失败样本；新checkpoint只在新episode加载。上述单元测试/GPU推理检查不能替代这些真机检查。尚未实现持续后台learner、W&B图表、自动奖励/阶段切换或底层motor ACK归因。

回归记录：VLA-Precision相关套件62项通过；Bridge选定套件342项通过，4项失败均来自既有 `pipette_align_twist2` 仿真lane（借用profile、缺少stats参数、stop sweep覆盖、remote行匹配），没有通过修改该实验或跳过其断言来掩盖这些失败。RLToken新入口及遥控器/阶段/接管相关用例通过。SQLite写入仍在控制线程窗口边界执行，因此磁盘延迟可能造成保持或watchdog中止；首轮还需测量实际控制周期，当前不宣称满足硬实时保证。

## 分阶段学习到自动完整任务：后续设计，不是已上线功能

推荐顺序为 P1 独立 RL → 其余阶段独立 RL → 相邻阶段串联 → 全程评估/联合微调。阶段训练可减少后段数据被前段失败截断的问题，但不能只从理想的 frame-0 姿态采样：P2 等后续阶段还需覆盖前一阶段真实成功后的状态，包括抓握偏差和物体位置变化。

完整执行不要求首先合并成一个共享 actor。一个冻结五任务 GR00T 配合分阶段小策略头和显式阶段管理器，也能自动完成任务；后续是否共享 RLToken/actor/critic，应通过遗忘、跨阶段泛化和完整任务成功率比较决定。

必须拆开两个操作：

- `reset_to_phase_start`：单阶段实验初始化/人工恢复。需要物体摆放与握持检查；frame-0 关节及手指到位不等于物体已正确抓住。
- `advance_phase`：成功后保持当前真实物理状态，切换指令、策略头及保持通道，不执行 frame-0 复位。短暂保持并使旧队列/在途推理结果失效，使用切换后的新观测、指令和相对动作锚点重新推理，限速恢复。

未来阶段管理器固定顺序 P1→P2→P3→P4→P5，低置信度、超时、滑落、追踪失效或接管都应阻止自动推进并进入安全保持/人工处理。阶段成功最初由人工确认，之后可用多视图成功分类器加实测姿态/手部反馈交叉验证，并要求持续满足；仅到达目标坐标、经过固定秒数或发出了闭手命令，都不足以证明成功。

建议成功条件：P1 移液枪已离架并稳定持有；P2 试管已离架并稳定持有；P3 完成用户明确规定的对准/插入/实际移液操作（这几种语义不可混为一谈）；P4 试管落座并释放、手安全退出；P5 移液枪落座并释放、手安全退出。分类器尚未训练，接触/力反馈不可假定硬件已经提供。

RL 边界也需要同步设计：当前每阶段是独立 terminal、不跨阶段 bootstrap；未来连续完整任务只在全任务结束时终止，阶段切换需记录 stage id、奖励定义、策略版本及下一阶段 observation，并使 critic 能正确处理下一阶段。不能把当前 phase-terminal replay 直接改标为整任务 replay。成功奖励只触发一次且按顺序推进，避免反复切阶段获取奖励。人工接管暂停自动推进，松手不等于本阶段成功。
