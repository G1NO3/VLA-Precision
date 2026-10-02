# G1 pipette：HIL293 relative π0.5 ACoB

完整中文环境定义、loss、GUI 启动、采集/训练/部署流程见工作区旁的
[PI05_HIL293_ACOB_RL_ENVIRONMENT.md](../../VLAPolicyBridge/docs/PI05_HIL293_ACOB_RL_ENVIRONMENT.md)。

本集成使用原生 JAX ACoB，冻结 HIL293 full-BC，仅更新 action-expert LoRA 和 critic。
控制复用 VLAPolicyBridge / TWIST2；不使用上游 UR/Franka 硬件启动器。

人工接管的异步 proposal 配对、覆盖率检查和离线补齐见
[PI05_HIL293_POLICY_PROPOSALS.md](../../VLAPolicyBridge/docs/PI05_HIL293_POLICY_PROPOSALS.md)。
`scripts/backfill_pipette_proposals.py` 默认只读检查，`--apply` 使用 GPU 追加反事实动作，
不连接机器人、不改写执行记录；learner 自动读取新增配对。

从 `/home/jwang3617/pipette` 执行只读预检查：

```bash
JAX_PLATFORMS=cpu VLA-Precision/.venv/bin/python VLA-Precision/scripts/pipette_rl.py check
```

在有人值守的 GUI 中采集 Success/Failure episode 后，PAUSE 并 Stop policy，训练：

```bash
env -u JAX_PLATFORMS CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  VLA-Precision/.venv/bin/python VLA-Precision/scripts/pipette_rl.py train \
  --config VLA-Precision/configs/pipette_rl/hil293.yaml \
  --max-updates 1000 --resume-latest
```

配置：`configs/pipette_rl/hil293.yaml`。代码：`src/vla_precision/pipette_rl/`。
checkpoint 位于工作区 `outputs/pipette-hil293-acob/learner/`，由 GUI 显式选择。
单 GPU 采集与训练交替；此入口不会连接机器人或发送动作。

当前是 primitive-step credit horizon 1 的 G1 适配，模型仍预测 10 步；没有接完整
ACoB-Stream，也没有导入离线 demonstration seed buffer。已通过离线真实 GPU 梯度、
checkpoint 和加载验证。2026-09-13 已用 4 条真实成功 episode 完成首轮 1000 步本机
更新；训练记录见工作区 `outputs/pipette-hil293-acob/learner/plots-step-00001000/`。
该轮训练早于 proposal 补齐；尚无据此证明真机成功率改善的结论。

远端 GPU 训练（`m_pi=0`）、Discard、源码交接、SQLite 快照及 checkpoint 回传命令见
[PI05_HIL293_REMOTE_LEARNER_SETUP.md](../../VLAPolicyBridge/docs/PI05_HIL293_REMOTE_LEARNER_SETUP.md)。
远端路径可不同；必须携带采集机 `source-contract.json`，不能只重算一个远端 hash。

## Reward 10：从原 step-4000 重新训练

2026-09-14 新建独立实验
`outputs/pipette-hil293-10epochs-acob-gamma0999-reward10/learner/`，从原 reward≈1
实验的 `step-00004000` 恢复 actor LoRA、critic/target 和优化器，训练至 step-5000。
原目录的 step-5000 保留用于对照；两个 step-5000 是不同奖励实验。

启动参数增加 `--reward-config VLA-Precision/configs/pipette_rl/reward10.json`：
成功终止步精确为 `10.0`，普通步及失败终止步为 `-0.01`。
保留 gamma=0.999、当前采样方式、margin、temperature 和 loss 权重。
该修改仅影响 learner 构造训练样本时的奖励，不修改 replay DB 的标签或采集契约。
旧奖励尺度的 critic 也从 step-4000 恢复，因此初期 TD loss 可能明显增大。

新目录恢复训练时，未显式指定奖励则继承 checkpoint 的 `training_rewards`；
从没有该字段的旧 checkpoint 恢复时，使用原契约的奖励定义。
改变奖励必须使用独立输出目录，防止混淆训练曲线和 checkpoint。
日志逐步记录 `reward_success_terminal`、`reward_failure_terminal`、`reward_time`，
checkpoint 记录实际 `training_rewards`。GUI 新模型标签包含 `reward 10`。

本轮可复现命令与记录：新实验目录旁 `setup/train_from_step4000.sh`、
`setup/run-plan.json`、`setup/train.log`。该脚本用于首次实验复现；继续训练使用
GUI learner 的 `--resume-latest`，不要再次指定原 step-4000 写入已有输出。

训练结束后绘制本实验曲线：

```bash
~/pipette/.conda-envs/starVLA/bin/python ~/pipette/VLA-Precision/scripts/plot_pipette_rl.py \
  --learner-dir ~/pipette/outputs/pipette-hil293-10epochs-acob-gamma0999-reward10/learner
```

## 训练结束后自动更新下次启动的 policy

GUI 选择 ACoB 模型时，policy 命令默认带 `--acob-follow-latest`。
Learner 完整训练一轮、保存最终 checkpoint 后会自动发布同目录的
`policy_latest.json`。下一次启动 policy 时只解析一次该指针，并加载该轮最终权重。
无需手动编辑 step 编号；仍由用户启动 policy，原有 Prepare Ready/PLAY 流程不变。
正在运行的 policy 不热换权重，训练中间保存或失败/停止的训练不推进发布指针。

首次为已有完成的训练补建指针，或手动检查并重新发布：

```bash
python3 ~/pipette/VLAPolicyBridge/scripts/publish_pipette_acob_policy.py
# 其他实验：增加 --learner-dir /path/to/learner
```

脚本只用标准库，检查 completed 状态、最终 step、READY、模型类型和实际文件 SHA，
通过后原子更新指针；不会启动/停止机器人或修改 checkpoint 内容。
加载时还会核对实验目录、baseline、契约、归一化和 gamma，随后正常 loader 校验权重。
没有指针的旧实验暂时沿用显式 checkpoint；已有指针损坏或来源不匹配则拒绝加载。
移除 `--acob-follow-latest` 可固定测试指定 checkpoint。

GUI 显示 `next launch step N (auto)`；当前运行中实际加载的权重以该次部署目录
`model.json` 的 `acob_step` / `acob_checkpoint` 为准。原始配置路径另存为
`configured_acob_checkpoint`，在线 replay proposal 仍记录实际加载的权重 SHA。
这些改动随下一次进程启动生效，无需为了安装它们中断正在执行的 policy。

## 优先采样最新策略下采集的整批数据

2026-09-15 起，本地 learner 默认使用 `configs/pipette_rl/terminal_tail_sampling.json` schema v4：
`latest_collection_fraction=0.5`。优先范围是**最近一个有效已结束 episode 所属的
实际采集策略下的全部有效 episode**，不限 episode 数量，也不按 policy 进程重启分批。
每个采样池以 50% 权重抽这个集合，另 50% 抽全部数据（也包含这个集合），因此该组
实际占比为 `0.5 + 0.5 × 该组原本在该池的占比`。

策略身份来自 replay 中 `policy_proposals_v1` 的在线推理来源记录：按 contract、
episode、step 和 actor 保存的 `policy_observation_key` 精确关联。RL 使用
`acob_state_sha256` 分组，不仅比较 step 编号（不同实验都可能有 step-1000），
同一份权重搬目录或重启进程仍属于同组。显式未加载 adapter 的 BC 记录使用绑定
baseline 的采集契约分组。离线 backfill 的模型不是行为策略，不作为分组证据。
人工接管时在线 inference 仍记录 proposal，因此可以识别这些 episode 的策略来源。

最新 episode 按采集时间确定。没有来源记录或出现多个策略的 episode 保留在普通池，
但不进入优先池；如果最新 episode 本身无法识别，整轮不加策略版本优先权重，
`replay.priority.collection_policy_id` 为 null，不擅自用旧策略冒充当前策略。
迟到的在线 proposal 会在后续 replay 刷新时重新识别。

人工纠正 quota 不变（当前有纠正数据时 batch=2 中占 1 个样本）；其余样本以 50% 概率
抽已完成 episode 的尾段、50% 抽全 replay。**成功和失败一视同仁**；尾段是最后
5 秒（30 Hz 下最后 150 个有效记录步，短 episode 使用全段），段内均匀抽帧。
终止帧也在该均匀池中，**没有单独的终止帧 quota**。50% 指非纠正槽位的尾段分支概率，
不是全 batch 的固定比例；全量分支也可能抽到尾段。
优先权重作用于纠正池、全 replay 池，以及所有已完成 episode 的尾段选择。
纠正/全 replay 池内部按 transition 均匀抽取，尾段先选择 episode 再选帧。
某个池没有该策略的数据时回退到该池全部数据。尚未标记、Discard、契约不匹配的
数据不进入采样池。

配置在下次启动/续训时生效，不热更新已有进程。状态中的 `replay.priority` 包含
策略身份和整个优先 episode 列表；日志记录 `sampled_priority_episode_count` 和
`replay_priority_episode_pool_size`。Checkpoint 保存 `replay_sampling` 及保存时的
`replay_priority`，绘图脚本可显示优先集合的批次采样数量。

将 `latest_collection_fraction` 设为 `0.0` 只关闭采集策略版本优先权重。
如需复现旧成功尾段设置，显式传 `--sampling-config configs/pipette_rl/success_tail_sampling.json`；
schema v1/v2/v3 保持旧含义。`--uniform-replay` 同时关闭尾段和优先集合采样。
新日志保留成功采样指标，并增加 `sampled_failure_terminal_count`、`sampled_terminal_count`、
`sampled_terminal_tail_count`，便于确认失败末段进入训练。Reward 和 gamma 不因采样改变。

**2026-09-14 来源核查：** 当 GUI catalog 已有 γ=0.999 step-3000 时，20:47 启动的
policy 进程 3096596 的 `--acob-checkpoint`、该运行的 `model.json` 及最新 episode
在线记录均指向 step-1000，SHA 为 `3de2ab8e1891f24af09875774e318c6e64dfccf4c406cab141afebef3a4dd659`。
原 GUI 按实验目录匹配并显示最新 checkpoint 的标签，导致看起来已部署 step-3000。
现已修正为显示配置中的 step，并另列 latest available。训练产生新 checkpoint 不会
自动切换已加载的 policy；已有采集记录不得据 GUI 标签改写为新策略。

## 训练结束后一次性绘制损失曲线

运行下面一条命令，读取本地 gamma=0.999、reward10 learner 的全部日志，生成 PNG 和 PDF：

```bash
~/pipette/.conda-envs/starVLA/bin/python ~/pipette/VLA-Precision/scripts/plot_pipette_rl.py
```

使用现有 starVLA 环境中的 NumPy/Matplotlib，仅在 CPU 上读取日志和绘图。
默认输出到 `outputs/pipette-hil293-10epochs-acob-gamma0999-reward10/learner/plots-step-NNNNNNNN/rl_curves.png`
和同名 PDF；步数自动取日志末步，重复运行会更新这两个文件。
脚本生成后退出，不持续刷新，也不需要训练进程仍在运行。
查看旧奖励实验时显式传入 `--learner-dir ~/pipette/outputs/pipette-hil293-10epochs-acob-gamma0999/learner`。

图中包含 critic/TD loss、actor loss、BC/improvement/reference 分量、Q、
有效 pair 加权的偏好达标率、成功终止帧与尾段帧的批次平均采样数量。
默认使用最近 50 条更新的滑动平均，并显示淡色原始 loss；从 checkpoint 元数据
识别续训边界，各阶段分别平滑。缺失的 actor 指标显示为空，不补零。
偏好指标没有有效 pair 的窗口留空；这些训练指标不等于真机成功率。

可选参数：`--smooth 100`、`--learner-dir /path/to/learner`、
`--output-dir /path/to/plots`。日志步数须严格递增，避免把重叠训练混成一条曲线。

## 快速查看本地指标

不需要激活训练环境，仅用系统 Python 标准库，只读日志、不使用 GPU：

```bash
python3 ~/pipette/VLA-Precision/scripts/pipette_rl_metrics.py
# 每5秒刷新；Ctrl+C只退出指标查看，不停止learner。
python3 ~/pipette/VLA-Precision/scripts/pipette_rl_metrics.py --watch
# 最近200步，每2秒刷新。
python3 ~/pipette/VLA-Precision/scripts/pipette_rl_metrics.py --window 200 --watch 2
```

默认读取工作区的 `outputs/pipette-hil293-10epochs-acob-gamma0999/learner/`，
统计最近100条已完整写入的learner日志。支持 `--learner-dir /path/to/learner`
查看其他实验，以及 `--json` 输出结构化数据。

显示最新与窗口平均损失、Q、动作优势、gamma、实际采样配置、checkpoint READY，
以及窗口内成功终止帧/尾段帧抽样次数（尾段包含终止帧；次数不是独立帧数量）。
偏好排序及margin达标率按有效pair数加权，无pair时显示n/a；不是机器人成功率。
`seconds`仅用于更新耗时中位数，不包含输入准备和保存。
状态来自日志，附上更新时间，不等同于进程存活检测。新一轮加载时若尚未写入指标，
会注明窗口仍属于上一轮；同时报告窗口内的NaN/Inf和未完整写入的末行。
