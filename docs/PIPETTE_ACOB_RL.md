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
