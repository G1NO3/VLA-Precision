# Native JAX π0.5 pipette SFT

For the current cross-repository status, migration steps and larger-GPU plan,
read [PI05_CLUSTER_HANDOFF.md](PI05_CLUSTER_HANDOFF.md).

Prepared on 2026-09-09. Native Orbax checkpoint loading and held-out inference
passed. Full-parameter training exceeds the local GPU memory budget. After the
user resumed work, a separate native JAX LoRA pilot was launched in tmux;
see the LoRA pilot section below. The full-SFT configuration is preserved.

## Data and model contract

- Source: `jren313/g1-pipette-3view-hgdagger-20260904`, revision
  `e74439f3eb2257661145571571b2dfaa5592f734`.
- Corrected source: `../datasets/g1-pipette-3view-hgdagger-20260904-e74439f3eb22`.
  Its wrist camera names are already repaired. Do not swap them again.
- Prepared cache: `../datasets/g1-pipette-pi05-30hz-e74439f3eb22`, about 12 GiB.
  This is a project-owned cache, **not** a newly converted LeRobot dataset.
  The source is v2.1 and the pinned LeRobot reader requires v3.0.
- All 46 whole recorded episodes are preserved, including approach, attachment,
  lift, disposal and ejection motion. This matches the preceding whole-episode
  BC dataset; it is not an attachment-only crop or a set of verified successes.
- 27,076 real 30 Hz transitions. Source frame pairs come from the audited expert
  label Parquets. Labels are `FK(action[next_frame,:29]) - FK(action[frame,:29])`,
  then cast to float32 exactly as in the existing reconstruction script.
  Neither the measured motion nor the low-level tracker commands are labels.
- State: measured joint positions 29D, followed by measured right-hand XYZ 3D.
  FK uses `right_rubber_hand`, pelvis frame, and the existing TWIST2 G1 URDF.
  Joint order is recorded in the cache manifest. All state inputs are current
  measurements; no target/future information is included.
- Actions: pelvis-frame XYZ translation in metres per 1/30-second step. Grip and
  wrist rotation have no learned output. Native OpenPI pads XYZ to 32 dimensions.
- Images: RGB, left wrist, right wrist; uint8 224×224 with the exact native
  OpenPI PIL bilinear resize-and-pad. Input/output SHA256 hashes are recorded
  per episode. The original videos and Parquets are read-only inputs.
- Holdout episodes: `9,10,12,18,21,23,28,29,33`. The other 37 train the model.
- Horizon 10 (0.333 seconds). Only complete windows of real transitions are
  sampled: 21,141 training chunks and 5,521 validation chunks. No chunks cross
  episodes, repeat the terminal action, or contain invented terminal zeros.
- Train-only native quantile normalization covers the actual sampled chunk
  distribution. State and action normalization assets are serialized in OpenPI
  format under `assets/pi05_full_finetune_pipette/local/g1-pipette-pi05-30hz`.
- `Pi0Config(pi05=True, discrete_state_input=True)`: normalized state enters the
  native discrete-state task prompt; actions remain continuous flow matching.
  All 27,076 prompts were checked; the maximum is 152 of 200 tokens.
- Official JAX base: `../models/openpi/openpi-assets/checkpoints/pi05_base`.
  No PyTorch conversion is used. The pinned model has 3,353,433,872 parameters.

## Configuration and resource result

`configs/stage1/pipette_pi05.yaml` selects full-parameter SFT, batch 1, EMA off,
1,000 pilot steps, 100-step warmup, peak LR 1e-5 and final LR 1e-6. No parameters
are frozen. The small pilot is for pipeline/learning checks, not a completed
baseline experiment. W&B is disabled; resume and overwrite both default false.

The native `train_step` was lowered and compiled with abstract arrays matching
these settings. This does not allocate or execute the full optimizer state.

| Quantity | GiB |
| --- | ---: |
| FP32 parameters | 12.493 |
| Adam state | 24.985 |
| Persistent train state, EMA off | 37.478 |
| Compiled step estimate, including temporaries | **52.308** |
| Physical local GPU memory | **47.788** |

The compiler estimate excludes additional runtime overhead. Full SFT is therefore
blocked on this single RTX PRO 5000. Do not interpret successful inference as
proof training fits. The resumed local run uses the separate native JAX LoRA
configuration; full-parameter SFT still needs a larger GPU.

## Commands

Run from the VLA-Precision repository root. Preparation is resumable, verifies
cached output hashes, and refuses changed input hashes in completed episodes.

```bash
source scripts/activate_training_env.sh
python scripts/prepare_pipette_sft.py
python -m vla_precision.cli --stage stage1 --mode norm-stats --config configs/stage1/pipette_pi05.yaml
python scripts/check_pipette_sft.py
python scripts/check_sft_memory.py
```

Once sufficient training memory is available, edit the absolute paths/device
settings as necessary and rerun the resource check. Start the native trainer:

```bash
python -m vla_precision.cli --stage stage1 --mode train --config configs/stage1/pipette_pi05.yaml
```

The checkpoint root is `../checkpoints/vla-precision-stage1`; native OpenPI adds
`pi05_full_finetune_pipette/pipette_pi05_pilot_20260909/<saved_step>/`. A saved
checkpoint contains `params/` and the normalization assets. To resume, set
`openpi.resume: true`, preserving the run name and data contract.

Evaluate a checkpoint directory containing `params/`:

```bash
python scripts/evaluate_pipette_sft.py \
  --checkpoint /path/to/checkpoint \
  --output ../outputs/pi05-sft-20260909/sft-validation.json
```

Defaults: 16 evenly spaced complete chunks per held-out episode, fixed noise
seed 42, 10 flow sampling steps, per-episode metrics and equal-episode aggregate.
Use `--samples-per-episode 0` for every complete validation chunk. No target
actions enter inference. Recompute normalization if data/state/actions/horizon
change, and keep the same evaluation selection/noise for before/after comparisons.
For trained checkpoints the runner loads saved normalization assets and checks
that they match the configuration. The official base uses the configuration's
train-only assets. Reports record the normalization hash.

## Checks and initial inference result

Artifacts are in `../outputs/pi05-sft-20260909/`:

- `resolved.yaml`: resolved pilot configuration.
- `memory-shape.json`, `memory-compile.json`: abstract resource measurements.
- `data-check.json`: episode separation, all action chunk boundaries, all prompt
  lengths, action padding, normalization inversion and a real JAX loader batch.
  18 independently decoded source camera frames matched the cache exactly.
- `base-validation.json`, `base-validation.predictions.json`: successful native
  JAX base checkpoint inference on 144 held-out chunks.
- `prepare.log`, `norm-stats.log`, `data-check.log`, `memory-compile.log`,
  `base-validation.log`: execution logs, including diagnostic attempts.

The **unfinetuned** base model on that pilot subset has equal-episode chunk
XYZ RMSE **0.3258 mm/step**, first-action RMSE **0.3132 mm/step**. The zero-action
reference on the same chunks is **0.1618 mm/step**. These are open-loop prediction
metrics under the pipette normalization, not attachment success rates and not
scores directly comparable with prior baselines on different evaluation samples.
These results were recorded before training. Updated LoRA pilot results are
recorded separately below. No real-robot evaluation has been performed.

The native SFT and offline validation operate at 30 Hz; the current robot actor's
10 Hz action contract still requires a deliberate deployment adapter and testing.


## Native JAX LoRA pilot (resumed 2026-09-09)

Configuration: `configs/stage1/pipette_pi05_lora.yaml`.
Runner: `scripts/run_pipette_lora_pilot.sh`.
Tmux session: `pipette-pi05-sft`.
Artifacts: `../outputs/pi05-lora-sft-20260909/`.

The model uses `gemma_2b_lora` and `gemma_300m_lora`, with the native OpenPI
`get_freeze_filter()` recipe. This freezes the non-LoRA LLM weights. The vision
encoder and small action/time projections remain trainable, as in upstream
OpenPI's low-memory SFT example. It is **not adapter-only training** and is not
full-parameter SFT.

- Total parameters: 3,403,421,456.
- LoRA parameters added: 49,987,584.
- Total trainable parameters: 466,957,072 (LoRA plus vision/projections).
- Batch 4, 1,000 updates, EMA off, warmup 100 updates, peak LR 5e-5, final LR 5e-6.
- Same 32D state, 10-step XYZ action chunks, three views, split and prompt.
- Identical train-only normalization assets (SHA256
  `c8126474b74a5a0218f3f8d8afebc6d1435d8074808260504715fdf7e2a7c29b`).
- Abstract compiled memory: 16.933 GiB. Actual `nvidia-smi` memory includes
  allocator reservations and runtime memory; measurements are in `gpu-usage.csv`.
- Runner writes `status.txt`, `train.log` and `validation.log`, and evaluates the
  final checkpoint on the same 144 held-out chunks/noise as the base comparison.

Monitor from another terminal:

```bash
tail -f ../outputs/pi05-lora-sft-20260909/train.log
cat ../outputs/pi05-lora-sft-20260909/status.txt
```

LoRA checkpoints retain the adapters and need the LoRA model configuration for
inference. They must not be loaded with the full-model configuration, which can
discard adapter parameters. The current Stage-II initialization route requires
a full-model configuration; adapter merging or a tested Stage-II integration is
still needed before using this checkpoint for ACoB. This pilot is offline SFT.

### Completed pilot result

All 1,000 native JAX optimizer updates completed and the final Orbax checkpoint
reloaded successfully for inference. Training, including checkpoint saves, took
approximately 10 minutes 24 seconds. Sampled peak GPU memory was
32.70 GiB.

| Equal-episode held-out metric | Base | LoRA SFT |
| --- | ---: | ---: |
| Chunk XYZ RMSE (mm/step) | 0.325798 | 0.201649 |
| First-action XYZ RMSE (mm/step) | 0.313232 | 0.191646 |

The zero-action chunk reference scores 0.161813
mm/step, which is better than the trained pilot. The SFT pipeline works, but this
short run has not established a useful pipette policy. The comparison covers
the exact same 144 held-out chunks/targets, normalization and random noise.
It is not a complete validation sweep or a robot success-rate result.

Final checkpoint: `/home/jwang3617/pipette/checkpoints/vla-precision-stage1/pi05_lora_finetune_pipette/pipette_pi05_lora_pilot_20260909/999`.
Load it with `configs/stage1/pipette_pi05_lora.yaml`.

Detailed results: `../outputs/pi05-lora-sft-20260909/RESULTS.md`,
`summary.json`, `validation.json`, `validation.predictions.json`,
`training-metrics.json`, and `gpu-usage.csv`. The tmux training session exits
after validation; status is `complete`. No training is left running.

## GUI inference preview (2026-09-09)

The sibling `VLAPolicyBridge/gui` now exposes the trained native JAX LoRA model
at `/pi05`; see [the operator GUI guide](../../VLAPolicyBridge/gui/README.md#π05--vla-precision-preview).
The worker entry point is `scripts/serve_pipette_preview.py`, owned by the GUI
controller and launched in this repository's `.venv`. It uses the same native
policy factory, discrete-state transforms, 10 flow steps and checkpoint
normalization as `scripts/evaluate_pipette_sft.py`. No checkpoint conversion is
involved.

Dataset preview displays complete 10-step windows, exact resized RGB pixels,
raw measured state and prediction/label XYZ increments in mm per 30 Hz step.
A separate read-only source accepts existing HIL schema-v1 observation packets;
it does not start the robot actor or write policy actions. This packet's 10 Hz
wire action convention is not an execution interface for the 30 Hz model.

Validation used checkpoint `999` of `pipette_pi05_lora_pilot_20260909`:

- Dataset episode 9, frame 0, seed 42 matches the earlier native offline
  evaluation predictions (`atol=1e-8 m, rtol=1e-5`). Repeated GUI requests are
  identical with the same seed.
- First inference including JIT was about 5.03 s; warmed requests observed
  about 72–73 ms on this machine. These are individual model-call timings,
  not a throughput benchmark or full robot-loop latency measurement.
- A separate loopback Redis stub replayed dataset state and JPEG images through
  the live adapter. It captured only `GET hil:observation`; the native model
  returned valid 10×3 actions. A 5-second-old packet was rejected. No real
  robot/publisher was used for this test.
- Chromium checks exercised desktop/mobile layout, images, curves, frame
  selection, continuous preview, missing-live-observation errors and recovery.
  Run records and screenshots are in `../outputs/pi05-gui-preview/`.

This completes the preview layer. Real camera/state parity, a separately
validated 30 Hz action execution bridge, intervention controls and ACoB
integration remain subsequent work. Preview error curves measure action-label
error and cannot establish attachment success or physical tip alignment.


## Published checkpoint (2026-09-09)

The native LoRA pilot is available in the public Hugging Face model repository
[GeertWang/pi05-pipette-lora-bc-20260909](https://huggingface.co/GeertWang/pi05-pipette-lora-bc-20260909),
revision `f09e444ffb651fb72bb559b34c57005fe115fd97`.
The native checkpoint directory inside the downloaded snapshot is
`checkpoints/999`. The 9.52 GB upload includes full parameters, training state,
normalization, training configs, the training-time source overlay, evaluation
results and a portable inference loader. All 51 remote file sizes and hashes
were verified; anonymous downloads of metadata/assets/loader were also checked.
Local upload verification: `../outputs/hf-pi05-pipette-upload.json`.


## Integration branches

The pipette training and preview worker changes are published on
[G1NO3/VLA-Precision:jy-vla-precision](https://github.com/G1NO3/VLA-Precision/tree/jy-vla-precision).
The matching operator GUI is on
[birbirll/VLAPolicyBridge:jy-vla-precision](https://github.com/birbirll/VLAPolicyBridge/tree/jy-vla-precision).
Keep the repositories as siblings in the pipette workspace; the GUI controller
launches the worker through `VLA-Precision/.venv/bin/python`.
