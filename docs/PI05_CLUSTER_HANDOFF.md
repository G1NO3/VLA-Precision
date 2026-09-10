# π0.5 pipette: repository handoff and larger-GPU plan

Updated 2026-09-10. Source workspace: `/home/jwang3617/pipette`.
This is the starting document for continuing the work on another cluster.
The status below describes completed work; the GPU experiments in section 8 are
proposed next steps, not runs that have already happened.

## 1. Current result and decisions

We implemented native **JAX π0.5** SFT in VLA-Precision, trained a **1,000-update
LoRA BC pilot**, evaluated it, published its complete checkpoint, and integrated
inference preview into VLAPolicyBridge's GUI. We chose native JAX instead of
JAX → PyTorch → starVLA conversion. State input uses native discrete state tokens.

The pilot improves held-out action RMSE over the base model, but is worse than
zero action on the same metric. It has **not** demonstrated successful real-robot
attachment. No π0.5 robot action execution, ACoB integration, multi-GPU training,
or multi-node training has been validated. The GUI only previews/records outputs.
No π0.5 training, preview worker or GUI server was running at this handoff audit.

The user accepted LoRA for the current machine, asked for GUI integration and a
public checkpoint, and now wants to move to a cluster with more GPUs. The agreed
robot sequence is read-only real observations → short, supervised BC execution
with an explicit action adapter → ACoB. A perfectly successful BC policy is not a
prerequisite for ACoB; coherent inputs and useful coarse actions are prerequisites.

## 2. What changed in each repository

The revisions below are **implementation revisions before this documentation
commit**. Use the `jy-vla-precision` branches to obtain the handoff as well.
A machine-readable inventory is in [pi05_cluster_manifest.json](pi05_cluster_manifest.json).

| Repository | Branch / revision at audit | π0.5 work and migration role |
|---|---|---|
| [VLA-Precision fork](https://github.com/G1NO3/VLA-Precision/tree/jy-vla-precision) | `jy-vla-precision`, `f5126a7d3e4a6905ed3a0571a887aa8e5982fda8` | Main implementation: environment, dataset, full/LoRA configs, training, evaluation, preview worker and reports. Fork used because the available GitHub account cannot push to `scy-v/VLA-Precision`. |
| [VLAPolicyBridge](https://github.com/birbirll/VLAPolicyBridge/tree/jy-vla-precision) | `jy-vla-precision`, `43f4b1939e76715113b403b42d96222b01ee2170` | Added `/pi05` GUI and stdlib worker controller. Existing HIL/FK/expert-label tools are dependencies; their earlier work is not part of this π0.5 commit. Based on `jy-HIL-SERL@aedf5e0`. |
| [TWIST2](https://github.com/G1NO3/TWIST2/tree/hil_control) | `hil_control`, `6acb9d25917283f577789803a5824185317c20b5` | Read hardware/control code and used the G1 URDF for FK. No changes for this π0.5 work. Required to rebuild data; not required to load a complete prepared cache. |
| [starVLA](https://github.com/LidarDexManip/starVLA/tree/jy-pipette-tube-gr00t) | `jy-pipette-tube-gr00t`, `65a13bb00b75f3b1d3f2dab5e92a824c04b6b22f` | Inspected its PI05/PyTorch interface and conversion route. No π0.5 changes or pipette π0.5 training here. Existing interface supports discrete state, though some example YAMLs disable it. Not needed by the selected native JAX pipeline. |
| OpenPI dependency | `2d70d966582e711128ad8358d8dbf23d2cc3d658`, installed from Git through `uv.lock` | Native π0.5 model, tokenizer, normalization, flow loss, policy loading and FSDP. No independent OpenPI checkout was edited. Adaptations live in VLA-Precision. |
| RFT | `jy`, `bc8c81237f1133d70f6a83f790507374f054ef74` | No work for this π0.5 effort. Two untracked pipette-phase scripts pre-exist; do not attribute them to this handoff or silently bundle them with π0.5 changes. |
| IsaacLab-3.0 | `release/3.0.0-beta2`, `d7824157a82b8466f6b24e85892754007ce2daab` | No changes or simulation validation for this π0.5 effort. Not needed for offline SFT. |
| g1_redball_eval | `oldisaac51`, `02fddc3fa8c0b2b6abcee5a27f7c26fdfdfd1ecb` | No π0.5 work. Existing untracked `isaac6/` is separate work. |
| unitree_sim_isaaclab | `main`, `e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc` | No changes for this π0.5 effort. Existing modified/untracked simulation files must be preserved separately if the whole workspace is migrated. |

VLA-Precision and VLAPolicyBridge were clean at their audited implementation
revisions. Existing changes in the other projects were not committed by this work.

### VLA-Precision file map

- Environment: `.python-version`, `pyproject.toml`, `uv.lock`,
  `scripts/setup_training_env.sh`, `activate_training_env.sh`,
  `runtime-linux-64.lock`, `check_training_env.py`; `.runtime` is ignored.
- Data: `scripts/prepare_pipette_sft.py`, `src/vla_precision/data/pipette.py`;
  `integrations/openpi/data_configs.py`, `data_loader.py`, `norm_stats.py`;
  `integrations/openpi/policies/pipette.py`.
- Training: `configs/stage1/pipette_pi05.yaml` (full SFT),
  `pipette_pi05_lora.yaml` (completed pilot); config registry, CLI and native
  trainer adjusted to allow the additional LoRA recipe and configurable JAX cache.
- Assets: both pipette config asset directories contain identical train-only
  `norm_stats.json`. These small assets are in Git; weights/datasets are not.
- Validation/reporting: `check_pipette_sft.py`, `check_sft_memory.py`,
  `evaluate_pipette_sft.py`, `pipette_axis_errors.py`, `summarize_pipette_pilot.py`.
- Original launcher: `run_pipette_lora_pilot.sh`. It is tied to the dated pilot
  output directory, GPU 0 and existing memory report; do not use it unchanged to
  start a new cluster experiment.
- GUI worker: `scripts/serve_pipette_preview.py` and
  `integrations/openpi/pipette_preview.py`. Uses native JAX in this repo's `.venv`.
- Tests: `tests/test_pipette_preview.py` covers live state/images, freshness,
  read-only Redis access and recording.
- Reports: [PIPETTE_JAX_SFT.md](PIPETTE_JAX_SFT.md),
  [PI05_BC_REPORT_ZH.md](PI05_BC_REPORT_ZH.md),
  [LOCAL_TRAINING_ENV.md](LOCAL_TRAINING_ENV.md). The environment document records
  the initial setup stage; its statement that training had not started is historical.

### VLAPolicyBridge file map

- `gui/pi05.html`: model load/stop, dataset episodes/frames, continuous preview,
  live HIL input, three RGB views, XYZ predictions/targets, state, logs and JSON export.
- `gui/pi05_preview.py`: isolated subprocess lifecycle; uses the sibling
  `VLA-Precision/.venv/bin/python`, JSON-lines stdin and atomic result/status files.
- `gui/server.py`: `/pi05`, `/pi05/status`, `/pi05/log`, `/pi05/frame`, and guarded
  POST routes `/pi05/start`, `/pi05/stop`, `/pi05/infer`. Existing Host/CSRF checks
  remain active. END SESSION starts the controller stop before waiting for the
  preview worker; regular E-stop retains model services. Console exit cleans up
  its preview worker.
- `gui/index.html`: link to the new page. No new robot scenario/execution lane.
- `tests/test_gui_pi05_preview.py`, `tests/test_gui_controls.py`: route guards,
  process lifecycle, one request in flight, and stop ordering.
- `gui/README.md`: operator instructions, checkpoint link and matching source branch.

## 3. Data and model contract: preserve these during migration

| Item | Audited definition |
|---|---|
| Source dataset | `jren313/g1-pipette-3view-hgdagger-20260904` at `e74439f3eb2257661145571571b2dfaa5592f734` |
| Scope | All 46 recorded episodes: approach, attachment, lift and disposal/ejection motion; not an attachment-only crop or an audited success-only set |
| Prepared cache | `vla_precision.pipette_sft.v1`, 27,076 real 30 Hz transitions |
| Split | 37 train / 9 validation episodes; validation IDs `9,10,12,18,21,23,28,29,33` |
| Chunks | Horizon 10; 21,141 train / 5,521 validation complete windows; no terminal padding or episode crossing |
| State | Measured joint positions (29) + FK of those measured joints for right-hand XYZ (3); current observations only |
| Action label | `FK(reference_action[next,:29]) - FK(reference_action[current,:29])`; checked against audited `expert_delta_xyz` Parquets; not measured motion or tracker commands |
| Frame / units | Pelvis; metres per 1/30-second step; `right_rubber_hand` reference link, not a calibrated pipette-tip TCP |
| Images | `rgb`, `wrist_left`, `wrist_right`; already-corrected names; native RGB PIL resize/pad to 224×224. Do not swap wrists again. |
| State encoding | Train-only quantile normalization, native discrete state tokens; max observed prompt 152 / configured 200 tokens |
| Actions | XYZ padded to native 32 dimensions; continuous flow matching; output sliced to 3 dimensions; no gripper/orientation prediction |
| Prompt | `Attach a pipette tip to the pipette, lift it, then move to the disposal box and eject the tip.` |

Normalization SHA256:
`c8126474b74a5a0218f3f8d8afebc6d1435d8074808260504715fdf7e2a7c29b`.
URDF SHA256:
`824ce02c2c1e489aa8dece47a10fe02d1872289c0e6d01ba51abd291e66b7b2c`.
FK source SHA256:
`d7ae137b39c547f4939fc06e51dd0e8e50bbccb7a4e329bc78d597410a55ad66`.
State joint order and per-episode input/output hashes are in the prepared cache's
`manifest.json` and each `episode_*/record.json`.

Native flow loss uses `A[B,10,32]` after XYZ normalization and zero padding:
`epsilon ~ N(0,I)`, `t = 0.999 * Beta(1.5,1) + 0.001`,
`x_t = (1-t) A + t epsilon`, target `u = epsilon - A`.
The scalar objective is `mean((v_theta(x_t,t,observation)-u)^2)` over B×10×32.
All 29 padded dimensions participate in this loss; their target flow is noise,
not simply zero. The loss is neither XYZ displacement error nor a robot success metric.

## 4. Completed training and evaluation

Original GPU: one RTX PRO 5000 Blackwell, 48,935 MiB (47.788 GiB), driver 610.43.02.
Full SFT B=1, horizon=10, EMA off compiled to an estimated **52.308 GiB** including
temporaries; persistent FP32 parameters + Adam state alone were 37.478 GiB.
This was an abstract compile, not a successful full-SFT optimizer update.

Completed LoRA pilot:

- `gemma_2b_lora` (rank 16) + `gemma_300m_lora` (rank 32).
- Native freeze filter freezes non-LoRA LLM weights; vision and small projections
  also train. 49,987,584 adapters; **466,957,072 total trainable parameters**.
- B=4, 1,000 updates, seed 42, EMA off; warmup 100, peak LR 5e-5, decay to 5e-6.
- About 10m24s including saves; sampled peak GPU usage 32.70 GiB.
- First logged loss 0.3847; last logged interval at step 990: 0.0245.
  Actual final checkpoint is index `999`, after 1,000 optimizer updates.
- 4,000 sampled chunks is only about 0.19 training-set passes, so this is a short pilot.

Held-out evaluation: 16 uniformly spaced complete chunks per validation episode,
144 total, seed 42, 10 flow steps. Same samples/noise/normalization for base and pilot.

| Policy | XYZ component RMSE, mm/step | First-action XYZ RMSE, mm/step |
|---|---:|---:|
| Official JAX base | 0.325798 | 0.313232 |
| LoRA pilot | 0.201649 | 0.191646 |
| Zero action | 0.161813 | — |

LoRA X/Y/Z RMSE: 0.233979 / 0.160636 / 0.203560 mm/step. Aggregate means squared
errors within each episode, averages episodes equally, then takes the square root.
These are action-label errors, not hand/tip position errors. Full validation over
all 5,521 chunks and physical success-rate evaluation remain undone.

135 targeted tests passed: 13 worker/input tests and 122 bridge GUI/control tests.
Chromium desktop/mobile checks exercised images, curves, continuous preview,
missing/stale live input, recovery, reload, stop and JSON download. Real checkpoint
inference matched offline evaluation; warmed calls observed ~72–73 ms, first JIT
~5.03 s. Synthetic live HIL packets were tested on a separate loopback Redis stub;
no real robot was used. These timings are examples, not a hardware benchmark.

## 5. Published artifacts and what must move

Public model: [GeertWang/pi05-pipette-lora-bc-20260909](https://huggingface.co/GeertWang/pi05-pipette-lora-bc-20260909).
Pin revision `f09e444ffb651fb72bb559b34c57005fe115fd97`.
Inside the snapshot use **`checkpoints/999`** as the native checkpoint directory.
The 9,524,680,531-byte upload has 51 files, all sizes/hashes verified, and public
anonymous download verified. It contains full model parameters (~6.34 GB),
training state (~3.19 GB), normalization, configs, reports, a training source
overlay and `source/load_policy.py`. It is not an adapter-only artifact or a
PyTorch checkpoint. Keep the complete Orbax/OCDBT `params/` tree.

The Git branches contain newer GUI/integration documentation than the HF source
overlay. Prefer the branches; the overlay reconstructs the exact training-time
modifications over upstream commit `75eb56a35ca6e8c3ee0a0a3d41b99e88be469b81`.
The snapshot loader can infer without original training dataset paths and was
checked against GUI output at float32 precision.

Paths below are relative to the old workspace. Sizes from `du` are approximate.

| Artifact | Size | Migration requirement |
|---|---:|---|
| `datasets/g1-pipette-pi05-30hz-e74439f3eb22/` | 12 GiB | Copy for quickest restart. Prepared NumPy cache is not on HF. Rebuild alternative below. |
| `datasets/g1-pipette-3view-hgdagger-20260904-e74439f3eb22/` | 4.8 GiB | Copy or re-download the pinned HF dataset; needed to rebuild/check camera frames. |
| `datasets/g1-pipette-3view-hgdagger-20260904-expert-xyz-30hz/actions/` | 1.8 MiB | **Copy**, together with `RECONSTRUCTION.json`; current preparation script requires these audited frame-pair/XYZ Parquets, even with the raw HF dataset available. |
| Same expert directory's `replay.sqlite3` | Most of the 2 GiB directory | Not needed for current π0.5 cache preparation/training. Preserve separately if continuing historical HIL/reconstruction work. |
| `models/openpi/openpi-assets/checkpoints/pi05_base/` | 12 GiB | Copy or download official base for fresh LoRA/full SFT. Not needed just to infer from complete trained parameters. |
| `models/openpi/big_vision/paligemma_tokenizer.model` | Small | Copy for offline use, or allow native tokenizer download on first load. |
| Trained checkpoint `.../pipette_pi05_lora_pilot_20260909/999` | 9.52 GB | Download from the public model snapshot; local duplicate need not also be transferred. |
| `outputs/pi05-sft-20260909/` | 300 KiB | Preserve base predictions, original hashes and memory/data checks. |
| `outputs/pi05-lora-sft-20260909/` | 848 KiB | Preserve full train/eval logs, per-chunk predictions, curves, GPU samples and provenance. HF has a useful subset, not every local file. |
| `outputs/pi05-gui-preview/` | 6.1 MiB | Preserve screenshots, validation report, exact NPZ input examples and logs; useful for parity tests. Contains recorded observation images. |
| `outputs/hf-pi05-pipette-upload.json` | Small | Upload revision/file verification record. |
| `ref/VLA-precision.pdf` | Small | Optional reference for prior paper checks. |

A copy list is provided in [pi05_transfer_files.txt](pi05_transfer_files.txt).
It includes the prepared/raw data, audited labels, tokenizer and evidence; code
comes from Git and checkpoints can come from HF/the official base source.
For an SSH-accessible destination, run from the **old workspace** after replacing
both destination placeholders:

```bash
rsync -ar --info=progress2 \
  --files-from=VLA-Precision/docs/pi05_transfer_files.txt \
  /home/jwang3617/pipette/ NEW_CLUSTER:/path/to/pipette/
```

The explicit `-r` matters with `--files-from`. To avoid redownloading the base,
copy `models/openpi/openpi-assets/checkpoints/pi05_base/` separately as well.
This list does not copy unrelated dirty repositories, credentials, environments,
compiled caches or duplicate HF upload staging directories.

Recreate `.venv`, `.runtime` and JAX compilation caches on the destination.
They contain absolute prefixes or hardware-specific artifacts. Existing old
starVLA Python/compiler environment was only reused as a source of GCC/sysroot
and plotting tools; it is not a required algorithm dependency.

**Prepared-cache relocation:** numeric/image reads work after copying and changing
`data.lerobot_root`. The manifest's `source`/`urdf` and per-episode provenance keys
still refer to old absolute paths. Keep the original manifest for provenance;
for camera validation, rebase a working copy's top-level `source`/`urdf` to the
new locations without altering numeric/image hashes. Do not rerun the preparation
script into that copied cache at a different root: its absolute input-path keys
will intentionally fail the provenance equality check. Rebuild into a **fresh**
output directory if regeneration is desired.

**Rebuild dependencies:** `prepare_pipette_sft.py --workspace <new-workspace>`
expects raw HF data, the expert `actions/` directory, the sibling bridge FK module
and TWIST2 URDF. It rechecks FK deltas against the labels. The original label
reconstruction script additionally requires a materialized dataset and a replay
SQLite input; raw HF download alone does not satisfy that script. Preserve the
small audited label files instead of silently inventing a different resampling.

## 6. Destination setup and reproduction

Keep `VLA-Precision`, `VLAPolicyBridge`, `TWIST2`, `datasets`, `models`,
`checkpoints`, and `outputs` as sibling directories. Choose a task-specific
workspace variable; the commands below do not assume the old username.

```bash
export PIPETTE_WORKSPACE="$HOME/pipette"
mkdir -p "$PIPETTE_WORKSPACE"
cd "$PIPETTE_WORKSPACE"
git clone --branch jy-vla-precision https://github.com/G1NO3/VLA-Precision.git
git clone --branch jy-vla-precision https://github.com/birbirll/VLAPolicyBridge.git
git clone --branch hil_control https://github.com/G1NO3/TWIST2.git
# Pin TWIST2 for exactly the audited URDF/FK contract.
git -C TWIST2 checkout 6acb9d25917283f577789803a5824185317c20b5
```

The audited TWIST2 `hil_control` branch is on the **G1NO3 personal remote**,
not the original `MLeggiero/TWIST2` remote, which has no
`hil_control` branch. Clone commands assume fresh destination directories. The two implementation SHAs
in section 2 are useful for bisecting; the branch tips also carry this handoff.
Copy the artifacts in section 5, or download the public artifacts:

```python
from pathlib import Path
import os
from huggingface_hub import snapshot_download
w = Path(os.environ['PIPETTE_WORKSPACE'])
snapshot_download(
    'GeertWang/pi05-pipette-lora-bc-20260909',
    revision='f09e444ffb651fb72bb559b34c57005fe115fd97',
    local_dir=w / 'models/pi05-pipette-lora-bc-20260909',
)
snapshot_download(
    'jren313/g1-pipette-3view-hgdagger-20260904', repo_type='dataset',
    revision='e74439f3eb2257661145571571b2dfaa5592f734',
    local_dir=w / 'datasets/g1-pipette-3view-hgdagger-20260904-e74439f3eb22',
)
```

Run downloads in an environment with `huggingface_hub` installed. Trained inference
checkpoint then lives under `models/pi05-pipette-lora-bc-20260909/checkpoints/999`.
For fresh SFT the official source remains `gs://openpi-assets/checkpoints/pi05_base`;
copy the verified local base or use native OpenPI download utilities with network
access. Its original file manifest is in `outputs/pi05-sft-20260909/`.

Environment reproduction:

```bash
cd "$PIPETTE_WORKSPACE/VLA-Precision"
# Requires uv, conda and a working C compiler/Linux headers for evdev.
# Set VLA_CONDA_EXE or VLA_BUILD_TOOLCHAIN only if those defaults need adjustment.
bash scripts/setup_training_env.sh
source scripts/activate_training_env.sh
python scripts/check_training_env.py --video-root \
  ../datasets/g1-pipette-3view-hgdagger-20260904-e74439f3eb22/videos/chunk-000
```

Validated baseline: Linux x86_64, Python 3.11.16, JAX/jaxlib/CUDA plugin 0.5.3,
Flax 0.10.2, NumPy 1.26.4, torch 2.7.1+cu128, torchvision 0.22.1+cu128,
TorchCodec 0.5, FFmpeg 7.1.1. OpenPI revision is in section 2; LeRobot is
`d6ea3bbce0fc8c75138f02639ac4ceaf12a67829`, AgentLace
`cf2c337c5e3694cdbfc14831b239bd657bc4894d`.
The CUDA 12.8 torch pin fixed a Blackwell kernel incompatibility on the old host.
Revalidate against the actual new driver/GPU architecture rather than assuming
all old binaries are appropriate. Use `uv sync --frozen --group stage2` (includes
stage1) or the setup script. Plain `uv sync` with no group removes training
packages because upstream `default-groups=[]`; use `uv run --no-sync` once set up.
Known upstream dependency metadata warnings about `typing` and OpenCV remain
recorded in `LOCAL_TRAINING_ENV.md`; setup was not a zero-warning `uv pip check`.

Create destination-specific YAML copies. Rebase these fields in both recipes:

- `data.lerobot_root` → the copied/rebuilt prepared NumPy cache, not raw LeRobot.
- `openpi.initialization_checkpoint` → official base **`params/`**, for fresh training.
- `paths.checkpoint_root`, `paths.cache_root`, `paths.openpi_assets_root`.
- `openpi.exp_name` → a new run name; keep `resume:false`, `overwrite:false` for new runs.
- `cuda_visible_devices` → the scheduler-assigned GPU mask; `openpi.fsdp_devices`
  and global `batch_size` must agree with the visible device count.

Do not recompute statistics merely because the filesystem root changed. The
committed assets and saved checkpoint statistics should still match. A changed
split, horizon or state/action definition requires newly versioned statistics.

Reproduce the pilot before changing training:

```bash
mkdir -p ../outputs/cluster-reproduce
# Here pipette_pi05_lora_cluster.yaml means your rebased copy of the LoRA YAML.
python scripts/check_pipette_sft.py \
  --config configs/stage1/pipette_pi05_lora_cluster.yaml \
  --output ../outputs/cluster-reproduce/data-check.json
python scripts/evaluate_pipette_sft.py \
  --config configs/stage1/pipette_pi05_lora_cluster.yaml \
  --checkpoint ../models/pi05-pipette-lora-bc-20260909/checkpoints/999 \
  --samples-per-episode 16 --seed 42 \
  --output ../outputs/cluster-reproduce/validation.json
python -m pytest tests/test_pipette_preview.py -q
```

Create `../outputs/cluster-reproduce/` before the data check. Camera checks need
the manifest's raw-data path rebased as described above. Compare against the
published 144-chunk metrics with normal floating-point tolerance across hardware.
The snapshot `source/load_policy.py` provides a quicker recorded-NPZ inference
check without requiring the full dataset. Do not load the LoRA checkpoint with
the full-SFT config: adapters can be discarded or fail to match.

## 7. GUI / robot separation on a cluster

The preview controller currently assumes a **local sibling worker**, not a remote
model RPC server, and hardcodes `CUDA_VISIBLE_DEVICES=0`. It does not inherit a
scheduler allocation correctly in all environments; adapt that before using it
on a shared cluster. Keep training and preview in separate allocations/processes.
A browser can reach a loopback GUI through SSH forwarding, but that does not
connect the cluster to robot cameras or Redis automatically.

The live path only issues `GET hil:observation`. Expected packet: schema v1,
29 measured joints + 3 measured hand coordinates, three JPEG views, pelvis frame,
`image_color_after_decode=BGR`, timestamp age ≤1 second. Actor wire metadata says
`metres_per_10hz_step`; model predictions remain native **30 Hz**. There are no
writes to `hil:policy_action_xyz`, no actor startup, and no GO/Reset controls on
this page. Never reinterpret the first predicted 30 Hz delta as a 10 Hz delta.

Remaining deployment work: actual camera/state parity, latency measurements,
a deliberate chunk/execution-rate adapter, limits/short runs/operator takeover,
and physical success metrics. Cluster training can proceed independently; extra
GPUs do not establish robot safety or task competence. A remote inference service,
if desired, is a separate implementation from this local preview integration.

## 8. Planned use of more GPUs

These are prioritized experiments and acceptance gates, not measured new-cluster
results. First inspect allocated GPU models/count/VRAM, interconnect, driver,
host RAM, local storage and scheduler policy. No fixed cluster launch script is
claimed to be ready before those details are known.

### Phase A — reproduce the pipeline (first allocation)

1. Restore the pinned code, artifacts and normalization. Verify data hashes and
   run the environment/data checks and 144-chunk checkpoint evaluation.
2. Compare decoded views and measured state with saved GUI NPZ examples. Retain
   the original base/LoRA/zero-action results as immutable reference artifacts.
3. Gate: expected shapes, finite outputs, same data split and approximately the
   prior metrics. Resolve mismatches before spending a long allocation.

### Phase B — validate native full-SFT sharding

Prefer **one node with 4 GPUs** for the first distributed experiment. Native
OpenPI constructs a `(data_parallel, fsdp)` mesh. With 4 visible GPUs,
`fsdp_devices:4` shards eligible large tensors; `fsdp_devices:1` replicates model
state despite exposing more GPUs. Small arrays and unshardable tensors can remain
replicated, so memory is not simply total memory divided by four.

Start with the existing **full** pipette config initialized from official
`pi05_base`, horizon 10, 32 action dimensions, discrete state, EMA off. Use global
batch 4 for the initial four-GPU smoke test, then test batch 16 for the planned
main runs. GPU count must divide the global batch; FSDP width must divide the
visible device count. On a single sufficiently large GPU, test full SFT with
`fsdp_devices:1` instead. Memory feasibility must be measured in either case.

- Compile `scripts/check_sft_memory.py --config <cluster-full.yaml>` with the
  intended GPU mask, then execute 20–50 actual optimizer steps in a separate
  smoke-run directory. Save/reload/evaluate its checkpoint and test resume.
- Measure per-GPU peak memory, step time, utilization and host-memory/I/O pressure.
  The abstract memory report is a preflight estimate, not proof that an update fits.
- Launch **one JAX Python process per node** for this single-node path, not
  `torchrun` or four independent training replicas. Scheduler example concept:
  one task with four GPUs; exact `srun`/batch directives depend on the cluster.
- The CLI explicitly assigns `CUDA_VISIBLE_DEVICES` from YAML. Preserve the
  scheduler's allocated device mask rather than accidentally exposing unallocated
  GPUs. The standalone memory/data check scripts import JAX directly and do not
  apply the YAML mask; their process environment must match the training mask.
- Multi-node launch/distributed initialization is not implemented/validated in
  this work. Do not expand to multiple nodes just by increasing `fsdp_devices`.

Gate: real full-SFT updates, native save/reload/resume and inference all succeed
within the allocation. If interconnect/VRAM makes full sharding impractical,
continue LoRA while keeping the full-SFT result explicitly unresolved.

### Phase C — useful BC training and fair comparisons

| Run | Proposed initial recipe | Purpose |
|---|---|---|
| Published pilot | Keep fixed: B4, 1k updates | Historical baseline, not a newly tuned control |
| Full SFT | 4 GPUs/FSDP4; target global B16 after smoke; 5k updates; full model; EMA off | Move toward VLA-Precision's full Stage-I method |
| Extended LoRA control | Same dataset/horizon/global batch/update count/evaluation as the full run | Determine whether full tuning earns its additional cost |
| Follow-up seeds / duration | 3 seeds for the chosen recipe; consider 10k/25k only after 5k learning curves justify it | Quantify variation and whether more training helps |

Start from the existing documented LRs (full 1e-5; LoRA 5e-5), with 100-step
warmup and cosine decay adapted to the new duration. Treat that as a **recipe
comparison**, not an isolated causal comparison of parameter-freezing alone;
include matched-LR controls if making that claim. Do not automatically scale LR
with GPU count. Keep B16 as a target subject to measured memory/throughput.
At B16, 5k steps samples 80,000 chunks (~3.78 passes of 21,141 training windows),
which is substantially more training than the pilot's 4,000 samples.

Use new config/run IDs, evaluate at fixed milestones (e.g. 500/1k/2.5k/5k), and
save enough checkpoints to identify learning regressions. Preserve the standard
native objective for the baseline. Validated launch shape:

```bash
python -m vla_precision.cli train vla --config configs/stage1/pipette_pi05_cluster.yaml
```

That site YAML is to be created on the new cluster; it does not exist in the
current implementation. Do not resume the LoRA training state into full SFT.

Paper alignment: our checked PDF's Table II lists pipette Stage I as **60 demos,
5,000 steps** before ACoB; standalone π0/π0.5 SFT as **120 demos, 25,000 steps**.
Our 37 training episodes, B, horizon and data contract differ, so 5k full SFT is
closer in method/step count, not an exact paper reproduction. The repo's 30k-step
brush example is another task. Table III's **4×A800** hardware entry is explicitly
for online training; BC hardware was not separately established by that table.

### Phase D — make measurements informative

Implement these diagnostics before large sweeps; they are currently missing:

- Log native total flow loss **and** XYZ-only and padding-only components,
  keeping the training objective unchanged initially. Add deterministic
  validation-loss samples to distinguish fitting from generalization.
- Evaluate all 5,521 held-out chunks, with per-episode/per-axis RMSE, MAE and bias,
  first-action and chunk metrics, plus zero-action baselines on identical data.
- Keep the original 144-chunk subset too. The current evaluator draws noise from
  a sequential RNG; changing the sample list changes later noise assignments.
  For multi-GPU evaluation, preassign/store noise per sample or version a stable
  sample-seed mapping, and recompute all compared models under that protocol.
  Do not average per-worker RMSEs; aggregate per-episode squared-error sums/counts.
- Audit motion magnitude and task phases before reporting approach/attachment/
  lift/disposal metrics. Existing phase-related scripts elsewhere are not yet
  validated for this π0.5 split. Choose definitions on training data, then freeze
  them; avoid filtering validation until a desired score appears.
- If padding loss or inactive frames are shown to dominate, run explicit
  XYZ-loss-weighting and/or sampling ablations separately from the native baseline.
  More GPUs permit controlled experiments; they do not prove these changes help.

Gate for proceeding: stable learning curves and useful moving-frame/coarse-motion
predictions; compare to zero action on the predefined metrics and inspect predicted
chunks. Better offline RMSE alone is not a claim of successful tip attachment.

### Phase E — real BC validation, then ACoB

Continue the agreed read-only camera/state check and then a bounded, supervised
BC execution bridge on the robot host. Measure physical success/failure and
contact behavior, not just action RMSE. If approach is coherent but precision or
contact remains difficult, that is a useful point to introduce ACoB; high BC
attachment success is not required first.

Current `build_stage2_initialization_train_config` expects a full-model base.
A freshly trained full-SFT checkpoint is the cleaner starting point, but G1 task,
reward, action transport and HIL integration still need implementation. The
existing Stage-II path must not silently drop this pilot's LoRA adapters. If
retaining LoRA, explicitly support/merge it and verify inference parity first.
Only then allocate GPUs separately for actor inference and learner work, measure
end-to-end latency and preserve operator intervention on the robot host.

## 9. Prompt to resume with on the new cluster

> Read VLA-Precision/docs/PI05_CLUSTER_HANDOFF.md and the matching
> VLAPolicyBridge/docs/PI05_CLUSTER_HANDOFF.md first. Continue the native JAX
> pi05 pipette project from the jy-vla-precision branches. The public pilot is
> GeertWang/pi05-pipette-lora-bc-20260909 at f09e444ffb651fb72bb559b34c57005fe115fd97.
> Audit this cluster's allocation and migrated data, reproduce the pilot, then
> follow the documented full-SFT/FSDP and controlled evaluation plan. Do not assume
> that robot execution, ACoB integration or multi-GPU training already works.
