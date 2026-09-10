# HIL194 π0.5 full fine-tuning — 2026-09-11

This run uses native JAX π0.5 with all **3,353,433,872 parameters trainable**,
no LoRA, and no frozen parameters. The previous cluster handoff describes the
older 46-episode LoRA pilot; this document records the new HIL194 experiment.

## Data and configuration

- Source: `jren313/g1-pipette-3view-hil194`, revision
  `643ece1b2e93f15022c83832302848c9978d5f05`; all 788 files verified.
- Initialization: original `gs://openpi-assets/checkpoints/pi05_base`, downloaded
  with per-object generation and checksum receipts. The production run starts
  from this base, independently of the smoke test.
- Prepared cache: `/data/jwang3617/pipette/datasets/g1-pipette-hil194-pi05-30hz-v1`.
  194 episodes, 122,808 transitions at 30 Hz; episode split 155 train / 39 held out,
  producing 94,865 / 26,197 complete action chunks. Norm statistics use train only.
- State: measured canonical 29 body joints plus FK XYZ. Action: consecutive
  reference FK XYZ deltas in metres, pelvis frame, `right_rubber_hand` endpoint.
  H=10, XYZ padded to D=32, native flow loss and discrete state tokenization.
- Three RGB views are decoded and resized using the native padding transform.
  `wrist_left` is the right-wrist view; `wrist_right` is the bracket view.
- Prompt: `Attach a green pipette tip from the tip rack to the pipette.`
- All source action labels are retained for this baseline. The feasibility audit
  records 24 transitions over 3 mm, including 9 over 10 mm; some reference jumps
  have tiny measured motion. Episode-constant action timestamps cannot establish
  label alignment. Grip targets are not used by this XYZ policy.
- Config: [pipette_pi05_hil194_full.yaml](../configs/stage1/pipette_pi05_hil194_full.yaml).
  Physical GPU 5 (B200), batch 16, 5,000 updates, seed 42, Adam, no EMA,
  100-step warmup to 1e-5 then cosine decay to 1e-6. FSDP device count is 1.
  5,000 updates correspond to about 0.843 passes over training chunks.

The paper's Stage-I full-parameter BC motivates this experiment. This is not an
exact replication of its dataset/training schedule, nor its Stage-II ACoB run.

## Preflight and runtime records

Production training launched at **2026-09-11 02:28 Asia/Taipei** with supervisor
PID `3110228` and initial training PID `3110273`. GPU 5 had acquired other small
jobs by launch time, so it is shared; observed free memory after initialization
was about 99 GiB. No other user's process was stopped or modified.

Local record directory:
`/home/jwang3617/pipette/outputs/pi05-hil194-full-20260911`.
`status.json` is the live authority for production-run state; `train.log` contains
training metrics, and `gpu-usage.csv` records GPU 5 usage every ten seconds.

The full-cache check passed: split isolation, target padding, inverse
normalization, 148/200 maximum prompt tokens, and 36 exact source-video frame
comparisons spanning both recording batches. GPU compilation estimated 54.31 GiB.
The actual batch-16 test completed 20 updates, saved a full checkpoint, restored
model and optimizer state, and continued through update 25. Loss and gradients
were finite. Sampled whole-device GPU-memory peak was 81,708 MiB (79.79 GiB);
the monitor reports the entire card, including any other processes.

Checkpoint inference also passed on all 39 held-out episodes (one chunk each).
This smoke subset is a pipeline check, not a policy-quality benchmark. Production
evaluation and original-base evaluation use the same 16 uniformly spaced chunks
per held-out episode (624 total), seed 42, and 10 flow steps. Results are
`validation.json` and `base-validation.json`, with per-sample prediction sidecars.
These are open-loop action errors, not robot attachment success rates.

Original-base results on the fixed 624-chunk validation subset: chunk RMSE
0.516900 mm, first-action RMSE 0.478159 mm, zero-action chunk RMSE 0.307753 mm.
Production training and evaluation completed after 5,000 updates. Final chunk
XYZ component RMSE: 0.301030 mm (zero action: 0.307753 mm). Axis RMSE X/Y/Z:
0.402052 / 0.226840 / 0.242395 mm; X/Y do not consistently beat zero action.
See `outputs/pi05-hil194-full-20260911/AXIS_ERROR_REPORT_ZH.md`.

`run-provenance.json`, `training-config.yaml`, `norm_stats.json`, source snapshots,
and `working-tree.patch` preserve the input/configuration fingerprints.
`preflight-complete.json` is only created after the sequential preflight succeeds.

## Storage and operation

All model/data/cache/checkpoint storage uses `/data/jwang3617/pipette` on the
large NVMe data disk. Do not use the nearly full workspace/root filesystem for
large assets. A full checkpoint occupies approximately 31 GiB compressed on disk;
the first save took several minutes, while subsequent saves were faster.

Checkpoint root:
`/data/jwang3617/pipette/checkpoints/vla-precision-stage1/pi05_full_finetune_pipette/pipette_hil194_full_20260911`.
Save interval is 500; keep period 1,000 preserves milestone checkpoints plus the
latest. The final checkpoint directory is `4999` (5,000 completed updates).
The supervisor automatically evaluates it after training finishes.

Activate the environment with:

```bash
cd /home/jwang3617/pipette
source outputs/pi05-hil194-feasibility/activate_training_env.sh
```

The production supervisor is `outputs/pi05-hil194-full-20260911/run_training.py`.
Do not launch it a second time while its recorded PID is alive. To stop this run,
send SIGTERM to its `supervisor_pid` from `status.json`; it terminates its own
child process group. For later recovery, first inspect the last complete numeric
checkpoint directory, then use the same experiment with `resume: true`; never
set `overwrite: true` to resume. Native resume restores optimizer/model/step, but
does not preserve the exact data-loader iterator position.

No robot execution or GUI deployment is included in this training run.

## Uploaded full checkpoint

Private Hugging Face model: [GeertWang/pi05-pipette-hil194-full-bc-20260911](https://huggingface.co/GeertWang/pi05-pipette-hil194-full-bc-20260911).
Pin revision `641412e9434ab36341bc5469540e5c4463d74ce5`; use `checkpoints/4999` inside the snapshot.
The upload contains complete inference parameters, Adam state, normalization,
tokenizer, configuration, code snapshot, evaluation reports and the loss curve.
All 187 uploaded files (32302254712 bytes) were verified against
remote sizes and LFS SHA256 or Git blob SHA1. Authenticated download was checked.
The packaged loader reproduced the stored validation prediction exactly.
Receipt: `outputs/pi05-hil194-full-20260911/hf-upload-receipt.json`.

## VLAPolicyBridge GUI integration

`VLAPolicyBridge/gui/pi05_config.json` selects this checkpoint and the `/data`
training environment, GPU 5. The preview worker now accepts both full and LoRA
pipette π0.5 configurations. `http://127.0.0.1:8080/pi05` is the loopback GUI;
use SSH port forwarding from a remote workstation. Runtime records are in
`outputs/pi05-hil194-full-20260911/gui-server.json` and `gui-check.json`.
The 194-episode index, exact validation-prediction parity and synthetic live HIL
GET-only inference passed. The GUI now offers both live preview and native 30-Hz action publication to the
existing HIL actor. A dedicated asynchronous worker consumes one native increment
per actor request; no temporal summation is applied. Actor defaults preserve
physical speed bounds at 30 Hz. Isolated testing measured 30.01 Hz with exact
proposal/sequence checks and stop cleanup; no real robot was connected. See the
bridge GUI README and `gui-execution-check.json` for operation and evidence.
