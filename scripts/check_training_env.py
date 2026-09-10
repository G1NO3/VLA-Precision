"""Hardware-free environment smoke check; optionally decode existing dataset videos."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path,
                        help="Directory with camera subdirectories containing episode_000000.mp4")
    args = parser.parse_args()
    for name in ("jax", "jaxlib", "flax", "torch", "torchvision", "torchcodec", "openpi", "lerobot"):
        print(f"{name}: {importlib.metadata.version(name)}", flush=True)

    # Import through the project's entry point first: it installs the pinned
    # OpenPI / LeRobot legacy-module compatibility shim.
    for name in (
        "vla_precision.integrations.openpi.train",
        "vla_precision.acob.agent",
        "vla_precision.acob_stream.learner",
        "vla_precision.acob_stream.actor",
        "vla_precision.data.preprocess",
        "torchcodec",
    ):
        importlib.import_module(name)
        print(f"Import OK: {name}", flush=True)

    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    import optax

    assert jax.default_backend() == "gpu", jax.devices()
    print(f"JAX GPU: {jax.devices()}", flush=True)

    class TinyVision(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Conv(8, (3, 3))(x)
            return nn.Dense(3)(nn.relu(x).mean((1, 2)))

    x = jnp.ones((2, 32, 32, 3), dtype=jnp.bfloat16)
    model = TinyVision()
    params = model.init(jax.random.key(0), x)["params"]
    optimizer = optax.adam(1e-3)
    state = optimizer.init(params)

    @jax.jit
    def step(params, state):
        loss, grad = jax.value_and_grad(
            lambda p: jnp.square(model.apply({"params": p}, x) - 1).mean()
        )(params)
        updates, state = optimizer.update(grad, state, params)
        return optax.apply_updates(params, updates), state, loss

    _, _, loss = step(params, state)
    assert np.isfinite(float(loss.block_until_ready()))
    print(f"JAX convolution / gradient / optimizer OK: loss={float(loss):.6f}", flush=True)

    import torch
    from torchvision.ops import nms

    assert torch.cuda.is_available()
    print(f"PyTorch GPU: {torch.cuda.get_device_name(0)}; CUDA {torch.version.cuda}", flush=True)
    t = torch.ones((64, 64), device="cuda", requires_grad=True)
    value = (t @ t).mean()
    value.backward()
    assert value.item() == 64 and torch.isfinite(t.grad).all()
    conv = torch.nn.Conv2d(3, 8, 3).cuda()
    loss_t = conv(torch.ones((2, 3, 32, 32), device="cuda")).square().mean()
    loss_t.backward()
    assert torch.isfinite(conv.weight.grad).all()
    boxes = torch.tensor([[0., 0., 1., 1.]], device="cuda")
    assert nms(boxes, torch.ones(1, device="cuda"), 0.5).tolist() == [0]
    torch.cuda.synchronize()
    print("PyTorch matrix / convolution / gradients / torchvision CUDA OK", flush=True)

    from vla_precision.config import load_config, load_stage1_config
    root = Path(__file__).resolve().parents[1]
    stage1 = load_stage1_config(root / "configs/stage1/insert_two_bottles_diagonal_rack.yaml")
    stage2 = load_config(root / "configs/stage2/tasks/insert_two_bottles_diagonal_rack.yaml",
                         deployment=root / "configs/stage2/deployments/single_ur.yaml")
    assert stage1.config.openpi.name and stage2.config.openpi.name
    subprocess.run([sys.executable, str(root / "main.py"), "--help"],
                   check=True, stdout=subprocess.DEVNULL)
    print("Stage I / II config parsing and CLI OK (no task launched)", flush=True)

    if args.video_root:
        from lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec
        videos = sorted(args.video_root.glob("*/episode_000000.mp4"))
        assert videos, f"No episode_000000.mp4 videos under {args.video_root}"
        for video in videos:
            frames = decode_video_frames(video, [0., 1 / 30, 2 / 30], 0.02, get_safe_default_codec())
            assert frames.shape[:2] == (3, 3) and torch.isfinite(frames).all()
            assert 0 <= frames.min() <= frames.max() <= 1
            print(f"Video OK: {video.parent.name}: {tuple(frames.shape)}", flush=True)
    print("Environment smoke check passed.", flush=True)


if __name__ == "__main__":
    main()
