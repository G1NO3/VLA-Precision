"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat

install_lerobot_import_compat()

import numpy as np
import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import tqdm
from openpi import transforms
from openpi.shared import normalize

from vla_precision.config import ResolvedStage1Config
from vla_precision.data.indexing import materialize_lerobot_indices
from vla_precision.integrations.openpi.configs import build_stage1_train_config
from vla_precision.integrations.openpi.data_loader import (
    TorchDataLoader,
    TransformedDataset,
    _lerobot_dataset,
    create_torch_dataset,
)


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    root_config,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = create_torch_dataset(data_config, action_horizon, model_config, root_config)
    materialize_lerobot_indices(_lerobot_dataset(dataset), root_config.data)
    dataset = TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _compute(config: _config.TrainConfig, root_config, max_frames: int | None = None):
    data_config = config.data.create(config.assets_dirs, config.model)

    # Prepared pipette arrays already contain the final numeric layout. Compute
    # exactly the sampled chunk distribution without decoding unused images.
    from pathlib import Path
    from vla_precision.data.pipette import PipetteDataset
    root = root_config.data.lerobot_root
    if root is not None and (Path(root) / "manifest.json").is_file():
        if root_config.data.state_indices or root_config.data.action_indices:
            raise ValueError("Prepared pipette data already has the final numeric layout")
        dataset = PipetteDataset(root, config.model.action_horizon)
        count = len(dataset) if max_frames is None else min(len(dataset), max_frames)
        if count < 2:
            raise ValueError("Need at least two samples for normalization")
        values = {key: [] for key in ("state", "actions")}
        for i in range(count):
            item = dataset.numeric_item(i)
            for key in values:
                values[key].append(item[key])
        norm_stats = {}
        for key, arrays in values.items():
            stats = normalize.RunningStats()
            stats.update(np.stack(arrays).astype(np.float64))
            norm_stats[key] = stats.get_statistics()
        output_path = config.assets_dirs / data_config.repo_id
        normalize.save(output_path, norm_stats)
        print(f"Wrote train-only stats for {count} complete chunks to {output_path}")
        return

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            root_config,
            max_frames,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


def run_norm_stats(resolved: ResolvedStage1Config, *, max_frames: int | None = None) -> None:
    """Compute Stage-I normalization statistics using the unified profile."""
    _compute(build_stage1_train_config(resolved.config), resolved.config, max_frames)
