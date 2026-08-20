"""Read one real C42 sample without downloading or loading a VLM."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

from starVLA.dataloader.raw_droid_c42 import RawDroidC42Dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/realRobots/DROID/train_files/starvla_qwen35_08b_oft_c42.yaml",
    )
    args = parser.parse_args()
    if OmegaConf is not None:
        cfg = OmegaConf.load(args.config)
        data_cfg = cfg.datasets.vla_data
        horizon = int(cfg.framework.action_model.action_horizon)
    else:
        # The fallback deliberately has no YAML dependency, allowing a raw-data
        # smoke test in a minimal DROID preprocessing environment.
        data_cfg = {
            "root": "/mnt/pfs/data/fenghaoran/droid/decompressed/1.0.1",
            "clip_mapping": "/mnt/pfs/users/shentingrui/code/robo/video_gen/gen2act/gen2act/metadata/c42_c39_to_raw_droid_mapping.json",
            "action_horizon": 15,
            "views": ["front", "wrist"],
            "obs_image_size": [224, 224],
            "parquet_cache_size": 1,
            "max_samples": 1,
        }
        horizon = 15
    dataset = RawDroidC42Dataset(data_cfg)
    sample = dataset[0]
    assert len(sample["image"]) == 2
    assert sample["state"].shape == (1, 8)
    assert sample["action"].shape == (horizon, 8)
    print(
        "OK",
        f"samples={len(dataset)}",
        f"images={[image.size for image in sample['image']]}",
        f"state={sample['state'].shape}",
        f"action={sample['action'].shape}",
        f"instruction={sample['lang']!r}",
    )


if __name__ == "__main__":
    main()
