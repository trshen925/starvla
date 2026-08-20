"""Read one real C42 sample without downloading or loading a VLM."""
from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from starVLA.dataloader.raw_droid_c42 import RawDroidC42Dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/realRobots/DROID/train_files/starvla_qwen35_08b_oft_c42.yaml",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    dataset = RawDroidC42Dataset(cfg.datasets.vla_data)
    sample = dataset[0]
    assert len(sample["image"]) == 2
    assert sample["state"].shape == (1, 8)
    assert sample["action"].shape == (int(cfg.framework.action_model.action_horizon), 8)
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
