# DROID recipes

## Raw DROID in original OXE-DROID semantics: VM4A Diffusion Policy

`starvla_dp_oxe_raw_smoke.yaml` uses StarVLA's original VM4A
`DiffusionPolicy` unchanged: independent ImageNet-pretrained ResNet-18
encoders, conditional 1-D U-Net, DDPM scheduler, and EMA checkpointing. It
does **not** use a Qwen backbone or the C42 data representation.

The direct raw-data adapter preserves the original `oxe_droid` contract:

- three RGB streams: exterior image 1, exterior image 2, and wrist image;
- 10-D state: absolute EEF XYZ, 6-D EEF rotation, and gripper;
- 7-D action: delta EEF XYZ, relative axis-angle, and binary gripper; and
- a native 16-step action chunk.

It reads `/mnt/pfs/data/fenghaoran/droid/decompressed/1.0.1` directly and
does not read a C42 mapping or C42 normalisation statistics. Start the smoke
run from the repository root with:

```bash
bash examples/realRobots/DROID/train_files/run_dp_oxe_raw_smoke.sh
```

For a full run, set `datasets.vla_data.max_samples: null` and choose an
appropriate training schedule. The raw loader is intentionally separate from
the legacy C42 loader so the two experiments cannot share data semantics.
Its state min/max cache is also keyed by the selected episode set, so a smoke
run cannot supply normalization statistics to a later full-DROID run.

For throughput, the loader retains a bounded, worker-local LRU of ffmpeg
readers and resized contiguous RGB blocks. These are only I/O optimisations
adopted from the C42 loader; they do not change OXE window selection, state,
or action construction. Tune `num_workers`, `video_reader_cache`,
`decoded_frame_cache_mb`, and `decoded_frame_cache_block_frames` to available
CPU/RAM. Keep `ffmpeg_threads: 1` when many workers/ranks are active.

## C42: Qwen3.5-0.8B + QwenOFT

This recipe reads the C42-filtered DROID 1.0.1 subset directly. It does not
copy or convert the raw videos. The mapping preserves C39's 565,946
event-balanced train windows and its 500 logical validation clips.

For a first smoke test, download (or point `BASE_VLM` at a local snapshot of)
`Qwen/Qwen3.5-0.8B`, then run:

```bash
python examples/realRobots/DROID/train_files/smoke_raw_droid_c42.py
bash examples/realRobots/DROID/train_files/run_qwen35_08b_oft_c42.sh
```

The checked-in config deliberately uses `max_samples: 128`, 20 steps, and a
frozen VLM. After it succeeds, set `max_samples: null`, raise
`max_train_steps`, and remove `qwen_vl_interface` from `freeze_modules` for
end-to-end fine-tuning. Actions are 15 future 15-Hz commands of
`[7 joint velocities, gripper position]`, q01/q99-normalized with the same
published pi0.5-DROID statistics used by C39/C42.
