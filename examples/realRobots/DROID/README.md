# DROID C42: Qwen3.5-0.8B + QwenOFT

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
