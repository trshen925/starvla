# BottleOnCoaster50：MiniCPM-GR00T 纯 VLA 训练与评测复现

本实验在 [`trshen925/starvla`](https://github.com/trshen925/starvla) 的 `main` 分支上完成；记录时的提交为 `a9d3b0b`（`modify droid`）。

任务是根据一条语言指令、当前固定相机图像、当前腕部相机图像和当前机器人状态，预测未来一段关节动作。本实验是纯 VLA，不使用视频片段、历史图像或视频抽帧。

## 实验状态

- 配置：`starVLA/examples/realRobots/DROID/train_files/starvla_minicpm_gr00t_bottle_on_coaster_30k.yaml`
- 目标训练步数：30,000
- 已保存并可用的模型：25,000 step
  - `starVLA/results/Checkpoints/minicpm_gr00t_bottle_on_coaster_30k/checkpoints/steps_25000_pytorch_model.pt`
- 训练进程在 25k 之后被暂停，因而目前**没有** `steps_30000_pytorch_model.pt` 或 `final_model/pytorch_model.pt`。下文所有评测默认使用 25k checkpoint。

## 数据集与样本定义

从 Hugging Face 下载的数据集为 [`trshen925/BottleOnCoaster50`](https://huggingface.co/datasets/trshen925/BottleOnCoaster50)，本地训练根目录：

```text
data/BottleOnCoaster50/train
```

训练集包含 45 条 HDF5 轨迹，每条 211 帧；以动作窗口生成 8,865 个训练样本。每个样本包含：

| 字段 | 内容 |
| --- | --- |
| `lang` | `Pick up the yellow bottle and place it on the white coaster.` |
| `image[0]` | `over_shoulder_left_camera`，固定视角 RGB 图像 |
| `image[1]` | `wrist_cam`，腕部视角 RGB 图像 |
| `state` | 当前 `[arm_joint_pos(6), gripper_pos(1)]`，shape 为 `(1, 7)` |
| `action` | 当前时刻起未来 15 步的 `[6 joint targets, gripper target]`，shape 为 `(15, 7)` |

图像被缩放到 448 × 448。状态和动作按配置中的 `q01/q99` 线性归一化到 `[-1, 1]`。模型不接收 joint velocity、gripper velocity、历史帧或未来图像。

## 前置条件

```bash
git clone https://github.com/trshen925/starvla.git starvla_droid
cd starvla_droid
```

准备以下本地资源，并按需修改 YAML 中的绝对路径：

1. MiniCPM-V-4.6 基座模型，默认路径为 `starVLA/playground/Pretrained_models/MiniCPM-V-4.6`。
2. BottleOnCoaster50 数据集，默认路径为 `../data/BottleOnCoaster50/train`（相对仓库目录）。
3. 用作初始化的 4-task checkpoint。其路径由 `trainer.pretrained_checkpoint` 指定；如果没有该模型，可将其替换为另一个兼容的 MiniCPM-GR00T checkpoint，或删除该项以从随机 action head 开始训练。

本实验使用仓库中新增的 HDF5 数据加载器：

```text
starVLA/dataloader/hdf5_mixed_posttrain.py
```

它依赖 `h5py`。训练环境还需要项目常规依赖（PyTorch、Transformers、Accelerate、OmegaConf、Pillow、NumPy 等）。

## 训练

从仓库根目录启动 8 卡训练：

```bash
export PYTHONPATH="$PWD"
export WANDB_MODE=disabled

torchrun --standalone --nproc_per_node=8 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/examples/realRobots/DROID/train_files/starvla_minicpm_gr00t_bottle_on_coaster_30k.yaml
```

关键超参数：

- `MiniCPMGR00T`，MiniCPM-V 4.6 + GR00T DiT-B action head
- action/state dimension：7
- action horizon：15
- 每卡 batch size：8（全局 batch size 64）
- 最大步数：30,000
- 保存间隔：5,000 step，保留最近两个 checkpoint
- 动作头学习率：`5e-5`；VLM 学习率：`5e-7`
- 推理扩散步数：4

### 从中断处恢复

将 YAML 中设置为：

```yaml
trainer:
  is_resume: true
```

训练器会从 `run_root_dir/run_id/checkpoints/` 自动选取最新的 `steps_*_pytorch_model.pt`。对于本实验，最新完整 checkpoint 是 25k；恢复命令与上节相同。

## 单样本动作评测

下面的脚本加载 25k checkpoint，从训练集读取一个真实样本，并仅执行动作预测。该过程不会启动训练。

```bash
export PYTHONPATH="$PWD"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0

python - <<'PY'
from omegaconf import OmegaConf
from starVLA.dataloader.hdf5_mixed_posttrain import build_dataset
from starVLA.model.framework.share_tools import apply_config_compat
import starVLA.model.framework.base_framework as base_framework
from starVLA.model.framework.VLM4A.MiniCPMGR00T import MiniCPM_GR00T  # registers framework
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

cfg = apply_config_compat(OmegaConf.load(
    "starVLA/examples/realRobots/DROID/train_files/"
    "starvla_minicpm_gr00t_bottle_on_coaster_30k.yaml"
))
# Avoid importing unrelated optional framework plugins during standalone eval.
base_framework._FRAMEWORKS_IMPORTED = True

checkpoint = (
    "starVLA/results/Checkpoints/minicpm_gr00t_bottle_on_coaster_30k/"
    "checkpoints/steps_25000_pytorch_model.pt"
)
model = build_framework(cfg)
TrainerUtils.load_pretrained_backbones(model, checkpoint)
model = model.cuda().eval()

dataset = build_dataset(cfg.datasets.vla_data, mode="train")
sample = dataset[0]
sample.pop("action")  # predict_action only needs images, language, and state
prediction = model.predict_action([sample])
print(prediction["normalized_actions"].shape)  # (1, 15, 7)
PY
```

输出是归一化动作。要映射回机器人原始动作空间，对每一个维度执行：

```text
action = (normalized_action + 1) / 2 * (action_q99 - action_q01) + action_q01
```

`action_q01` 和 `action_q99` 必须使用同一 YAML 的 `datasets.vla_data` 中的数值。

## 已测显存

在 A800 上，以单卡、batch=1、两张 448 × 448 图像、语言、7 维状态和 15 步动作预测实测：

- 加载模型后：约 2.93 GiB
- 第一次推理峰值：约 3.26 GiB
- 后续稳态推理：约 2.94 GiB
- 后续单样本推理时间：约 0.33 秒

因此单样本在线评测建议至少预留 4 GiB 显存；若要并发、增大 batch 或保留额外视觉缓存，应相应增加余量。

## 复现注意事项

- 数据集相机顺序不可交换：固定相机在前，腕部相机在后。
- 状态、动作的维度和归一化统计必须与 checkpoint 一致；本实验是 7 维而非其他 DROID 配置中常见的 8 维。
- checkpoint 加载会跳过形状不兼容的 action/state 输出参数，以支持从旧 8 维任务初始化到当前 7 维任务；评测当前 7 维 25k checkpoint 时，所有 972 个 checkpoint tensor 均能加载。
- 文档记录的是本实验环境的本地路径；将项目迁移到其他机器时，优先修改 YAML 的 `base_vlm`、`roots`、`run_root_dir` 和 `pretrained_checkpoint`。
