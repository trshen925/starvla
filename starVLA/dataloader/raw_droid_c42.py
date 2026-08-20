"""Direct, zero-copy loader for the C42-filtered raw DROID 1.0.1 subset.

C42 records logical clips and its exact event-balanced training windows in a
JSON mapping, while RGB videos and robot data remain in the original per-
episode DROID layout.  Converting the full set to another video format would
duplicate a large dataset, so this loader reads the mapping and raw files
directly and emits StarVLA's normal ``examples`` interface.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - clear runtime error on minimal installs
    pq = None

try:
    import av
except ImportError:  # OpenCV fallback supports the raw DROID MP4s as well.
    av = None
    import cv2


_OBS_JOINT = "steps/observation/joint_position"
_OBS_GRIP = "steps/observation/gripper_position"
_ACT_JOINT_VEL = "steps/action_dict/joint_velocity"
_ACT_GRIP = "steps/action_dict/gripper_position"
_LANGUAGE = "steps/language_instruction"

# Released pi0.5-DROID normalisation statistics, also used by Gen2Act C39/C42.
_STATE_Q01 = np.asarray([-0.8279732212, -0.8398311848, -0.8425482082, -2.7730152783,
                         -1.8426181348, 1.1716566390, -2.0472648380, 0.0], dtype=np.float32)
_STATE_Q99 = np.asarray([0.8996522881, 1.3854674704, 0.6920277433, -0.4542043057,
                         1.7323142409, 3.4672964780, 2.1984972073, 0.991], dtype=np.float32)
_ACTION_Q01 = np.asarray([-0.4580, -0.8076, -0.4472, -0.9268, -0.6456, -0.6460,
                          -0.7616, 0.0], dtype=np.float32)
_ACTION_Q99 = np.asarray([0.4476, 0.7652, 0.4480, 0.7944, 0.6484, 0.6628,
                          0.7344, 0.9998], dtype=np.float32)


def _decode_column(table, name: str) -> np.ndarray:
    values = table.column(name).to_pylist()
    values = [json.loads(value) if isinstance(value, str) else value for value in values]
    return np.asarray(values, dtype=np.float32)


def _q99_normalize(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    values = 2.0 * (values - q01) / (q99 - q01) - 1.0
    return np.clip(values, -2.2, 2.2).astype(np.float32)


class RawDroidC42Dataset(Dataset):
    """C42 raw DROID samples compatible with ``QwenOFT.forward(examples)``."""

    def __init__(self, data_cfg, mode: str = "train", **_: object) -> None:
        if pq is None:
            raise ImportError("raw_droid_c42 requires pyarrow; install starVLA requirements first")
        self.data_cfg = data_cfg
        self.mode = mode
        self.root = Path(str(data_cfg.root))
        mapping_path = Path(str(data_cfg.clip_mapping))
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        if int(mapping.get("version", 0)) != 1:
            raise ValueError(f"Unsupported C42 mapping version: {mapping.get('version')}")
        self.clips = {str(item["clip_id"]): item for item in mapping["clips"]}
        horizon = int(data_cfg.get("action_horizon", 15))
        stride = int(data_cfg.get("val_stride", 10))
        if mode == "train":
            samples = [(str(clip), int(start)) for clip, start in mapping["train_samples"]]
        elif mode in ("val", "eval"):
            samples = []
            for clip_id, clip in self.clips.items():
                if str(clip["split"]) != "val":
                    continue
                for start in range(0, int(clip["frame_range"][1]) - int(clip["frame_range"][0]) - horizon + 1, stride):
                    samples.append((clip_id, start))
        else:
            raise ValueError(f"Unsupported mode={mode!r}")
        limit = data_cfg.get("max_samples")
        self.samples = samples[: int(limit)] if limit not in (None, "") else samples
        self.horizon = horizon
        self.image_size = tuple(int(value) for value in data_cfg.get("obs_image_size", [224, 224]))
        self.views = list(data_cfg.get("views", ["front", "wrist"]))
        self.cache_size = int(data_cfg.get("parquet_cache_size", 8))
        self._payload_cache: OrderedDict[str, dict[str, np.ndarray | str]] = OrderedDict()
        print(f"[RawDroidC42Dataset] mode={mode} samples={len(self.samples)} clips={len(self.clips)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_payload_cache"] = OrderedDict()
        return state

    def _payload(self, clip_id: str) -> dict[str, np.ndarray | str]:
        if clip_id in self._payload_cache:
            self._payload_cache.move_to_end(clip_id)
            return self._payload_cache[clip_id]
        clip = self.clips[clip_id]
        episode_dir = self.root / str(clip["raw_episode_id"])
        table = pq.read_table(episode_dir / "episode.parquet", columns=[
            _OBS_JOINT, _OBS_GRIP, _ACT_JOINT_VEL, _ACT_GRIP, _LANGUAGE,
        ])
        language_values = table.column(_LANGUAGE).to_pylist()
        language = next((str(value) for value in language_values if value), "Control the robot.")
        payload: dict[str, np.ndarray | str] = {
            "joint": _decode_column(table, _OBS_JOINT),
            "obs_gripper": np.asarray(table.column(_OBS_GRIP).to_pylist(), dtype=np.float32).reshape(-1, 1),
            "velocity": _decode_column(table, _ACT_JOINT_VEL),
            "action_gripper": np.asarray(table.column(_ACT_GRIP).to_pylist(), dtype=np.float32).reshape(-1, 1),
            "language": language,
        }
        self._payload_cache[clip_id] = payload
        if len(self._payload_cache) > self.cache_size:
            self._payload_cache.popitem(last=False)
        return payload

    def _frame(self, video_path: Path, index: int) -> Image.Image:
        if av is None:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"Unable to decode frame {index} from {video_path}")
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
                (self.image_size[1], self.image_size[0])
            )
        # C42 requests one current observation from each stream. Decoding up to
        # the requested frame is slower than a cached reader, but robust across
        # the DROID MP4 encodings and keeps worker state pickle-safe.
        frame_array = None
        with av.open(str(video_path)) as container:
            for frame_idx, frame in enumerate(container.decode(video=0)):
                if frame_idx == int(index):
                    frame_array = frame.to_ndarray(format="rgb24")
                    break
        if frame_array is None:
            raise RuntimeError(f"Unable to decode frame {index} from {video_path}")
        return Image.fromarray(frame_array).resize((self.image_size[1], self.image_size[0]))

    def __getitem__(self, index: int) -> dict:
        clip_id, local_start = self.samples[index]
        clip = self.clips[clip_id]
        absolute_start = int(clip["frame_range"][0]) + int(local_start)
        payload = self._payload(clip_id)
        episode_dir = self.root / str(clip["raw_episode_id"])
        camera = int(clip["camera"])
        video_paths = {
            "front": episode_dir / f"steps_observation_exterior_image_{camera}_left.mp4",
            "wrist": episode_dir / "steps_observation_wrist_image_left.mp4",
        }
        images = [self._frame(video_paths[view], absolute_start) for view in self.views]
        action_end = absolute_start + self.horizon
        velocity = payload["velocity"][absolute_start:action_end]  # type: ignore[index]
        gripper = payload["action_gripper"][absolute_start:action_end]  # type: ignore[index]
        if len(velocity) != self.horizon:
            raise IndexError(f"C42 sample exceeds action range: {clip_id}:{local_start}")
        state = np.concatenate([payload["joint"][absolute_start], payload["obs_gripper"][absolute_start]], axis=0)  # type: ignore[index]
        action = np.concatenate([velocity, gripper], axis=-1)
        return {
            "image": images,
            "lang": payload["language"],
            "state": _q99_normalize(state[None], _STATE_Q01, _STATE_Q99),
            "action": _q99_normalize(action, _ACTION_Q01, _ACTION_Q99),
            "robot_tag": "droid_joint_velocity",
        }


def collate_fn(batch):
    return batch


def get_vla_dataset(data_cfg, mode: str = "train", **kwargs):
    return RawDroidC42Dataset(data_cfg, mode=mode, **kwargs)
