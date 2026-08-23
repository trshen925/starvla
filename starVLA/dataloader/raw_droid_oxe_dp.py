"""Raw DROID 1.0.1 adapter for StarVLA's unmodified VM4A Diffusion Policy.

This loader deliberately mirrors the *semantic* OXE-DROID contract instead of
the C42 joint-velocity contract:

* RGB: exterior camera 1, exterior camera 2, and wrist camera;
* state: absolute end-effector position, 6D rotation, and gripper (10-D);
* action: end-effector translation delta, relative axis-angle rotation, and
  binary gripper target (7-D); and
* a 16-step action window beginning at the observation timestep.

It reads the original decompressed DROID parquet/MP4 layout directly.  An
optional C42/C39 clip mapping can restrict it to C39's validated, event-
balanced sample index while retaining this loader's OXE EEF-delta action
representation.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, Sampler

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - produces a useful runtime error
    pq = None

_OBS_CART = "steps/observation/cartesian_position"
_OBS_GRIP = "steps/observation/gripper_position"
_ACT_CART = "steps/action_dict/cartesian_position"
_ACT_GRIP = "steps/action_dict/gripper_position"
_LANGUAGE = "steps/language_instruction"
_CAMERAS = ("exterior_image_1", "exterior_image_2", "wrist_image")


def _decode_column(table, name: str) -> np.ndarray:
    values = table.column(name).to_pylist()
    values = [json.loads(value) if isinstance(value, str) else value for value in values]
    return np.asarray(values, dtype=np.float32)


def _read_scalar_column(table, name: str) -> np.ndarray:
    return np.asarray(table.column(name).to_pylist(), dtype=np.float32).reshape(-1, 1)


class RawDroidOxeDPDataset(Dataset):
    """Original raw DROID represented exactly as OXE-DROID VM4A samples."""

    def __init__(self, data_cfg, mode: str = "train", **_: object) -> None:
        if pq is None:
            raise ImportError("raw_droid_oxe_dp requires pyarrow; install StarVLA requirements")
        if mode != "train":
            raise ValueError("raw_droid_oxe_dp currently supplies the training split only")

        self.data_cfg = data_cfg
        self.root = Path(str(data_cfg.root))
        self.horizon = int(data_cfg.get("action_horizon", 16))
        if self.horizon != 16:
            raise ValueError(
                "raw_droid_oxe_dp is intentionally pinned to the native OXE-DROID 16-step horizon"
            )
        self.image_size = tuple(int(value) for value in data_cfg.get("obs_image_size", [224, 224]))
        self.cache_size = int(data_cfg.get("parquet_cache_size", 8))
        self.frame_cache_mb = float(data_cfg.get("decoded_frame_cache_mb", 128))
        self.frame_cache_block_frames = int(data_cfg.get("decoded_frame_cache_block_frames", 32))
        self.seed = int(data_cfg.get("seed", 42))

        # C42 stores logical clips as half-open ranges in raw episodes, plus an
        # exact event-balanced ``train_samples`` list of (clip_id, offset).
        # Convert each selected logical offset to the raw episode timestep here
        # so __getitem__ remains the native OXE-DROID implementation below.
        mapping_path = data_cfg.get("clip_mapping")
        mapped_samples: list[tuple[str, int]] | None = None
        mapped_episode_ids: set[str] | None = None
        if mapping_path not in (None, ""):
            mapping_file = Path(str(mapping_path))
            try:
                mapping_payload = json.loads(mapping_file.read_text(encoding="utf-8"))
                clips = {
                    str(clip["clip_id"]): clip
                    for clip in mapping_payload["clips"]
                    if str(clip.get("split", "train")) == "train"
                }
                mapped_samples = []
                for clip_id, offset in mapping_payload["train_samples"]:
                    clip = clips.get(str(clip_id))
                    if clip is None:
                        continue
                    frame_start, frame_stop = (int(value) for value in clip["frame_range"])
                    offset = int(offset)
                    # A DP label includes actions [start, start + horizon), so
                    # never permit a sample to cross the validated clip end.
                    if offset < 0 or frame_start + offset + self.horizon > frame_stop:
                        continue
                    mapped_samples.append((str(clip["raw_episode_id"]), frame_start + offset))
                if not mapped_samples:
                    raise ValueError("no valid 16-step OXE samples in clip_mapping.train_samples")
                # Apply a smoke cap before collecting episode metadata/state
                # statistics; otherwise a tiny mapped smoke would still scan
                # every raw episode referenced by the complete C42 index.
                max_samples = data_cfg.get("max_samples")
                if max_samples not in (None, ""):
                    mapped_samples = mapped_samples[: int(max_samples)]
                mapped_episode_ids = {episode_id for episode_id, _ in mapped_samples}
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Unable to load raw DROID clip_mapping {mapping_file}") from exc

        requested_episode_ids = data_cfg.get("episode_ids")
        if mapped_episode_ids is not None:
            episode_dirs = [self.root / episode_id for episode_id in sorted(mapped_episode_ids)]
        elif requested_episode_ids not in (None, ""):
            episode_dirs = [self.root / str(episode_id) for episode_id in requested_episode_ids]
        else:
            episode_glob = str(data_cfg.get("episode_glob", "episode_*"))
            episode_dirs = sorted(
                path for path in self.root.glob(episode_glob)
                if path.is_dir() and (path / "episode.parquet").is_file()
            )
        max_episodes = data_cfg.get("max_episodes")
        if max_episodes not in (None, ""):
            episode_dirs = episode_dirs[: int(max_episodes)]
        if not episode_dirs:
            raise FileNotFoundError(f"No raw DROID episodes found in {self.root}")

        self.episodes: list[tuple[str, int]] = []
        for episode_dir in episode_dirs:
            required = [
                episode_dir / "steps_observation_exterior_image_1_left.mp4",
                episode_dir / "steps_observation_exterior_image_2_left.mp4",
                episode_dir / "steps_observation_wrist_image_left.mp4",
            ]
            if not all(path.is_file() and path.stat().st_size > 0 for path in required):
                continue
            num_rows = int(pq.ParquetFile(episode_dir / "episode.parquet").metadata.num_rows)
            if num_rows >= self.horizon:
                self.episodes.append((episode_dir.name, num_rows))
        if not self.episodes:
            raise RuntimeError("No complete raw DROID episodes have 16 actions and all three camera streams")

        # State bounds must be scoped to precisely the selected episodes.  In
        # particular, never let a one-episode smoke run leave statistics that
        # a later full-DROID run accidentally reuses.
        selection = json.dumps(self.episodes, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        selection_hash = hashlib.sha256(selection).hexdigest()[:16]
        default_stats_path = self.root / f".starvla_oxe_dp_state_stats_{selection_hash}.json"
        self.stats_path = Path(str(data_cfg.get("state_stats_path", default_stats_path)))

        (
            self._state_position_min,
            self._state_position_max,
            self._state_gripper_min,
            self._state_gripper_max,
        ) = self._load_or_compute_state_bounds()

        if mapped_samples is None:
            # One entry per valid immediate action window for ordinary raw
            # DROID training.
            self.samples = [
                (episode_id, start)
                for episode_id, num_steps in self.episodes
                for start in range(num_steps - self.horizon + 1)
            ]
        else:
            available_episodes = {episode_id for episode_id, _ in self.episodes}
            self.samples = [
                (episode_id, start)
                for episode_id, start in mapped_samples
                if episode_id in available_episodes
            ]
        max_samples = data_cfg.get("max_samples")
        if max_samples not in (None, ""):
            self.samples = self.samples[: int(max_samples)]

        self._payload_cache: OrderedDict[str, dict[str, np.ndarray | str]] = OrderedDict()
        self._frame_cache: OrderedDict[tuple[str, int], tuple[np.ndarray, int]] = OrderedDict()
        self._frame_cache_bytes = 0
        # Keep a small bounded set of ffmpeg readers alive per worker.  This
        # is a throughput optimisation borrowed from the C42 loader only: it
        # does not affect OXE sampling or action/state semantics.
        self._video_cache: OrderedDict[Path, Any] = OrderedDict()
        print(
            f"[RawDroidOxeDPDataset] episodes={len(self.episodes)} samples={len(self.samples)} "
            f"horizon={self.horizon} root={self.root}"
        )

    def _load_or_compute_state_bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load the OXE min/max state statistics, or compute/cache them once.

        The original OXE transform min-max normalizes absolute EEF position and
        gripper state, while leaving the 6D rotation untouched. Raw DROID does
        not ship the converted LeRobot statistics file, so we reproduce those
        exact semantics by computing bounds over the selected raw episodes.
        """
        configured = self.data_cfg.get("state_bounds")
        if configured not in (None, ""):
            try:
                position_min = np.asarray(configured["position_min"], dtype=np.float32)
                position_max = np.asarray(configured["position_max"], dtype=np.float32)
                gripper_min = np.asarray(configured.get("gripper_min", [0.0]), dtype=np.float32)
                gripper_max = np.asarray(configured.get("gripper_max", [1.0]), dtype=np.float32)
                if position_min.shape == (3,) and position_max.shape == (3,) and gripper_min.shape == (1,) and gripper_max.shape == (1,):
                    return position_min, position_max, gripper_min, gripper_max
            except (KeyError, TypeError, ValueError):
                pass
            raise ValueError("state_bounds must provide 3-D position_min/position_max and optional scalar gripper bounds")
        if self.stats_path.is_file():
            try:
                payload = json.loads(self.stats_path.read_text(encoding="utf-8"))
                return tuple(
                    np.asarray(payload[key], dtype=np.float32)
                    for key in ("position_min", "position_max", "gripper_min", "gripper_max")
                )
            except (OSError, KeyError, TypeError, ValueError):
                pass
        position_min = np.full(3, np.inf, dtype=np.float64)
        position_max = np.full(3, -np.inf, dtype=np.float64)
        gripper_min = np.full(1, np.inf, dtype=np.float64)
        gripper_max = np.full(1, -np.inf, dtype=np.float64)
        for episode_id, _ in self.episodes:
            table = pq.read_table(self.root / episode_id / "episode.parquet", columns=[_OBS_CART, _OBS_GRIP])
            values = _decode_column(table, _OBS_CART)
            position_min = np.minimum(position_min, np.nanmin(values[:, :3], axis=0))
            position_max = np.maximum(position_max, np.nanmax(values[:, :3], axis=0))
            gripper = _read_scalar_column(table, _OBS_GRIP)
            gripper_min = np.minimum(gripper_min, np.nanmin(gripper, axis=0))
            gripper_max = np.maximum(gripper_max, np.nanmax(gripper, axis=0))
        if not all(np.all(np.isfinite(value)) for value in (position_min, position_max, gripper_min, gripper_max)):
            raise RuntimeError("Unable to compute finite raw DROID EEF position statistics")
        try:
            self.stats_path.write_text(
                json.dumps(
                    {
                        "position_min": position_min.tolist(),
                        "position_max": position_max.tolist(),
                        "gripper_min": gripper_min.tolist(),
                        "gripper_max": gripper_max.tolist(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
        return (
            position_min.astype(np.float32),
            position_max.astype(np.float32),
            gripper_min.astype(np.float32),
            gripper_max.astype(np.float32),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self):
        # imageio readers own ffmpeg subprocesses and cannot be safely passed
        # to spawned DataLoader workers.
        self._close_video_cache()
        state = self.__dict__.copy()
        state["_payload_cache"] = OrderedDict()
        state["_frame_cache"] = OrderedDict()
        state["_frame_cache_bytes"] = 0
        state["_video_cache"] = OrderedDict()
        return state

    def __del__(self):  # pragma: no cover - interpreter shutdown is timing-dependent
        try:
            self._close_video_cache()
        except Exception:
            pass

    def _payload(self, episode_id: str) -> dict[str, np.ndarray | str]:
        cached = self._payload_cache.get(episode_id)
        if cached is not None:
            self._payload_cache.move_to_end(episode_id)
            return cached
        table = pq.read_table(
            self.root / episode_id / "episode.parquet",
            columns=[_OBS_CART, _OBS_GRIP, _ACT_CART, _ACT_GRIP, _LANGUAGE],
        )
        language_values = table.column(_LANGUAGE).to_pylist()
        language = next((str(value) for value in language_values if value), "Control the robot.")
        payload: dict[str, np.ndarray | str] = {
            "observation_cartesian": _decode_column(table, _OBS_CART),
            "observation_gripper": _read_scalar_column(table, _OBS_GRIP),
            "action_cartesian": _decode_column(table, _ACT_CART),
            "action_gripper": _read_scalar_column(table, _ACT_GRIP),
            "language": language,
        }
        self._payload_cache[episode_id] = payload
        if len(self._payload_cache) > self.cache_size:
            self._payload_cache.popitem(last=False)
        return payload

    def _reader(self, video_path: Path):
        """Return a worker-local persistent ffmpeg reader for ``video_path``."""
        video_path = Path(video_path)
        reader = self._video_cache.get(video_path)
        if reader is not None:
            self._video_cache.move_to_end(video_path)
            return reader

        max_cache = int(self.data_cfg.get("video_reader_cache", 8))
        if max_cache > 0 and len(self._video_cache) >= max_cache:
            self._close_reader(next(iter(self._video_cache)))
        threads = str(max(1, int(self.data_cfg.get("ffmpeg_threads", 1))))
        retries = max(0, int(self.data_cfg.get("video_reader_retries", 3)))
        retry_delay = max(0.0, float(self.data_cfg.get("video_reader_retry_delay", 0.25)))
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                reader = imageio.get_reader(
                    str(video_path),
                    format="ffmpeg",
                    input_params=["-threads", threads],
                    output_params=["-threads", threads],
                )
                self._video_cache[video_path] = reader
                return reader
            except Exception as exc:
                last_error = exc
                # Under many DataLoader workers a failed ffmpeg spawn is often
                # temporary resource pressure; release this worker's old
                # readers before retrying.
                self._close_video_cache()
                if attempt < retries and retry_delay:
                    time.sleep(retry_delay * (2 ** attempt))
        raise OSError(f"Could not open raw DROID video after {retries + 1} attempts: {video_path}") from last_error

    def _close_reader(self, video_path: Path) -> None:
        reader = self._video_cache.pop(Path(video_path), None)
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass

    def _close_video_cache(self) -> None:
        for video_path in list(self._video_cache):
            self._close_reader(video_path)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(frame).resize((self.image_size[1], self.image_size[0])))

    def _cached_frame_block(self, video_path: Path, index: int) -> np.ndarray | None:
        budget = int(max(self.frame_cache_mb, 0.0) * 1024 * 1024)
        block_size = self.frame_cache_block_frames
        if budget <= 0 or block_size <= 0:
            return None
        block_id = int(index) // block_size
        key = (str(video_path), block_id)
        entry = self._frame_cache.get(key)
        if entry is None:
            start, stop = block_id * block_size, (block_id + 1) * block_size
            frames: list[np.ndarray] = []
            reader = self._reader(video_path)
            try:
                # imageio's ffmpeg reader seeks directly to the requested
                # frame; caching this contiguous range makes adjacent native
                # OXE windows cheap while avoiding decode-from-zero per block.
                for frame_index in range(start, stop):
                    frames.append(self._resize_frame(reader.get_data(frame_index)))
            except Exception:
                # Do not retain a reader whose underlying ffmpeg process died.
                self._close_reader(video_path)
                if not frames:
                    return None
            if not frames:
                return None
            array = np.stack(frames, axis=0)
            size = int(array.nbytes)
            if size > budget:
                return None
            while self._frame_cache and self._frame_cache_bytes + size > budget:
                _, (_, evicted_size) = self._frame_cache.popitem(last=False)
                self._frame_cache_bytes -= evicted_size
            self._frame_cache[key] = (array, size)
            self._frame_cache_bytes += size
            entry = (array, size)
        else:
            self._frame_cache.move_to_end(key)
        offset = int(index) - block_id * block_size
        return entry[0][offset] if 0 <= offset < len(entry[0]) else None

    def _frame(self, video_path: Path, index: int) -> Image.Image:
        cached = self._cached_frame_block(video_path, index)
        if cached is not None:
            return Image.fromarray(cached)
        try:
            return Image.fromarray(self._resize_frame(self._reader(video_path).get_data(int(index))))
        except Exception as exc:
            self._close_reader(video_path)
            raise RuntimeError(f"Unable to decode frame {index} from {video_path}") from exc

    def _state(self, observation_cartesian: np.ndarray, observation_gripper: np.ndarray) -> np.ndarray:
        # This is the OXE-DROID state transform: absolute XYZ, absolute rotation
        # in Zhou 6D form, and gripper. Dataset min/max normalization is kept in
        # the loader so VM4A's internal normalizer can remain identity.
        position = observation_cartesian[:3]
        position_range = np.maximum(self._state_position_max - self._state_position_min, 1e-6)
        position = 2.0 * (position - self._state_position_min) / position_range - 1.0
        # pytorch3d.transforms.matrix_to_rotation_6d, used by the original
        # OXE transform, flattens the first two matrix rows.
        rotation_6d = Rotation.from_euler("xyz", observation_cartesian[3:6]).as_matrix()[:2, :].reshape(-1)
        gripper_range = np.maximum(self._state_gripper_max - self._state_gripper_min, 1e-6)
        gripper = 2.0 * (observation_gripper[:1] - self._state_gripper_min) / gripper_range - 1.0
        return np.concatenate((position, rotation_6d, gripper)).astype(np.float32)

    @staticmethod
    def _actions(action_cartesian: np.ndarray, action_gripper: np.ndarray, current_observation: np.ndarray) -> np.ndarray:
        # Original OXE-DROID action semantics: target EEF pose -> delta from the
        # current observed EEF pose. Translation stays in base coordinates and
        # rotation is the relative axis-angle, matching eef_*_delta fields.
        current_rotation = Rotation.from_euler("xyz", current_observation[3:6])
        target_rotation = Rotation.from_euler("xyz", action_cartesian[:, 3:6])
        delta_rotation = (target_rotation * current_rotation.inv()).as_rotvec().astype(np.float32)
        delta_position = action_cartesian[:, :3] - current_observation[None, :3]
        gripper = (action_gripper[:, :1] > 0.5).astype(np.float32)
        return np.concatenate((delta_position, delta_rotation, gripper), axis=-1).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_id, start = self.samples[index]
        payload = self._payload(episode_id)
        observation_cartesian = payload["observation_cartesian"]  # type: ignore[assignment]
        observation_gripper = payload["observation_gripper"]  # type: ignore[assignment]
        action_cartesian = payload["action_cartesian"]  # type: ignore[assignment]
        action_gripper = payload["action_gripper"]  # type: ignore[assignment]
        assert isinstance(observation_cartesian, np.ndarray)
        assert isinstance(observation_gripper, np.ndarray)
        assert isinstance(action_cartesian, np.ndarray)
        assert isinstance(action_gripper, np.ndarray)

        end = start + self.horizon
        episode_dir = self.root / episode_id
        images = [
            self._frame(episode_dir / f"steps_observation_{camera}_left.mp4", start)
            for camera in _CAMERAS
        ]
        return {
            "image": images,
            "lang": str(payload["language"]),
            "state": self._state(observation_cartesian[start], observation_gripper[start])[None],
            "action": self._actions(action_cartesian[start:end], action_gripper[start:end], observation_cartesian[start]),
            "robot_tag": "oxe_droid",
        }


class EpisodeLocalSampler(Sampler[int]):
    """Shuffle episodes while retaining each episode's chronological windows."""

    def __init__(self, dataset: RawDroidOxeDPDataset, *, shuffle: bool = True, seed: int = 42) -> None:
        self.dataset, self.shuffle, self.seed, self.epoch = dataset, bool(shuffle), int(seed), 0
        groups: dict[str, list[int]] = {}
        for index, (episode_id, _) in enumerate(dataset.samples):
            groups.setdefault(episode_id, []).append(index)
        self._groups = [groups[episode_id] for episode_id, _ in dataset.episodes if episode_id in groups]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        order = list(range(len(self._groups)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        for group_index in order:
            yield from self._groups[group_index]

    def __len__(self) -> int:
        return len(self.dataset)


def collate_fn(batch):
    return batch


def get_vla_dataset(data_cfg, mode: str = "train", **kwargs):
    return RawDroidOxeDPDataset(data_cfg, mode=mode, **kwargs)
