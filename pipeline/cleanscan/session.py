"""Reading CleanScan capture sessions (see docs/capture_format.md).

Conventions (identical for ARKit and ARCore, which is why both apps can share
one format):
  * world frame: right-handed, +Y up (gravity aligned), metres
  * camera frame: OpenGL style, +X right, +Y up, camera looks down -Z
  * ``camera_to_world`` is a row-major 4x4 for the *sensor* (landscape) image
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# OpenGL camera (x right, y up, -z forward) -> OpenCV camera (x right, y down, +z forward)
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


@dataclass
class Frame:
    index: int
    timestamp: float
    color_path: Path
    depth_path: Path | None  # None: camera-only capture (no LiDAR/ToF), depth is estimated
    confidence_path: Path | None
    mask_path: Path | None
    K_color: np.ndarray
    color_size: tuple[int, int]  # (w, h)
    K_depth: np.ndarray
    depth_size: tuple[int, int]  # (w, h)
    pose_gl: np.ndarray  # camera -> world, OpenGL camera axes

    @property
    def pose_cv(self) -> np.ndarray:
        """camera -> world with OpenCV camera axes."""
        return self.pose_gl @ GL_TO_CV

    @property
    def extrinsic_cv(self) -> np.ndarray:
        """world -> camera (OpenCV axes), what Open3D integration expects."""
        return np.linalg.inv(self.pose_cv)


# High-rate ARKit/ARCore poses (every camera frame, ~60 Hz): f64 time + row-major 4x4 (f32)
POSE_DTYPE = np.dtype([("t", "<f8"), ("m", "<f4", (16,))])
# LiDAR readings already converted to the phone clock (see rig/cleanscan_rig/protocol.py)
LIDAR_DTYPE = np.dtype([("t", "<f8"), ("angle", "<f4"), ("range", "<f4"), ("intensity", "u1")])


class Session:
    def __init__(self, root: Path, meta: dict, frames: list[Frame]):
        self.root = root
        self.meta = meta
        self.frames = frames
        self.depth_unit = float(meta.get("depth_unit", 0.001))
        self.confidence_levels = int(meta.get("confidence_levels", 3))
        self.poses: tuple[np.ndarray, np.ndarray] | None = None  # (t, camera_to_world GL)
        self.lidar: np.ndarray | None = None  # LIDAR_DTYPE records
        self.rig: dict = {}
        self.planes: list[dict] = []
        self._load_rig()

    def _load_rig(self) -> None:
        """Optional extras: 60 Hz poses, an external LiDAR (rig) and ARKit planes."""
        p = self.root / "poses.bin"
        if p.exists():
            rec = np.fromfile(p, POSE_DTYPE)
            if len(rec) >= 2:
                order = np.argsort(rec["t"])
                self.poses = (rec["t"][order], rec["m"][order].reshape(-1, 4, 4).astype(np.float64))
        if (self.root / "rig.json").exists():
            self.rig = json.loads((self.root / "rig.json").read_text())
        p = self.root / "lidar.bin"
        if p.exists():
            rec = np.fromfile(p, LIDAR_DTYPE)
            self.lidar = rec[np.argsort(rec["t"], kind="stable")] if len(rec) else None
        if (self.root / "planes.json").exists():
            self.planes = json.loads((self.root / "planes.json").read_text())

    @property
    def has_lidar(self) -> bool:
        return self.lidar is not None and len(self.lidar) > 0

    @classmethod
    def load(cls, root: str | Path) -> "Session":
        root = Path(root)
        meta_path = root / "session.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found - is this a CleanScan session folder?")
        meta = json.loads(meta_path.read_text())
        if meta.get("format") != "cleanscan-capture":
            raise ValueError(f"{meta_path}: unexpected format {meta.get('format')!r}")

        frames = []
        with open(root / "frames.jsonl") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                f = json.loads(line)

                def opt(key):
                    return root / f[key] if f.get(key) else None

                frames.append(
                    Frame(
                        index=int(f["index"]),
                        timestamp=float(f.get("timestamp", 0.0)),
                        color_path=root / f["color"],
                        depth_path=opt("depth"),
                        confidence_path=opt("confidence"),
                        mask_path=opt("mask"),
                        K_color=intrinsics_matrix(*f["color_intrinsics"]),
                        color_size=tuple(f["color_size"]),
                        K_depth=intrinsics_matrix(*f["depth_intrinsics"]),
                        depth_size=tuple(f["depth_size"]),
                        pose_gl=np.array(f["camera_to_world"], dtype=np.float64).reshape(4, 4),
                    )
                )
        if not frames:
            raise ValueError(f"{root}: session has no frames")
        frames.sort(key=lambda fr: fr.timestamp)
        return cls(root, meta, frames)

    @property
    def has_sensor_depth(self) -> bool:
        return all(f.depth_path is not None for f in self.frames)

    # ------------------------------------------------------------------ loaders
    def load_depth(self, frame: Frame) -> np.ndarray:
        """Depth in metres, float32 (H, W); 0 = invalid."""
        if frame.depth_path is None:
            raise ValueError("frame has no depth - run stereo.estimate_session_depth first")
        raw = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f"cannot read {frame.depth_path}")
        return raw.astype(np.float32) * self.depth_unit

    def load_confidence(self, frame: Frame) -> np.ndarray | None:
        """Confidence normalised to [0, 1] at depth resolution, or None."""
        if frame.confidence_path is None or not frame.confidence_path.exists():
            return None
        raw = cv2.imread(str(frame.confidence_path), cv2.IMREAD_UNCHANGED)
        return raw.astype(np.float32) / max(self.confidence_levels - 1, 1)

    def load_color(self, frame: Frame) -> np.ndarray:
        """RGB uint8 (H, W, 3) at colour resolution."""
        bgr = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"cannot read {frame.color_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def load_mask(self, frame: Frame, size: tuple[int, int] | None = None) -> np.ndarray | None:
        """Dynamic-object mask (True = person/animal) resized to ``size`` (w, h)."""
        if frame.mask_path is None or not frame.mask_path.exists():
            return None
        m = cv2.imread(str(frame.mask_path), cv2.IMREAD_GRAYSCALE)
        size = size or frame.depth_size
        if (m.shape[1], m.shape[0]) != tuple(size):
            m = cv2.resize(m, tuple(size), interpolation=cv2.INTER_NEAREST)
        return m > 127
