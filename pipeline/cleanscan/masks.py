"""Dynamic-object (people / animals) masks.

Masks come from the phone (ARKit person segmentation + on-device YOLO-seg).
On the desktop they can be regenerated with a larger YOLO-seg model, and they
are always grown a little so hair, fur and depth "halo" pixels around a person
never make it into the map.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .session import Session

log = logging.getLogger(__name__)

# COCO class ids treated as "living beings" that must not appear in the map.
COCO_LIVING = {
    0: "person", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}


def grow_mask(mask: np.ndarray, depth: np.ndarray | None, dilate_px: int) -> np.ndarray:
    """Dilate the mask and also swallow depth pixels that 'belong' to the masked
    object (similar depth, touching the mask) so silhouettes are fully removed."""
    if not mask.any():
        return mask
    m = mask.astype(np.uint8)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    if depth is not None:
        # Region growing in depth: neighbours within 10 cm of the masked depth.
        masked_depth = np.where(m > 0, depth, 0).astype(np.float32)
        k3 = np.ones((3, 3), np.uint8)
        for _ in range(max(dilate_px, 2)):
            ring = cv2.dilate(m, k3) - m
            if not ring.any():
                break
            neigh = cv2.dilate(masked_depth, k3)  # max masked depth around each pixel
            grow = (ring > 0) & (depth > 0) & (np.abs(depth - neigh) < 0.10)
            if not grow.any():
                break
            m[grow] = 1
            masked_depth[grow] = depth[grow]
    return m > 0


class DesktopSegmenter:
    """Optional re-segmentation with ultralytics YOLO-seg (pip install ultralytics)."""

    def __init__(self, model: str = "yolo11x-seg.pt", conf: float = 0.25):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Desktop re-segmentation needs `pip install ultralytics`") from exc
        self.model = YOLO(model)
        self.conf = conf

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        res = self.model.predict(rgb[..., ::-1], conf=self.conf, classes=list(COCO_LIVING),
                                 retina_masks=True, verbose=False)[0]
        out = np.zeros((h, w), bool)
        if res.masks is not None:
            for m in res.masks.data.cpu().numpy():
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                out |= m > 0.5
        return out


def resegment_session(session: Session, model: str) -> dict[int, np.ndarray]:
    """Run the desktop segmenter on every colour frame. Returns masks at depth
    resolution keyed by frame index, OR-ed with the phone masks."""
    seg = DesktopSegmenter(model)
    out = {}
    for i, fr in enumerate(session.frames):
        m = seg(session.load_color(fr))
        m = cv2.resize(m.astype(np.uint8), tuple(fr.depth_size), interpolation=cv2.INTER_NEAREST) > 0
        phone = session.load_mask(fr)
        out[fr.index] = m | phone if phone is not None else m
        if i % 50 == 0:
            log.info("re-segmented %d/%d frames", i + 1, len(session.frames))
    return out
