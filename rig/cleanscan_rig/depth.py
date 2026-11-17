"""Depth-image sources for the depth-module variant of the rig (protocol: CSD1 frames).

    cleanscan-rig --sensor arducam_tof   # Arducam ToF camera (240 x 180) on a Pi Zero 2 W
    cleanscan-rig --sensor simdepth      # synthetic room, no hardware (tests, app development)

The ESP32-S3 + MaixSense A010 module (module/firmware) sends the same packets.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


@dataclass
class DepthFrame:
    frame_id: int
    t: float  # module clock (time.monotonic), like LiDAR readings
    width: int
    height: int
    mm: list[int]  # row-major, 0 = no measurement


class SimDepthSensor:
    """A 100 x 100 ToF in the middle of a 3.6 x 3 m room, slowly turning (like the A010)."""

    sensor, width, height, fov_deg = "simdepth", 100, 100, (70.0, 60.0)

    def __init__(self, fps: float = 15.0) -> None:
        self.period = 1.0 / fps
        self.n = 0

    def read(self) -> DepthFrame:
        t = time.monotonic()
        yaw = 0.3 * t
        fx = (self.width / 2) / math.tan(math.radians(self.fov_deg[0]) / 2)
        fy = (self.height / 2) / math.tan(math.radians(self.fov_deg[1]) / 2)
        mm = []
        for v in range(self.height):
            for u in range(self.width):
                dx, dy, dz = (u + 0.5 - self.width / 2) / fx, (v + 0.5 - self.height / 2) / fy, 1.0
                c, s = math.cos(yaw), math.sin(yaw)
                wx, wz = c * dx + s * dz, -s * dx + c * dz
                hits = [r for r in ((1.8 / wx) if wx > 1e-6 else math.inf, (-1.8 / wx) if wx < -1e-6 else math.inf,
                                    (1.5 / wz) if wz > 1e-6 else math.inf, (-1.5 / wz) if wz < -1e-6 else math.inf,
                                    (1.2 / dy) if dy > 1e-6 else math.inf)]  # walls, floor 1.2 m below
                z = min(hits)  # unit z in the camera, so the ray parameter is depth
                mm.append(int(z * 1000) if 0.2 < z < 2.5 else 0)
        self.n += 1
        return DepthFrame(self.n, t, self.width, self.height, mm)


class ArducamToF:
    """Arducam ToF camera through its Python SDK (pip install ArducamDepthCamera).

    The SDK calls follow Arducam's examples; verify them against the SDK version you install
    (README: needs testing on hardware)."""

    sensor, fov_deg = "arducam_tof", (70.0, 55.0)

    def __init__(self) -> None:
        import ArducamDepthCamera as ac  # noqa: N813 - vendor module name

        self.ac = ac
        self.cam = ac.ArducamCamera()
        if self.cam.open(ac.Connection.CSI, 0) != 0:
            raise RuntimeError("Arducam ToF camera not found on the CSI port")
        if self.cam.start(ac.FrameType.DEPTH) != 0:
            raise RuntimeError("Arducam ToF camera did not start")
        self.n = 0
        self.width = self.height = None

    def read(self) -> DepthFrame | None:
        frame = self.cam.requestFrame(200)
        if frame is None:
            return None
        t = time.monotonic()
        try:
            depth = frame.depth_data  # numpy (h, w), millimetres
            conf = getattr(frame, "confidence_data", None)
            h, w = depth.shape
            self.width, self.height = w, h
            mm = depth.astype("float32")
            if conf is not None:
                mm[conf < 30] = 0  # low-confidence pixels (dark or far surfaces)
            mm = [int(x) if 0 < x < 65535 else 0 for x in mm.reshape(-1)]
        finally:
            self.cam.releaseFrame(frame)
        self.n += 1
        return DepthFrame(self.n, t, w, h, mm)


def depth_source(name: str):
    if name == "simdepth":
        return SimDepthSensor()
    if name == "arducam_tof":
        return ArducamToF()
    raise ValueError(f"unknown depth sensor {name!r}")


class DepthReader(threading.Thread):
    """Reads frames in a thread and hands them to the event loop's sink."""

    kind = "depth"

    def __init__(self, sensor: str, sink, period: float = 1 / 15) -> None:
        super().__init__(daemon=True)
        self.source = depth_source(sensor)
        self.sink = sink
        self.period = period

    def run(self) -> None:
        while True:
            t0 = time.monotonic()
            f = self.source.read()
            if f is not None:
                self.sink(f)
            time.sleep(max(0.0, self.period - (time.monotonic() - t0)))
