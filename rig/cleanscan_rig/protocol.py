"""Wire protocol between the rig (Pi) and the CleanScan app. UDP, port 5600, little-endian.

Control messages are JSON objects (first byte ``{``):
  app -> rig  {"op": "hello"}                       subscribe (repeat every <= 2 s)
  rig -> app  {"op": "info", "sensor": "ld06", "kind": "2d", "version": 1}
  app -> rig  {"op": "ping", "id": 7, "t1": <app clock>}
  rig -> app  {"op": "pong", "id": 7, "t1": <echo>, "t2": <rig monotonic clock>}

Data packets (binary):
  magic "CSL1" | u8 kind (1 = 2D, 2 = 1D) | u8 reserved | u16 count | f64 t_base
  then ``count`` x (f32 dt, f32 angle_rad, f32 range_m, u8 intensity)    [13 bytes each]
  reading time on the rig clock = t_base + dt

The app converts rig time to its own clock with the NTP-style estimate from the
ping exchange that had the smallest round trip: offset = t2 - (t1 + t3) / 2.
"""

from __future__ import annotations

import json
import struct

from .sensors import Reading

PORT = 5600
MAGIC = b"CSL1"
KIND_CODE = {"2d": 1, "1d": 2}
CODE_KIND = {v: k for k, v in KIND_CODE.items()}
HEADER = struct.Struct("<4sBBHd")
POINT = struct.Struct("<fffB")
MAX_POINTS = (1400 - HEADER.size) // POINT.size  # stay under a typical Wi-Fi MTU


def encode_points(kind: str, readings: list[Reading]) -> list[bytes]:
    out = []
    for i in range(0, len(readings), MAX_POINTS):
        chunk = readings[i:i + MAX_POINTS]
        t_base = chunk[0].t
        b = bytearray(HEADER.pack(MAGIC, KIND_CODE[kind], 0, len(chunk), t_base))
        for r in chunk:
            b += POINT.pack(r.t - t_base, r.angle, r.range_m, max(0, min(255, int(r.intensity))))
        out.append(bytes(b))
    return out


def decode_points(packet: bytes) -> tuple[str, list[Reading]]:
    magic, kind, _, count, t_base = HEADER.unpack_from(packet, 0)
    if magic != MAGIC:
        raise ValueError("not a CleanScan rig data packet")
    if len(packet) != HEADER.size + count * POINT.size:
        raise ValueError("truncated packet")
    out = []
    for k in range(count):
        dt, ang, rng, inten = POINT.unpack_from(packet, HEADER.size + k * POINT.size)
        out.append(Reading(t_base + dt, ang, rng, inten))
    return CODE_KIND[kind], out


def encode_json(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()


def decode_json(data: bytes) -> dict | None:
    if not data.startswith(b"{"):
        return None
    try:
        return json.loads(data.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


class ClockSync:
    """Rig-clock -> app-clock offset from ping/pong exchanges (keeps the best of the last N)."""

    def __init__(self, keep: int = 30) -> None:
        self.samples: list[tuple[float, float]] = []  # (rtt, offset)
        self.keep = keep

    def add(self, t1: float, t2: float, t3: float) -> None:
        self.samples.append((t3 - t1, t2 - (t1 + t3) / 2.0))
        self.samples = self.samples[-self.keep:]

    @property
    def ready(self) -> bool:
        return bool(self.samples)

    @property
    def offset(self) -> float:
        """rig_time - app_time."""
        return min(self.samples)[1]

    @property
    def rtt(self) -> float:
        return min(self.samples)[0]

    def to_app_time(self, rig_t: float) -> float:
        return rig_t - self.offset
