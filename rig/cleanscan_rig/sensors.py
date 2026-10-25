"""LiDAR serial protocols.

Every sensor yields ``Reading(t, angle, range_m, intensity)`` where ``t`` is the
Pi's monotonic clock (seconds), ``angle`` is in radians in the sensor's own frame
(0 = sensor front; counter-clockwise seen from the sensor's top) and ``range_m`` is
0 for an invalid return.

Supported:
  * LD06 / LD19 (LDRobot, 2D, 230400 baud, 47-byte packets of 12 points)
  * TF-Luna / TFmini / TFmini-S / TFmini Plus (Benewake, 1D, 115200 baud, 9-byte frames)
  * "sim": a synthetic 2D scan for testing without hardware
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Reading:
    t: float
    angle: float
    range_m: float
    intensity: int


# --------------------------------------------------------------------------- LD06 / LD19

def _crc8_table(poly: int = 0x4D) -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return table


_LD_CRC = _crc8_table()
LD_HEADER, LD_VERLEN, LD_PACKET_LEN, LD_POINTS = 0x54, 0x2C, 47, 12


def ld_crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _LD_CRC[(crc ^ b) & 0xFF]
    return crc


class LD06Parser:
    """Incremental parser for LD06/LD19 packets.

    The sensor spins clockwise seen from the top and reports angles increasing in that
    direction; they are converted to the counter-clockwise convention used everywhere else.
    """

    kind = "2d"
    baud = 230400

    def __init__(self) -> None:
        self.buf = bytearray()
        self.bad_crc = 0

    def feed(self, data: bytes, t_now: float) -> list[Reading]:
        """Parse bytes received at ``t_now`` (Pi monotonic seconds)."""
        self.buf += data
        out: list[Reading] = []
        while True:
            i = self.buf.find(bytes([LD_HEADER, LD_VERLEN]))
            if i < 0:
                del self.buf[:-1]
                return out
            if len(self.buf) - i < LD_PACKET_LEN:
                del self.buf[:i]
                return out
            pkt = bytes(self.buf[i:i + LD_PACKET_LEN])
            if ld_crc8(pkt[:-1]) != pkt[-1]:
                self.bad_crc += 1
                del self.buf[:i + 1]
                continue
            del self.buf[:i + LD_PACKET_LEN]
            out.extend(self._decode(pkt, t_now))

    @staticmethod
    def _decode(pkt: bytes, t_now: float) -> list[Reading]:
        speed_dps = int.from_bytes(pkt[2:4], "little")
        start = int.from_bytes(pkt[4:6], "little") / 100.0
        end = int.from_bytes(pkt[42:44], "little") / 100.0
        span = (end - start) % 360.0
        step = span / (LD_POINTS - 1)
        # the packet arrives after its last point was measured: back-date each point
        dt_point = (step / speed_dps) if speed_dps > 0 else 0.0
        out = []
        for k in range(LD_POINTS):
            o = 6 + 3 * k
            dist_mm = int.from_bytes(pkt[o:o + 2], "little")
            conf = pkt[o + 2]
            ang_cw = math.radians((start + step * k) % 360.0)
            t = t_now - dt_point * (LD_POINTS - 1 - k)
            out.append(Reading(t, (-ang_cw) % (2 * math.pi), dist_mm / 1000.0, conf))
        return out


def ld_encode_packet(points: list[tuple[float, int]], start_deg: float, end_deg: float,
                     speed_dps: int = 3600, timestamp_ms: int = 0) -> bytes:
    """Build a valid LD06 packet (used by tests and the simulator)."""
    assert len(points) == LD_POINTS
    b = bytearray([LD_HEADER, LD_VERLEN])
    b += int(speed_dps).to_bytes(2, "little")
    b += int(round(start_deg * 100) % 36000).to_bytes(2, "little")
    for dist_m, conf in points:
        b += int(round(dist_m * 1000)).to_bytes(2, "little") + bytes([conf])
    b += int(round(end_deg * 100) % 36000).to_bytes(2, "little")
    b += int(timestamp_ms % 30000).to_bytes(2, "little")
    b.append(ld_crc8(bytes(b)))
    return bytes(b)


# --------------------------------------------------------------------------- TF-Luna / TFmini

class TFMiniParser:
    """Benewake 9-byte frames: 0x59 0x59 dist(cm) amp temp checksum."""

    kind = "1d"
    baud = 115200

    def __init__(self) -> None:
        self.buf = bytearray()
        self.bad_crc = 0

    def feed(self, data: bytes, t_now: float) -> list[Reading]:
        self.buf += data
        out = []
        while True:
            i = self.buf.find(b"\x59\x59")
            if i < 0 or len(self.buf) - i < 9:
                if i > 0:
                    del self.buf[:i]
                elif i < 0:
                    del self.buf[:-1]
                return out
            f = bytes(self.buf[i:i + 9])
            if sum(f[:8]) & 0xFF != f[8]:
                self.bad_crc += 1
                del self.buf[:i + 1]
                continue
            del self.buf[:i + 9]
            dist_cm = int.from_bytes(f[2:4], "little")
            amp = int.from_bytes(f[4:6], "little")
            ok = 100 <= amp < 65535 and 0 < dist_cm < 1200
            out.append(Reading(t_now, 0.0, dist_cm / 100.0 if ok else 0.0, min(255, amp // 16)))


def tfmini_encode_frame(dist_m: float, amp: int = 2000, temp: int = 2000) -> bytes:
    b = bytearray(b"\x59\x59")
    b += int(round(dist_m * 100)).to_bytes(2, "little") + int(amp).to_bytes(2, "little")
    b += int(temp).to_bytes(2, "little")
    b.append(sum(b) & 0xFF)
    return bytes(b)


# --------------------------------------------------------------------------- simulator

class SimSensor:
    """A 2D LiDAR spinning at 10 Hz in the middle of a 6 x 5 m room (no hardware needed)."""

    kind = "2d"

    def __init__(self, rate_hz: float = 4500.0, rev_hz: float = 10.0) -> None:
        self.rate_hz, self.rev_hz = rate_hz, rev_hz
        self.t0 = time.monotonic()
        self.n = 0

    @staticmethod
    def room_range(angle: float, half_x: float = 3.0, half_y: float = 2.5) -> float:
        c, s = math.cos(angle), math.sin(angle)
        rx = half_x / abs(c) if abs(c) > 1e-9 else math.inf
        ry = half_y / abs(s) if abs(s) > 1e-9 else math.inf
        return min(rx, ry)

    def read(self) -> list[Reading]:
        now = time.monotonic()
        due = int((now - self.t0) * self.rate_hz)
        out = []
        while self.n < due:
            t = self.t0 + self.n / self.rate_hz
            a = (2 * math.pi * self.rev_hz * (t - self.t0)) % (2 * math.pi)
            out.append(Reading(t, a, self.room_range(a), 200))
            self.n += 1
        return out


def parser_for(name: str):
    name = name.lower()
    if name in ("ld06", "ld19", "ld"):
        return LD06Parser()
    if name in ("tfluna", "tf-luna", "tfmini", "tfmini-s", "tfmini-plus", "tf"):
        return TFMiniParser()
    raise ValueError(f"unknown sensor {name!r} (use ld06, ld19, tfluna, tfmini or sim)")


def iter_chunks(data: bytes, size: int) -> Iterable[bytes]:
    for i in range(0, len(data), size):
        yield data[i:i + size]


def readings_from_stream(parser, data: bytes, t: float = 0.0) -> Iterator[Reading]:
    for chunk in iter_chunks(data, 13):
        yield from parser.feed(chunk, t)
