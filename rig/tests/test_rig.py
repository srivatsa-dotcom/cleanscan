import asyncio
import math
import threading
import time

from cleanscan_rig import protocol
from cleanscan_rig.sensors import (LD06Parser, Reading, TFMiniParser, ld_crc8, ld_encode_packet,
                                   tfmini_encode_frame)


def test_ld06_crc_known_table_values():
    # first entries of the CRC table in LDRobot's SDK (poly 0x4D)
    assert [ld_crc8(bytes([i])) for i in range(5)] == [0x00, 0x4D, 0x9A, 0xD7, 0x79]


def test_ld06_parser_roundtrip_with_garbage_and_split_reads():
    pts = [(1.0 + 0.1 * k, 200) for k in range(12)]
    pkt = ld_encode_packet(pts, start_deg=10.0, end_deg=21.0, speed_dps=3600)
    corrupt = bytearray(pkt)
    corrupt[10] ^= 0xFF  # bad CRC: must be skipped
    stream = b"\x00\x54\x13" + bytes(corrupt) + pkt + pkt[:20]
    p = LD06Parser()
    out = []
    for i in range(0, len(stream), 7):
        out += p.feed(stream[i:i + 7], t_now=100.0)
    assert len(out) == 12 and p.bad_crc >= 1
    assert abs(out[0].range_m - 1.0) < 1e-6 and abs(out[11].range_m - 2.1) < 1e-6
    # clockwise sensor angles become counter-clockwise: 10 deg cw -> 350 deg ccw
    assert abs(math.degrees(out[0].angle) - 350.0) < 1e-3
    assert abs(math.degrees(out[11].angle) - 339.0) < 1e-3
    # points are back-dated from the packet arrival time (1 deg per point at 3600 deg/s)
    assert out[-1].t == 100.0 and abs((out[-1].t - out[0].t) - 11 / 3600) < 1e-9


def test_ld06_angle_wraps_through_zero():
    pkt = ld_encode_packet([(2.0, 100)] * 12, start_deg=355.0, end_deg=6.0)
    out = LD06Parser().feed(pkt, 0.0)
    degs_cw = [(-math.degrees(r.angle)) % 360 for r in out]
    assert abs(degs_cw[0] - 355) < 1e-3 and abs(degs_cw[-1] - 6) < 1e-3


def test_tfmini_parser():
    stream = b"\x59" + tfmini_encode_frame(1.23) + tfmini_encode_frame(4.5, amp=50) + b"\x59\x59\x01"
    p = TFMiniParser()
    out = p.feed(stream, 5.0)
    assert len(out) == 2
    assert abs(out[0].range_m - 1.23) < 1e-9
    assert out[1].range_m == 0.0  # amplitude too weak -> invalid


def test_protocol_roundtrip_and_split():
    rs = [Reading(10.0 + i * 1e-3, i * 0.01, 1.0 + i * 1e-3, i % 256) for i in range(250)]
    pkts = protocol.encode_points("2d", rs)
    assert len(pkts) == 3 and all(len(p) <= 1400 for p in pkts)
    back = []
    for p in pkts:
        kind, r = protocol.decode_points(p)
        assert kind == "2d"
        back += r
    assert len(back) == 250
    assert all(abs(a.t - b.t) < 1e-6 and abs(a.range_m - b.range_m) < 1e-6 for a, b in zip(rs, back))


def test_clock_sync_uses_fastest_exchange():
    s = protocol.ClockSync()
    true_offset = 1234.5  # rig - app
    for rtt, asym in [(0.050, 0.03), (0.004, 0.0), (0.020, -0.01)]:
        t1 = 100.0
        t2 = t1 + rtt / 2 + asym + true_offset
        s.add(t1, t2, t1 + rtt)
    assert abs(s.offset - true_offset) < 1e-9 and abs(s.rtt - 0.004) < 1e-12


def test_bridge_and_monitor_end_to_end():
    """Run the real bridge (simulated sensor) and the monitor client over UDP on localhost."""
    from cleanscan_rig import bridge, monitor

    port = 56123
    t = threading.Thread(target=lambda: asyncio.run(bridge.serve("sim", "", port)), daemon=True)
    t.start()
    time.sleep(0.5)
    r = monitor.monitor("127.0.0.1", port, seconds=2.0, quiet=True)
    assert r["info"]["sensor"] == "sim" and r["info"]["kind"] == "2d"
    assert r["points_per_s"] > 3000  # simulated 4500 points/s
    assert r["rtt_ms"] < 50
    assert 2.4 < r["median_range_m"] < 4.0  # inside the simulated 6 x 5 m room
    assert r["latency_ms"] < 100


def test_depth_frame_roundtrip_out_of_order_and_lost_chunks():
    import random

    w, h = 100, 100
    frame = [(x * 7 + y * 13) % 4000 for y in range(h) for x in range(w)]
    pk = protocol.encode_depth_frame(42, 123.456, w, h, frame)
    assert len(pk) == 15 and all(len(p) <= 1400 for p in pk)  # 20 kB in MTU-sized packets
    asm = protocol.DepthAssembler()
    random.Random(0).shuffle(pk)
    out = [r for p in pk if (r := asm.feed(p))]
    assert len(out) == 1
    fid, t, ww, hh, px = out[0]
    assert (fid, t, ww, hh) == (42, 123.456, w, h) and px == frame
    # a frame missing one chunk is dropped once a newer frame completes
    lost = protocol.encode_depth_frame(43, 124.0, w, h, frame)[:-1]
    for p in lost:
        assert asm.feed(p) is None
    newer = [asm.feed(p) for p in protocol.encode_depth_frame(44, 124.1, w, h, frame)]
    assert newer[-1][0] == 44 and asm.dropped == 1


def test_depth_module_bridge_streams_frames_with_clock_sync():
    """simdepth sensor -> bridge -> UDP -> app-side assembler, like the phone apps do."""
    import socket

    from cleanscan_rig import bridge

    port = 56124  # the LiDAR bridge test above keeps 56123 open
    th = threading.Thread(target=lambda: asyncio.run(bridge.serve("simdepth", "", port)), daemon=True)
    th.start()
    time.sleep(0.5)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.5)
    s.sendto(protocol.encode_json({"op": "hello"}), ("127.0.0.1", port))
    asm = protocol.DepthAssembler()
    info, frame, deadline = None, None, time.time() + 10
    s.sendto(protocol.encode_json({"op": "ping", "id": 1, "t1": time.monotonic()}), ("127.0.0.1", port))
    pong = None
    while time.time() < deadline and (frame is None or info is None or pong is None):
        try:
            data, _ = s.recvfrom(2048)
        except socket.timeout:
            s.sendto(protocol.encode_json({"op": "hello"}), ("127.0.0.1", port))
            continue
        msg = protocol.decode_json(data)
        if msg and msg["op"] == "info":
            info = msg
        elif msg and msg["op"] == "pong":
            pong = msg
        elif data.startswith(protocol.DEPTH_MAGIC):
            frame = asm.feed(data) or frame
    assert info["kind"] == "depth" and info["sensor"] == "simdepth" and info["width"] == 100
    assert pong is not None
    fid, t, w, h, mm = frame
    assert (w, h) == (100, 100) and sum(1 for v in mm if v) > 5000  # walls within the 2.5 m range


def test_firmware_packets_decode_in_python(tmp_path):
    """The ESP32 firmware's C++ encoder (module/firmware) and the Python decoder agree."""
    import shutil
    import subprocess
    from pathlib import Path

    import pytest

    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    fw = Path(__file__).resolve().parents[2] / "module" / "firmware"
    exe = tmp_path / "csd1_dump"
    subprocess.run([cxx, "-std=c++17", "-O1", f"-I{fw / 'lib/cleanscan_proto/src'}",
                    str(fw / "test/host/csd1_dump.cpp"), str(fw / "lib/cleanscan_proto/src/cleanscan_proto.cpp"),
                    "-o", str(exe)], check=True)
    n = int(subprocess.run([str(exe), str(tmp_path)], check=True, capture_output=True, text=True).stdout)
    asm = protocol.DepthAssembler()
    out = [r for k in range(n) if (r := asm.feed((tmp_path / f"pkt{k}.bin").read_bytes()))]
    fid, t, w, h, mm = out[0]
    assert (fid, t, w, h) == (4242, 1000.125, 100, 100)
    assert mm == [(i * 37) % 2600 for i in range(10000)]
