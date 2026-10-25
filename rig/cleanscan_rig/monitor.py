"""Laptop-side check of a running rig, without the iPhone: does exactly what the app does
(hello, clock sync, receive) and prints rate, clock sync and a few ranges.

    cleanscan-rig-monitor --host 10.42.0.1
"""

from __future__ import annotations

import argparse
import socket
import time

from . import protocol


def monitor(host: str, port: int, seconds: float, quiet: bool = False) -> dict:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.05)
    sync = protocol.ClockSync()
    info, n, ping_id, last_ping = None, 0, 0, 0.0
    ranges: list[float] = []
    t_end = time.monotonic() + seconds
    latencies = []
    while time.monotonic() < t_end:
        now = time.monotonic()
        if now - last_ping > 0.5:
            sock.sendto(protocol.encode_json({"op": "hello"}), (host, port))
            ping_id += 1
            sock.sendto(protocol.encode_json({"op": "ping", "id": ping_id, "t1": now}), (host, port))
            last_ping = now
        try:
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            continue
        t3 = time.monotonic()
        msg = protocol.decode_json(data)
        if msg is not None:
            if msg.get("op") == "info":
                info = msg
            elif msg.get("op") == "pong":
                sync.add(msg["t1"], msg["t2"], t3)
            continue
        kind, readings = protocol.decode_points(data)
        n += len(readings)
        ranges.extend(r.range_m for r in readings if r.range_m > 0)
        if sync.ready:
            latencies.append(t3 - sync.to_app_time(readings[-1].t))
    sock.close()
    result = {
        "info": info,
        "points_per_s": n / seconds,
        "rtt_ms": sync.rtt * 1000 if sync.ready else None,
        "latency_ms": 1000 * sorted(latencies)[len(latencies) // 2] if latencies else None,
        "median_range_m": sorted(ranges)[len(ranges) // 2] if ranges else None,
    }
    if not quiet:
        print(result)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cleanscan-rig-monitor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="10.42.0.1", help="rig address (Pi hotspot default)")
    ap.add_argument("--port", type=int, default=protocol.PORT)
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args(argv)
    r = monitor(a.host, a.port, a.seconds, quiet=True)
    if r["info"] is None:
        print(f"No answer from {a.host}:{a.port}. Is the Pi on, and is this computer on its Wi-Fi?")
        return
    print(f"sensor {r['info']['sensor']} ({r['info']['kind']}): {r['points_per_s']:.0f} points/s, "
          f"round trip {r['rtt_ms']:.1f} ms, median range {r['median_range_m']}")
    if r["points_per_s"] < 1:
        print("Connected but no points: check the LiDAR wiring (sensor TX -> Pi GPIO15/RXD).")


if __name__ == "__main__":
    main()
