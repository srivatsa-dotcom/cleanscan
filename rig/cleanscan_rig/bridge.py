"""Runs on the Raspberry Pi: reads the LiDAR on the serial port and streams it to the app.

    cleanscan-rig --sensor ld06                 # LD06 / LD19 on /dev/serial0
    cleanscan-rig --sensor tfluna               # TF-Luna / TFmini on /dev/serial0
    cleanscan-rig --sensor sim                  # no hardware, synthetic scan
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time

from . import protocol
from .sensors import Reading, SimSensor, parser_for

log = logging.getLogger("cleanscan-rig")
CLIENT_TIMEOUT = 5.0  # stop streaming to an app that stopped saying hello


class SerialReader(threading.Thread):
    """Reads the UART in a thread and hands parsed readings to the event loop."""

    def __init__(self, port: str, sensor: str, sink) -> None:
        super().__init__(daemon=True)
        import serial  # pyserial, imported here so tests/sim need no hardware libs

        self.parser = parser_for(sensor)
        self.kind = self.parser.kind
        self.ser = serial.Serial(port, self.parser.baud, timeout=0.02)
        self.sink = sink

    def run(self) -> None:
        while True:
            data = self.ser.read(512)
            if data:
                readings = self.parser.feed(data, time.monotonic())
                if readings:
                    self.sink(readings)


class SimReader(threading.Thread):
    def __init__(self, sink) -> None:
        super().__init__(daemon=True)
        self.sim = SimSensor()
        self.kind = self.sim.kind
        self.sink = sink

    def run(self) -> None:
        while True:
            time.sleep(0.01)
            readings = self.sim.read()
            if readings:
                self.sink(readings)


class Bridge(asyncio.DatagramProtocol):
    def __init__(self, sensor: str, kind: str) -> None:
        self.sensor, self.kind = sensor, kind
        self.clients: dict[tuple, float] = {}
        self.pending: list[Reading] = []
        self.lock = threading.Lock()
        self.transport = None
        self.sent = 0

    # -- readings from the sensor thread
    def add(self, readings: list[Reading]) -> None:
        with self.lock:
            self.pending.extend(readings)

    # -- UDP
    def connection_made(self, transport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        msg = protocol.decode_json(data)
        if not msg:
            return
        op = msg.get("op")
        if op == "hello":
            if addr not in self.clients:
                log.info("app connected: %s:%d", *addr[:2])
            self.clients[addr] = time.monotonic()
            self.transport.sendto(protocol.encode_json(
                {"op": "info", "sensor": self.sensor, "kind": self.kind, "version": 1}), addr)
        elif op == "ping":
            self.clients[addr] = time.monotonic()
            self.transport.sendto(protocol.encode_json(
                {"op": "pong", "id": msg.get("id"), "t1": msg.get("t1"), "t2": time.monotonic()}), addr)

    async def pump(self, period: float = 0.02) -> None:
        last_log = time.monotonic()
        while True:
            await asyncio.sleep(period)
            with self.lock:
                batch, self.pending = self.pending, []
            now = time.monotonic()
            for addr, seen in list(self.clients.items()):
                if now - seen > CLIENT_TIMEOUT:
                    log.info("app gone: %s:%d", *addr[:2])
                    del self.clients[addr]
            if batch and self.clients:
                for pkt in protocol.encode_points(self.kind, batch):
                    for addr in self.clients:
                        self.transport.sendto(pkt, addr)
                self.sent += len(batch)
            if now - last_log > 10:
                log.info("%d app(s), %.0f points/s", len(self.clients), self.sent / (now - last_log))
                self.sent, last_log = 0, now


async def serve(sensor: str, serial_port: str, udp_port: int) -> None:
    loop = asyncio.get_running_loop()
    holder: dict = {}

    def sink(readings):
        holder["bridge"].add(readings)

    reader = SimReader(sink) if sensor == "sim" else SerialReader(serial_port, sensor, sink)
    bridge = Bridge(sensor, reader.kind)
    holder["bridge"] = bridge
    await loop.create_datagram_endpoint(lambda: bridge, local_addr=("0.0.0.0", udp_port))
    reader.start()
    log.info("streaming %s (%s) on UDP port %d", sensor, reader.kind, udp_port)
    await bridge.pump()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cleanscan-rig", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor", default="ld06", help="ld06, ld19, tfluna, tfmini or sim")
    ap.add_argument("--serial", default="/dev/serial0", help="UART device")
    ap.add_argument("--port", type=int, default=protocol.PORT, help="UDP port")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    asyncio.run(serve(a.sensor, a.serial, a.port))


if __name__ == "__main__":
    main()
