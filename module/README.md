# CleanScan depth module: a clip-on ToF camera for phones without LiDAR

An iPhone 11 (or any phone without a depth sensor) knows **where it is** (ARKit/ARCore) but
not **how far things are**. This clip-on module adds a small time-of-flight camera that
streams depth images to the CleanScan app over its own Wi-Fi; the app records them next to the
camera frames and `cleanscan process` / `process-rl` fuse them.

```
 phone (rear camera) ── Wi-Fi "CleanScanDepth" ── ESP32-S3 ── UART 2 Mbit/s ── MaixSense A010
                         UDP 5600, clock sync        (firmware/)                100 x 100 ToF
```

* **Hardware:** MaixSense A010 + ESP32-S3 on a 3D-printed clip, powered by a USB power bank.
  Parts: [hardware/BOM.md](hardware/BOM.md), clip: [hardware/clip.scad](hardware/clip.scad).
* **Alternative:** Raspberry Pi Zero 2 W + Arducam ToF (240 x 180), using the LiDAR rig software:
  `sudo ./setup/install.sh arducam_tof` in [rig/](../rig/README.md).
* **Licences:** firmware MIT ([firmware/LICENSE](firmware/LICENSE)); hardware design
  CERN-OHL-P-2.0 ([hardware/LICENSE](hardware/LICENSE)).

In simulation (synthetic kitchen, `pipeline/tests/test_kitchen_e2e.py`) an iPhone 11 with the
module rebuilds fridge, drawer, bottles and glass within the RL-scene tolerances and the
mount is calibrated automatically to < 1° / < 1 cm. **Real-world numbers will be worse; nothing
here has run on real hardware yet** (see "Needs testing" below).

## Wiring (A010 → ESP32-S3)

| A010 pin | ESP32-S3-DevKitC-1 | XIAO ESP32S3 |
|---|---|---|
| 5V | 5V | 5V |
| GND | GND | GND |
| TX (A010 sends) | GPIO18 (ESP32 RX) | GPIO44 / D7 |
| RX (A010 receives) | GPIO17 (ESP32 TX) | GPIO43 / D6 |

The A010's UART is 3.3 V logic, so it connects directly. Power the ESP32 board from the power
bank's USB; it passes 5 V to the A010.

## Flash the firmware (one command)

```bash
pip install platformio
cd module/firmware
pio run -t upload                 # ESP32-S3-DevKitC-1;  XIAO:  pio run -e xiao_esp32s3 -t upload
```

Or without PlatformIO: download `cleanscan-depth-esp32s3-factory.bin` from the repository's
`latest` release and flash it at offset 0 with `pip install esptool` and
`esptool.py --chip esp32s3 write_flash 0x0 cleanscan-depth-esp32s3-factory.bin`.

**Wi-Fi password:** default `cleanscan123`. Change it at build time with
`PLATFORMIO_BUILD_FLAGS='-DCLEANSCAN_AP_PASSWORD=\"your-password\"' pio run -t upload`, or on a
flashed module: `pio device monitor`, type `pass your-password` (8+ characters), Enter.

`pio device monitor` shows the frame rate every 10 s (`15.0 frames/s, 0 bad frames`).

## Mounting

* Clip it on so the A010's lens sits **next to the phone's main rear camera, looking the same
  way**. The software finds the exact offset and angle itself (the clip only has to be rigid);
  it assumes the module looks roughly where the camera looks (within ~20°).
* Keep the lens clear of fingers and the phone case.

## Scanning

1. Power the module; join the Wi-Fi **CleanScanDepth** on the phone (iOS keeps it although it
   has no internet).
2. In CleanScan: **Scans → External depth module → on**. The scan screen shows
   `Depth module: A010 · 15 fps · sync 4 ms`. On iOS, allow "Local network" access.
3. Scan as usual (room pass, then the guided object captures). Stay within **0.2-2.5 m** of
   surfaces where you can: the A010 sees nothing further away; there the pipeline falls back to
   camera stereo.
4. Process on the computer: `cleanscan process-rl Scan-XXXX.zip -o my_kitchen`.
   `report.json → room.depth_module` shows the calibration that was found.

## How it works

* **Clock sync:** the app pings the module twice a second; the exchange with the shortest round
  trip gives the module-to-phone clock offset (same maths as the LiDAR rig). Each frame is
  timestamped when its first UART byte arrives, minus the frame's transfer time.
* **Protocol:** UDP port 5600; `hello` → `info` (sensor, size, field of view), `ping` → `pong`;
  depth frames as `CSD1` packets (one 100 x 100 frame = 15 datagrams of ≤ 1400 bytes),
  see `rig/cleanscan_rig/protocol.py`. The firmware's packet code is checked against the Python
  decoder in `rig/tests`.
* **Calibration** (`pipeline/cleanscan/depthmodule.py`): the module → camera transform is the
  one that makes the depth points from different moments agree (the LiDAR rig's method), started
  from "next to the camera, looking the same way". The room pass and all object captures are
  used together.
* **Depth for the camera frames:** module points are projected into each camera frame (pose
  interpolated at each depth frame's time) and filled in between with an edge-aware,
  colour-guided upsampling; the module's measurements are kept as they are.

## A010 settings used (from Sipeed's documentation)

`AT+ISP=0`, `AT+DISP=4` (UART output), `AT+BINN=1` (100 x 100), `AT+UNIT=0`
(depth code → mm = (code / 5.1)², up to 2.5 m), `AT+FPS=15`, `AT+BAUD=7` (2 Mbit/s), `AT+ISP=1`.
Frames on the UART: `00 FF | u16 length | 16-byte header | depth codes | checksum | DD`.

## Needs testing on real hardware

Everything in this folder compiles and the protocol code is unit-tested on a computer, but
nothing has run on an ESP32-S3 or an A010 yet:

* the A010's AT commands, baud-rate index and frame layout above (taken from Sipeed's docs; if
  your firmware differs, `pio device monitor` shows `bad frames` going up)
* the achieved frame rate over Wi-Fi and the timestamp accuracy
* the clip dimensions (`clip.scad` has not been printed; measure your phone and boards)
* the Arducam variant (its Python SDK calls follow Arducam's examples)
