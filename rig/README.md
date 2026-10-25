# CleanScan LiDAR rig: iPhone 11 (or any phone) + Raspberry Pi + LiDAR

A phone without a depth sensor knows **where it is** (ARKit/ARCore) but not **how far
things are**. A cheap LiDAR knows the distance but not where it is. Mounted together they
make a handheld scanner: the Pi streams the LiDAR to the CleanScan app over Wi-Fi, the app
records it next to the camera, and `cleanscan process` combines them.

```
   ┌──────────── handheld rig ─────────────┐
   │  iPhone  <── Wi-Fi "CleanScanRig" ──┐ │
   │  camera + ARKit pose (60 Hz)        │ │
   │                            Raspberry Pi Zero (W / 2 W)
   │                                     │ UART (serial)
   │                            LD06 / LD19 2D LiDAR  (or TF-Luna 1D)
   └───────────────────────────────────────┘   all on one rigid bracket
```

Simulated results on the test room (`pipeline/tests`), same walk each time:

| Setup | Median error | 90 % within | Walls/floor covered |
|---|---|---|---|
| iPhone 11 camera only | 1.3 cm | 3.6 cm | 43 % |
| + LD06 2D LiDAR | **0.9 cm** | **1.6 cm** | 50 % |
| + LD06 + camera stereo (default) | **0.9 cm** | 2.0 cm | **61 %** |
| + TF-Luna 1D LiDAR | no real gain: too few readings (100/s) | | |

**Use a 2D LiDAR (LD06 or LD19).** 1D sensors (TF-Luna, TFmini-S) are supported but add
little. Real-world numbers will be worse than the simulation; please report what you get.

## Parts

| Part | Notes | Approx. price (India) |
|---|---|---|
| LDRobot **LD06** or **LD19** | 2D, 360°, 4500 points/s, 12 m | ₹2,000–3,000 |
| Raspberry Pi Zero 2 W (Zero W also works) | with micro-SD card (8 GB+) | ₹1,600 |
| USB power bank + micro-USB cable | powers the Pi; the Pi powers the LiDAR | — |
| Phone clamp + handle (selfie stick / tripod clamp) | | ₹300 |
| Rigid bracket (3D print, aluminium strip, acrylic) | the LiDAR must not wobble against the phone | — |
| 4 female-female jumper wires | | — |

## Wiring

Pi Zero header pins: pin 1 has a **square** solder pad on the underside; even-numbered pins (2, 4, 6, …) are the row along the board's edge.

| LiDAR wire | LD06 / LD19 (4-pin) | TF-Luna (6-pin) | Pi Zero pin |
|---|---|---|---|
| 5 V | P5V | pin 1 (+5 V) | **pin 2** (5 V) |
| GND | GND | pin 4 (GND) | **pin 6** (GND) |
| LiDAR TX → Pi RX | Tx | pin 3 (TXD) | **pin 10** (GPIO15 / RXD) |
| (not needed) | PWM: leave unconnected | pin 2 (RXD), pin 5 (config: leave open = UART), pin 6 | — |

Both sensors use 3.3 V logic on their TX line, so they connect directly to the Pi.

## Mounting

* Mount the LD06 **above the phone**, spinning plane roughly level when you hold the phone
  upright. Any orientation works: the software finds the exact mount itself.
* It must be **rigid**: if the LiDAR can move relative to the phone, the scan smears.
* Keep the LiDAR's view clear of your hand and the phone where possible.

## Pi setup (once)

1. Flash **Raspberry Pi OS Lite (Bookworm)** with Raspberry Pi Imager; in its settings set
   a username/password, your home Wi-Fi, and enable SSH.
2. Copy this folder to the Pi and run the setup (it needs internet once):

   ```bash
   scp -r rig pi@raspberrypi.local:~/
   ssh pi@raspberrypi.local
   cd rig && sudo ./setup/install.sh ld06      # or ld19, tfluna, tfmini
   sudo reboot
   ```

3. After the reboot the Pi creates the Wi-Fi network **CleanScanRig** (password
   `cleanscan123`, change it with `CLEANSCAN_PASS=... sudo ./setup/install.sh ld06`), and
   streams automatically on every power-up.

**Check it from your laptop** (join the CleanScanRig Wi-Fi first):

```bash
pip install ./rig
cleanscan-rig-monitor --host 10.42.0.1
# sensor ld06 (2d): 4500 points/s, round trip 4.1 ms, median range 2.3
```

Without any hardware you can try the whole chain with a simulated sensor:
`cleanscan-rig --sensor sim` on one terminal, `cleanscan-rig-monitor --host 127.0.0.1` on another.

## Scanning

1. Power the rig, join **CleanScanRig** on the iPhone (iOS keeps it although it has no internet).
2. In CleanScan: **Scans → External LiDAR rig → Use LiDAR rig** on. Allow "local network"
   access when iOS asks. The scan screen shows `Rig: LD06 2d · 4500 pts/s`.
3. Press **Record**. First point the rig at a **corner** (floor + two walls) for ~5 seconds;
   this helps the automatic calibration. Then walk slowly along the walls, **tilting the
   phone up and down** so the LiDAR's scan line sweeps from floor to ceiling.
4. Stop, export, and process as usual on the laptop:

   ```bash
   cleanscan process Scan-XXXX.zip -o my_room          # LiDAR + camera stereo
   cleanscan process Scan-XXXX.zip -o my_room --rig-no-stereo   # LiDAR only, faster
   ```

`report.json` shows the calibration that was found (`rig.calibration`) and how many LiDAR
readings hit people/animals and were dropped.

## How it works

* **Clock sync:** the app pings the Pi twice a second; the exchange with the shortest round
  trip gives the offset between the Pi's and the phone's clocks (`protocol.py`), so every
  reading gets a phone timestamp to a few milliseconds.
* **Poses:** the app records every ARKit pose (60 Hz); each reading is placed in the world
  with the pose interpolated at its exact time.
* **Automatic calibration:** with a wrong LiDAR→camera transform, the same wall measured at
  different moments lands in different places. The pipeline tries every 90° mounting
  orientation and refines the one that makes the point cloud crispest, together with
  ARKit's detected floor/wall planes (`pipeline/cleanscan/lidar.py`). In the simulation it
  recovers the mount to 0.04° and 2 mm with no hints.
* **People and animals:** readings that land on a masked person/animal in the nearest
  camera frame are dropped; readings that later passes see through are dropped by ghost
  removal, exactly like camera depth.
