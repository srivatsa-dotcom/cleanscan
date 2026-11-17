# Depth module: parts list

| Part | Notes | Approx. price |
|---|---|---|
| Sipeed **MaixSense A010** ToF camera | 100 x 100 px, 0.2-2.5 m, UART + USB | USD 25-35 |
| **ESP32-S3** board: ESP32-S3-DevKitC-1 (N8R2 or bigger) or Seeed XIAO ESP32S3 | Wi-Fi + 2 Mbit/s UART | USD 8-15 |
| 4 female-female jumper wires (or a 4-pin JST-GH cable for the A010) | | |
| USB-C power bank (5 V, 1 A is plenty) + cable | powers both boards (about 1.5 W) | |
| 4 x M2.5 x 6 screws | A010 on the clip | |
| 3D-printed clip (`clip.scad`) | PETG or PLA, ~25 g | |

Alternative (reuses `rig/`): Raspberry Pi Zero 2 W + **Arducam ToF camera** (240 x 180,
CSI) instead of the ESP32 and the A010: `sudo ./setup/install.sh arducam_tof` (rig/README.md).
