# Licences

| Part of this repository | Licence |
|---|---|
| Code: `ios/`, `android/`, `pipeline/`, `rig/`, `tools/` | **Apache-2.0** ([LICENSE](LICENSE)) |
| Depth-module firmware `module/firmware/` | MIT ([module/firmware/LICENSE](module/firmware/LICENSE)) |
| Depth-module clip and hardware design `module/hardware/` | CERN-OHL-P-2.0 ([module/hardware/LICENSE](module/hardware/LICENSE)) |

## ⚠ Read this before you distribute the apps

| Item | Licence | Where it is used | How to replace it |
|---|---|---|---|
| **Bundled YOLO11n-seg model**: `ios/Models/LivingSeg.mlpackage`, `android/app/src/main/assets/living_seg.tflite` | **AGPL-3.0** (Ultralytics) | on-device people/animal masking and object detection in both apps | Config, no code change: iOS `CleanScanSegModel` in `ios/project.yml`, Android `-Pcleanscan.segModel=<file>.tflite` (`android/app/build.gradle.kts`). Any model with the YOLO-seg output layout (exported with `nms=False`) and COCO class order works; `tools/export_models.py` exports Ultralytics models. Distributing the apps with the bundled model requires AGPL compliance or an Ultralytics Enterprise licence. |
| `ultralytics` Python package (optional `seg` extra, `cleanscan process --resegment`) | **AGPL-3.0** | optional desktop re-segmentation | Not installed by default; leave out `--resegment`. |
| Sideloader CLI (`.github/workflows/sideloader.yml`) | GPL-3.0 | a separate helper binary built from upstream source for Linux users; not linked into CleanScan | — |
| Arducam ToF SDK (`ArducamDepthCamera`, Pi depth-module variant only) | **verify** before redistributing (not checked here) | `rig/cleanscan_rig/depth.py`, installed by `install.sh arducam_tof` | use the ESP32-S3 + A010 module instead |
| ARCore SDK for Android | Google terms (Google APIs Terms of Service) | Android app | — |
| `usd-core` (optional `sim` / `dev` extras) | Tomorrow Open Source Technology License 1.0 (Pixar, Apache-2.0-based) | `cleanscan validate` USD checks only; USD files are written without it | leave it out: the USD check is skipped |

There are **no non-commercial-only models or datasets** in this repository. The optional
desktop open-vocabulary refinement planned in docs/RL_SCENES_PLAN.md (OWLv2 / Grounding DINO,
SAM 2, all Apache-2.0) is not implemented yet.

## Permissive dependencies

| Dependency | Licence | Used by |
|---|---|---|
| Open3D | MIT | pipeline (fusion, meshes, raycasting) |
| OpenCV (`opencv-python-headless`) | Apache-2.0 | pipeline |
| NumPy, SciPy, scikit-image | BSD-3-Clause | pipeline |
| PyYAML | MIT | pipeline (`materials.yaml`) |
| MuJoCo | Apache-2.0 | `cleanscan validate`, tests |
| CoACD (optional `collision` extra) | MIT | convex decomposition |
| vhacdx (optional `collision` extra) | BSD-3-Clause (V-HACD) | convex decomposition fallback |
| pyserial | BSD-3-Clause | rig |
| TensorFlow Lite, AndroidX, Kotlin | Apache-2.0 | Android app |
| Apple frameworks (ARKit, RealityKit, Core ML, Vision) | Apple SDK terms | iOS app |
| ESP-IDF, PlatformIO | Apache-2.0 | depth-module firmware (the Arduino core, LGPL-2.1, is deliberately not used) |
| Unity test framework | MIT | firmware host tests |
| pytest | MIT | tests |
