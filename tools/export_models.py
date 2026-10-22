"""Export the on-device people/animal segmentation model for both apps.

    python tools/export_models.py              # yolo11n-seg (fast, ~6 MB Core ML / ~12 MB TFLite)
    python tools/export_models.py --model yolo11s-seg.pt   # more accurate, slower

Produces
    ios/Models/LivingSeg.mlpackage                       (Xcode compiles it to .mlmodelc)
    android/app/src/main/assets/living_seg.tflite

Tested toolchains (October 2026, Linux). The two exports need different torch versions,
so use two virtualenvs and pass --skip-ios / --skip-android:
    Core ML : python 3.12, torch==2.7.1, numpy<2.3, coremltools 9.0, ultralytics 8.4
              (coremltools fails on newer torch/numpy with "only 0-dimensional arrays ...")
    TFLite  : latest torch + ultralytics + litert-torch (ultralytics >= 8.4.83 exports
              format='tflite' through Google LiteRT)

Both are exported WITHOUT built-in NMS: the apps decode the raw outputs themselves
(YOLOSegDecoder.swift / YoloSegDecoder.kt) and keep only living-being classes
(person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe).
Check the Android model end to end with tools/verify_android_model.py.

License: Ultralytics YOLO models are AGPL-3.0. Shipping them inside an app means the app
must be AGPL-compatible, or you need an Ultralytics Enterprise license.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--skip-ios", action="store_true")
    ap.add_argument("--skip-android", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO

    if not a.skip_ios:
        out = Path(YOLO(a.model).export(format="coreml", imgsz=a.imgsz, nms=False, half=True))
        dest = ROOT / "ios" / "Models" / "LivingSeg.mlpackage"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out, dest)
        print("iOS model ->", dest)

    if not a.skip_android:
        out = Path(YOLO(a.model).export(format="tflite", imgsz=a.imgsz))  # float32 TFLite
        src = out if out.suffix == ".tflite" else next(out.rglob("*float32.tflite"))
        dest = ROOT / "android" / "app" / "src" / "main" / "assets" / "living_seg.tflite"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        print("Android model ->", dest)


if __name__ == "__main__":
    main()
