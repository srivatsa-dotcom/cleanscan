"""Export the on-device people/animal segmentation model for both apps.

    pip install ultralytics
    python tools/export_models.py              # yolo11n-seg (fast, ~3 MB)
    python tools/export_models.py --model yolo11s-seg.pt   # more accurate, slower

Produces
    ios/Models/LivingSeg.mlpackage                       (Xcode compiles it to .mlmodelc)
    android/app/src/main/assets/living_seg.tflite

Both are exported WITHOUT built-in NMS: the apps decode the raw outputs themselves
(YOLOSegDecoder.swift / YoloSegDecoder.kt) and keep only living-being classes
(person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe).
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
