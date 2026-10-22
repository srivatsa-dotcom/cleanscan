"""End-to-end check of the Android segmentation path with the real exported model.

    (cd tools/decoder-test && ./gradlew -q installDist)
    python tools/verify_android_model.py IMAGES_DIR android/app/src/main/assets/living_seg.tflite \
        yolo11n-seg.pt tools/decoder-test/build/install/decoder-test/bin/decoder-test

Needs: pip install ultralytics ai-edge-litert opencv-python-headless

sensor image (landscape) --[app: rotate 90 CW + resize, simulated here]--> Kotlin packInput
--> real TFLite model --> Kotlin Outputs/decode/rasterize --> landscape depth-res mask,
compared with the Ultralytics reference segmentation of the same picture.
"""
import subprocess, sys
from pathlib import Path
import cv2, numpy as np
from ai_edge_litert.interpreter import Interpreter
from ultralytics import YOLO

E2E = Path(sys.argv[1]); TFL = sys.argv[2]; PT = sys.argv[3]; JVM = sys.argv[4]
LIVING = [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
DW, DH = 256, 192
it = Interpreter(TFL); it.allocate_tensors()
inp = it.get_input_details()[0]; outs = it.get_output_details()
ref_model = YOLO(PT)

def jvm(*args):
    r = subprocess.run([JVM, *map(str, args)], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr)
    return r.stdout.strip()

for img_path in sorted(E2E.glob("*.jpg")):
    upright = cv2.imread(str(img_path))                     # what a person sees (portrait)
    sensor = cv2.rotate(upright, cv2.ROTATE_90_COUNTERCLOCKWISE)  # what the phone camera buffer holds
    # --- app (LivingSegmenter.segment): rotate 90 CW, scale to model size (Bitmap ops simulated)
    shape = inp["shape"].tolist()
    H, W = (shape[2], shape[3]) if shape[1] == 3 else (shape[1], shape[2])
    up = cv2.rotate(sensor, cv2.ROTATE_90_CLOCKWISE)
    rgb = cv2.cvtColor(cv2.resize(up, (W, H), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    argb = (0xFF << 24) | (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2]
    argb.astype(">u4").tofile(E2E / "argb.bin")
    jvm("pack", E2E / "argb.bin", ",".join(map(str, shape)), E2E / "input.bin")
    x = np.fromfile(E2E / "input.bin", ">f4").reshape(shape).astype(np.float32)
    it.set_tensor(inp["index"], x); it.invoke()
    det = next(o for o in outs if len(o["shape"]) == 3); pro = next(o for o in outs if len(o["shape"]) == 4)
    it.get_tensor(det["index"]).astype(">f4").tofile(E2E / "det.bin")
    it.get_tensor(pro["index"]).astype(">f4").tofile(E2E / "proto.bin")
    msg = jvm("decode", E2E / "det.bin", ",".join(map(str, det["shape"])), E2E / "proto.bin",
              ",".join(map(str, pro["shape"])), ",".join(map(str, shape)), DW, DH, E2E / "mask.bin")
    kmask = np.fromfile(E2E / "mask.bin", np.uint8).reshape(DH, DW) > 0

    # --- reference: Ultralytics on the same stretched upright image, mapped to the sensor frame
    stretched = cv2.resize(up, (W, H), interpolation=cv2.INTER_LINEAR)
    r = ref_model.predict(stretched, imgsz=W, conf=0.3, classes=LIVING, retina_masks=True, verbose=False)[0]
    ref = np.zeros((H, W), bool)
    if r.masks is not None:
        for m in r.masks.data.cpu().numpy():
            ref |= cv2.resize(m, (W, H)) > 0.5
    ref_up = cv2.resize(ref.astype(np.uint8), (up.shape[1], up.shape[0]), interpolation=cv2.INTER_NEAREST)
    ref_sensor = cv2.rotate(ref_up, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ref_d = cv2.resize(ref_sensor, (DW, DH), interpolation=cv2.INTER_NEAREST) > 0
    iou = (kmask & ref_d).sum() / max((kmask | ref_d).sum(), 1)
    names = [ref_model.names[int(c)] for c in r.boxes.cls] if r.boxes is not None else []
    print(f"{img_path.name:18s} kotlin {msg} | reference {names} | masked {kmask.mean()*100:.1f}% | IoU {iou:.3f}")

    # visual: sensor image with Kotlin mask in red
    vis = cv2.resize(sensor, (DW * 2, DH * 2))
    km = cv2.resize(kmask.astype(np.uint8), (DW * 2, DH * 2), interpolation=cv2.INTER_NEAREST) > 0
    vis[km] = (0.4 * vis[km] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.imwrite(str(E2E / f"vis_{img_path.stem}.png"), cv2.rotate(vis, cv2.ROTATE_90_CLOCKWISE))
