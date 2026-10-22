# CleanScan capture format (v1)

Both phone apps write the same folder layout, which the desktop pipeline reads.
The app shares it as `Scan-YYYYMMDD-HHMMSS.zip`; `cleanscan process` accepts the zip directly.

```
Scan-20261022-101500/
  session.json        # metadata (below)
  frames.jsonl        # one JSON object per recorded frame
  color/000000.jpg    # RGB, sensor (landscape) orientation, full camera resolution
  depth/000000.png    # 16-bit grayscale, millimetres, 0 = no data
  confidence/000000.png  # 8-bit, 0..confidence_levels-1 (optional)
  mask/000000.png     # 8-bit, 255 = person/animal, 0 = static scene (depth resolution)
```

## session.json

| key | meaning |
| --- | --- |
| `format` | always `"cleanscan-capture"` |
| `version` | `1` |
| `platform` | `"ios"`, `"android"` or `"synthetic"` |
| `depth_unit` | metres per depth PNG unit (`0.001`) |
| `confidence_levels` | 3 on iOS (ARConfidenceLevel low/medium/high), 256 on Android |
| `recommended_min_confidence` | optional 0..1 default for `--min-confidence` |
| `world_up` | `"y"` - ARKit/ARCore worlds are gravity aligned, +Y up |
| `camera_convention` | `"opengl"` - camera +X right, +Y up, looks down -Z |

## frames.jsonl record

| key | meaning |
| --- | --- |
| `index`, `timestamp` | frame number, seconds |
| `color`, `depth`, `confidence`, `mask` | paths relative to the session folder |
| `color_intrinsics` | `[fx, fy, cx, cy]` in pixels of the colour image |
| `color_size` | `[w, h]` of the colour image |
| `depth_intrinsics` | `[fx, fy, cx, cy]` in pixels of the depth image |
| `depth_size` | `[w, h]` of the depth / confidence / mask images |
| `camera_to_world` | 16 numbers, row-major 4x4 pose of the sensor-oriented camera |

ARKit (`ARCamera.transform`) and ARCore (`Camera.getPose()`) both use exactly this
convention, so no per-platform conversion is needed in the pipeline. The pipeline
converts to the Gazebo/ROS convention (Z up, floor at z = 0) itself.
