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
  poses.bin           # optional: every ARKit pose (~60 Hz)
  lidar.bin           # optional: external LiDAR rig readings (rig/README.md)
  rig.json            # optional: rig sensor description, written when lidar.bin exists
  planes.json         # optional: ARKit plane anchors at the end of the scan
  instances/000000.png  # optional (RL scenes): 16-bit object labels, value = track id, 0 = none
  tracks.json         # optional (RL scenes): objects the phone's detector tracked
  thumbs/<id>.jpg     # optional (RL scenes): best crop of each tracked object
  selection.json      # optional (RL scenes): objects the user made interactive
  captures/<id>/<stage>/  # optional (RL scenes): one guided capture = a complete session
                          # (+ object_mask/000000.png: 255 = the object, depth resolution)
  tof/000000.png      # optional: clip-on depth module frames, 16-bit mm, 0 = no data
  tof.jsonl           # optional: one record per module frame (below)
  depth_module.json   # optional: module description, written when tof/ exists
```

`depth/` and `confidence/` are absent for camera-only captures (phones without LiDAR/ToF);
`frames.jsonl` then has no `depth` key and the pipeline computes depth by stereo.

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

## High-rate streams (little-endian, packed)

| file | record | size |
| --- | --- | --- |
| `poses.bin` | f64 time (phone clock, same as `timestamp`) + 16 x f32 camera_to_world, row-major | 72 B |
| `lidar.bin` | f64 time (phone clock) + f32 angle (rad, counter-clockwise in the sensor frame) + f32 range (m, 0 = none) + u8 intensity | 17 B |

LiDAR sensor frame: x = sensor front (the 1D beam), y = left, z = up; a 2D sensor scans
its x-y plane. The LiDAR->camera transform is found automatically by the pipeline, or can
be given as `"extrinsic"` (16 numbers, row-major, LiDAR -> OpenGL camera) in `rig.json`.

`planes.json`: list of `{"transform": 16 numbers row-major, "center": [x,y,z],
"extent": [x, 0, z], "alignment": "horizontal"|"vertical"}`; the plane lies in its local
x-z plane, normal = local +y (ARKit convention).

## RL-scene additions (optional, backwards compatible)

Written by the apps' object selection and guided capture (docs/RL_SCENES_PLAN.md); old
pipelines ignore them, `cleanscan process-rl` uses them. `session.json` may list
`"extensions": ["objects"]`; `frames.jsonl` records may have `instances` / `object_mask` paths.

`tracks.json`: `[{"id", "class", "coco_id", "score", "frames", "first_frame", "best_frame", "thumb"}]`.

`selection.json`:

```json
{"objects": [
  {"id": 2, "class": "bottle", "name": "bottle", "mode": "pickable", "source": "detected",
   "captures": {"scan": "captures/2/scan"}},
  {"id": 101, "class": "cupboard", "name": "Cupboard", "mode": "articulated", "source": "tap",
   "captures": {"closed": "captures/101/closed", "open": "captures/101/open"},
   "seed": {"frame": 57, "u": 0.41, "v": 0.63}}
]}
```

`mode`: `pickable` (stage `scan`), `articulated` (`closed`, `open`) or `transparent`
(`profile`, `top`). `seed`: the tap that added an object the detector missed: room-pass frame
index and position in 0..1 of the sensor (landscape) image; the pipeline turns it into a 3D
point with that frame's depth. An object whose `captures` is empty was skipped and stays part
of the static room. Guided captures run in the same AR session as the room pass, so all poses
share one world frame.

## Depth module (optional)

Written by the apps when the clip-on depth module (module/README.md) is on, in the room pass
and in every guided capture (each with its own `poses.bin`, which the pipeline needs to place
module frames between camera frames).

`tof.jsonl` record: `{"index", "t", "depth", "intrinsics": [fx, fy, cx, cy], "size": [w, h]}`;
`t` is on the phone clock (same as `timestamp` in frames.jsonl), converted from the module's
clock with the ping/pong sync. Module images use OpenCV camera axes (x right, y down, z forward).

`depth_module.json`: `{"sensor": "a010", "fov_deg": [70, 60], "depth_unit": 0.001,
"frame": "opencv", "clock_rtt_ms", "frames"}`, optionally `"extrinsic"` (16 numbers, row-major,
module -> phone camera in OpenGL axes) to skip the automatic calibration.
