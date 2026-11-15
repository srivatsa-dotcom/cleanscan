# CleanScan RL: scan a room → RL-ready simulation scene

Status: **plan, waiting for review**. Nothing below is implemented yet.

This plan extends CleanScan without removing anything. `cleanscan process` keeps
producing exactly what it does today. Everything new is additive: new files in the
capture, new CLI commands, new modules, new app screens.

## 0. What exists today (baseline)

| Area | State on `main` (dea8e91) | Reused for |
|---|---|---|
| `pipeline/` | masked TSDF fusion (`fusion.py`), ghost removal (`ghosts.py`), stereo depth for camera-only phones (`stereo.py`), LiDAR-rig calibration (`lidar.py`), Z-up floor alignment (`mesh.py`), texture baking, Gazebo export (`gazebo.py`), Nav2 map | the room, and every object capture (each capture is a normal CleanScan session) |
| tests | `pipeline/tests/test_pipeline.py`: **7** tests (the README says 5; camera-only and rig were added later), 77 s on 4 CPUs. `rig/tests`: 7 tests | must stay green |
| `rig/` | Pi bridge, UDP protocol `CSL1` on port 5600, NTP-style `ClockSync` | the depth module uses the same port, handshake and clock sync |
| iOS app | ARKit, YOLO11n-seg (living classes only), ARKit people segmentation, `RigClient` | detection + tracking, depth-module client |
| Android app | ARCore, the same YOLO model through TFLite. **No rig client yet** | needs a module client written from scratch (Kotlin port of `RigClient`) |
| CI | `ios.yml` → `ios-latest` release, `android.yml` → `android-latest` (debug APK), `sideloader.yml`. **No CI runs the Python tests** | extended |

## 1. Architecture

```
 PHONE (iOS 16+ / ARCore)                                   DESKTOP  `cleanscan process-rl`
 ┌──────────────────────────────────────────────┐           ┌──────────────────────────────────────────────┐
 │ 1 Room pass: RGB, 60 Hz pose, masks,         │           │ room   = existing pipeline (masked fusion,   │
 │   ToF frames from module (optional)          │           │          ghosts) + ToF/stereo depth           │
 │ 2 Live YOLO-seg on all 80 COCO classes +     │  zip      │ objects:                                      │
 │   IoU/3D tracker -> instance masks, tracks   │ ───────►  │   lift 2D tracks -> 3D region + pose          │
 │ 3 Selection screen (+ tap to add + name)     │           │   pickable     -> TSDF -> watertight -> ICP   │
 │ 4 Guided captures, same ARSession (shared    │           │   articulated  -> closed/open diff -> joint    │
 │   world frame): pickable / articulated /     │           │   transparent  -> silhouette -> revolution    │
 │   transparent                                │           │ carve objects out of room, physics props,     │
 │ 5 Export zip (format v1 + new files)         │           │ convex decomposition, settle on supports      │
 └──────────────────────────────────────────────┘           │ export: scene.json, USD, MJCF, SDF            │
        ▲ Wi-Fi "CleanScanDepth", UDP 5600                  └──────────────────────────────────────────────┘
 ┌──────┴──────────────────────┐                                           │ `cleanscan validate`
 │ ESP32-S3 + MaixSense A010   │                                           ▼  MuJoCo 5 s sim (+ pxr check)
 │ (or Pi Zero 2W + Arducam)   │                                         report.json
 └─────────────────────────────┘
```

**One world frame for everything.** Object captures run in the same `ARSession` /
ARCore session as the room pass (no `resetTracking`), so they share the room's world
frame. That gives free coarse registration for articulated and transparent objects,
which stay in place. Pickable objects are moved to a flat surface, so they need a real
registration step (coarse from the room-pass track, then ICP). Drift and relocalization
are handled by ICP refinement on the static surroundings.

### New pipeline modules (`pipeline/cleanscan/`)

| module | job |
|---|---|
| `objects.py` | read `tracks.json` / `selection.json` / `instances/`, lift 2D masks into the fused room (vote per TSDF surface point → object region, OBB, pose) |
| `pickable.py` | object capture → foreground (dominant plane removed + connected component at the orbit centre, or app masks) → masked TSDF → watertight mesh → registration to the room pose |
| `articulated.py` | align closed/open captures on the static body, find the moved part, fit revolute/prismatic joint, rebuild body + part meshes |
| `transparent.py` | silhouettes → metric radius profile → surface of revolution; cylinder fallback |
| `physics.py` | `materials.yaml` lookup, volume / mass / inertia (closed-form, tetrahedra), hollow shells, convex decomposition (CoACD → V-HACD → convex hull) |
| `scene.py` | scene bundle writer + `scene.json` schema, room carving, settling objects onto support surfaces |
| `export_mjcf.py`, `export_usd.py` | MuJoCo and USD writers (USD written as plain `.usda` text, no `pxr` needed to write) |
| `gazebo.py` (extended) | per-object models with inertia, collision pieces and joints. The existing room export stays unchanged |
| `validate.py` | MuJoCo checks, optional `pxr` checks, `report.json` |
| `depthmodule.py` | load ToF frames, calibrate the module→camera extrinsic (generalising `lidar.py`), RGB-guided upsampling → per-frame metric depth that feeds the existing TSDF |
| `refine/` | optional open-vocabulary detector + promptable segmenter plug-ins (off by default) |
| `synthetic_kitchen.py` | kitchen scene, captures, A010 simulation, ground truth |

## 2. Capture format additions (still `version: 1`, backward compatible)

Old pipelines ignore files they don't know. New files only:

```
Scan-…/
  (all v1 files unchanged: session.json, frames.jsonl, color/, depth/, mask/, poses.bin, …)
  instances/000000.png     # 16-bit, 0 = background, k = track id k (depth resolution), room pass
  tracks.json              # [{id, class, coco_id, score, frames, first_frame, thumb, approx_position_world, approx_size}]
  thumbs/<id>.jpg          # crop shown on the selection screen
  selection.json           # what the user picked (below)
  captures/<obj_id>/<stage>/     # a complete v1 session each (own session.json, frames.jsonl, …)
       stage = scan (pickable) | closed, open (articulated) | profile, top (transparent)
       + object_mask/000000.png  # 8-bit, 255 = this object (optional; from on-device seg)
  tof.jsonl                # one record per depth-module frame: {t, path, intrinsics, size, sensor}
  tof/000000.png           # 16-bit mm, 0 = no return (A010: 100x100, Arducam: 240x180)
  depth_module.json        # {sensor, firmware, fov_deg, clock_rtt_ms, extrinsic?: 16 numbers}
```

`session.json` gains an optional `"extensions": ["objects", "depth_module"]`. The
synthetic generator also writes `truth.json` (tests only).

`selection.json`:

```json
{"objects": [
  {"id": 3, "class": "refrigerator", "name": "fridge", "mode": "articulated",
   "source": "detected", "captures": {"closed": "captures/3/closed", "open": "captures/3/open"}},
  {"id": 101, "class": "cupboard", "name": "cupboard", "mode": "articulated",
   "source": "tap", "seed": {"frame": 57, "u": 0.41, "v": 0.63, "world": [x, y, z]}},
  {"id": 7, "class": "bottle", "mode": "pickable", "captures": {"scan": "captures/7/scan"}},
  {"id": 9, "class": "wine glass", "mode": "transparent",
   "captures": {"profile": "captures/9/profile", "top": "captures/9/top"}}
]}
```

Default class → mode table (shared by the app and the pipeline, overridable):
pickable = bottle, cup, bowl, box, utensil, fork, knife, spoon, book, vase…;
articulated = refrigerator, cupboard, cabinet, drawer, oven, microwave, door, dishwasher;
transparent = wine glass, glass, tumbler (steel tumbler: reflective, same silhouette path).

## 3. Scene bundle (`cleanscan process-rl … -o <scene>`)

```
<scene>/
  scene.json        # manifest (below)
  room/             # existing `process` output unchanged inside (models/, worlds/, maps/, …),
                    # built from the carved room mesh; + room/collision_boxes.json for MuJoCo
  objects/<id>/     # visual.obj/.mtl/_texture.png, collision/000.obj…, joint.json (articulated)
                    # articulated: links/<link>/visual.obj + collision/
  export/scene.usda # Isaac Sim / Isaac Lab
  export/scene.xml  # MuJoCo
  export/world.sdf  # Gazebo Sim (+ models/), run_gazebo.sh
  export/isaaclab_scene.py   # example InteractiveSceneCfg (untested, see §8)
  report.json       # quality metrics, warnings, validation results
scene.zip           # the same folder, zipped
```

`scene.json` (units = m, Z up, floor at z = 0, quaternions wxyz):

```json
{"format": "cleanscan-scene", "version": 1, "units": "m", "up": "z",
 "room": {"visual": "room/…/visual.obj", "collision": "room/…/collision.stl",
          "collision_boxes": "room/collision_boxes.json", "map": "room/maps/room.yaml"},
 "objects": [
  {"id": "bottle_7", "class": "bottle", "type": "pickable", "material": "plastic",
   "pose": {"position": [x, y, z], "quaternion": [w, x, y, z]},
   "mass": 0.031, "com": [0, 0, 0.09], "inertia": [[…], […], […]],
   "friction": 0.5, "restitution": 0.1, "hollow": true, "shell_thickness": 0.0006,
   "visual": "objects/bottle_7/visual.obj", "collision": ["objects/bottle_7/collision/000.obj"],
   "overrides": {}},
  {"id": "fridge_3", "type": "articulated", "fixed_base": true,
   "links": [{"name": "body", …}, {"name": "door", "mass": …, "visual": …, "collision": […]}],
   "joints": [{"name": "door_hinge", "type": "revolute", "parent": "body", "child": "door",
               "axis": [0, 0, 1], "origin": [x, y, z], "lower": 0.0, "upper": 1.75,
               "damping": 0.5, "friction": 0.05}]}]}
```

Anything under `overrides` (mass, friction, material, joint limits, type) wins, and
`cleanscan export <scene>` re-writes USD/MJCF/SDF from an edited `scene.json` without
reprocessing.

### Simulator mapping

| | Isaac (USD, UsdPhysics) | MuJoCo (MJCF) | Gazebo (SDF) |
|---|---|---|---|
| room | static `Mesh` + `CollisionAPI`, `MeshCollisionAPI approximation="none"` (triangle mesh) | **floor plane + boxes** (see note) | triangle mesh (existing) + floor plane |
| pickable | `RigidBodyAPI`, `MassAPI` (mass, `diagonalInertia`, `principalAxes`, `centerOfMass`), convex pieces | body + `freejoint`, one mesh geom per convex piece, `inertial` explicit | model with `inertial`, collision pieces |
| articulated | `ArticulationRootAPI`, body fixed to world by `FixedJoint`, `RevoluteJoint`/`PrismaticJoint` with limits (deg / m) | body welded to world, child with `hinge`/`slide`, `range`, `damping` | model, `<joint type="fixed">` to world + `revolute`/`prismatic` |
| materials | `PhysicsMaterialAPI` (static/dynamic friction, restitution) | `friction`, `solref` | `<surface><friction>` |

**MuJoCo note:** MuJoCo collides every mesh geom as its convex hull, so a whole-room
triangle mesh cannot be used for collision. The MJCF room collision is the floor plane plus
boxes from greedy merging of the occupied voxels of the carved room (about 2 cm). Flat
supports such as the counter top become exact boxes because they are fitted from horizontal
plane segments. The visual mesh is the full textured room.

## 4. Algorithms

**Room + carving.** `pipeline.run` is refactored into `reconstruct_room()` (internal,
same steps) so `process` output stays byte-for-byte equivalent (guarded by a test that
compares mesh stats and file lists). In `process-rl`, after objects are registered,
room triangles inside each object's collision hull dilated by 2 cm are removed. The
footprints left under pickables, never seen by the room pass, are patched with a flat cap at
the measured support height. Larger holes (behind the fridge) stay open.

**3D lifting of tracks.** For each room-pass frame, every pixel of instance k votes
for the fused surface point it hits (raycast of the fused room mesh, depth test against
the frame depth). A surface point belongs to object k if `votes_k / visible_count > 0.5`.
That gives the object's region, its OBB (PCA, yaw-only for upright objects) and its
centroid. Tap-added objects start from the seed point and grow over the connected surface
region bounded by the support planes. For articulated objects the closed/open diff
defines the region more reliably anyway.

**Optional desktop refinement (off by default).** A `Refiner` plug-in interface:
`detect(rgb, prompts) -> boxes` and `segment(rgb, box|point) -> mask`. Shipped adapters:
OWLv2 or Grounding DINO (Apache-2.0) for detection, SAM 2 (Apache-2.0) for segmentation.
They are enabled with `--refine owlv2+sam2` and only if `transformers` and the weights are
present, otherwise logged and skipped. YOLO-World is not used by default (Ultralytics,
AGPL).

**Pickable.** Capture session → depth (sensor, ToF-upsampled, or stereo) → fuse at 5 mm
voxels with a bounded volume around the orbit centre (the camera rays' closest-approach
point) → remove the support plane (RANSAC) → keep the largest component above it → close
the bottom (the object stood on it) with a planar cap → watertight check (fallback: Poisson
on the points, or the convex hull with a warning). Registration: candidate poses = room
OBB centroid × 36 yaw angles, scored by the chamfer distance to the room-pass object points,
then point-to-plane ICP. Upright objects (axis ∥ gravity) only search yaw. Rotationally
symmetric objects are allowed any yaw, since it doesn't matter for physics.

**Articulated.** (1) ICP the open capture onto the closed one using only points far from
the change region (iterative trimmed ICP). (2) Change detection: points of each state
farther than 2 cm from the other state's surface = *disappeared* (closed part) and
*appeared* (open part + newly visible interior). (3) Hypothesis search on the disappeared
part P_c: revolute about each edge of P_c's oriented rectangle (4 hinge candidates,
angle 0–180° both ways), prismatic along ±P_c's normal (0–0.8 m). Score = robust one-sided
chamfer of transformed P_c to the open cloud. (4) Continuous least-squares refinement of
(axis, origin, angle) or (axis, distance). A rotation over 5° means revolute, otherwise
prismatic. Limits: `[0, observed]`, slightly extended (5 %) and capped at the observed opening
if extending would collide. (5) Meshes: the part is fused in its own frame from both captures
(open-capture camera poses are mapped through the inverse joint motion, so the part is
static). The body is fused from both captures with the part pixels removed, so the interior
seen while open becomes part of the body.

**Transparent.** Side-profile frames (phone level, checked from the gravity vector):
per row, the silhouette half-width in pixels. The metric scale comes from the camera-to-axis
distance. That distance is the distance to where the object's base ray meets the support
plane (ARKit plane or room floor/counter), so no depth sensor is needed. Several frames
are combined into a median radius profile r(z), then a lathe into a surface of revolution
(closed bottom, open top with wall thickness). If fewer than N good frames are available,
or the profile is noisy, it falls back to a cylinder with the measured height and the
maximum diameter. The top-down shot gives an independent diameter check, recorded in the
report.

**Physics.** `materials.yaml` (packaged, overridable with `--materials`):

```yaml
materials:
  glass:    {density: 2500, friction: 0.4, restitution: 0.1}
  ceramic:  {density: 2400, friction: 0.5}
  steel:    {density: 7850, friction: 0.4}
  plastic:  {density: 1000, friction: 0.5}   # PET ~1380; bottles are thin shells
  wood:     {density: 600,  friction: 0.6}
  laminate: {density: 700,  friction: 0.6}
classes:
  bottle:       {material: plastic, hollow: true, shell_thickness: 0.0008, fill: 0.0}
  wine glass:   {material: glass,   hollow: true, shell_thickness: 0.0015}
  cup:          {material: ceramic, hollow: true, shell_thickness: 0.004}
  refrigerator: {material: steel,   hollow: true, shell_thickness: 0.002,
                 links: {door: {hollow: true, shell_thickness: 0.003}}}
  cupboard:     {material: wood,    hollow: true, shell_thickness: 0.016}
default: {material: plastic, hollow: false}
```

Solid: m = ρ·V, inertia by exact tetrahedral integration over the watertight mesh.
Hollow: thin-shell integration over the surface (m = ρ·A·t), plus optional liquid fill
(fill fraction × inner volume × 1000 kg/m³). Inertia is reported about the COM, in the body
frame, and also diagonalised for USD's `principalAxes`.

**Collision.** CoACD (MIT, `pip install coacd`) → V-HACD through `vhacdx` (BSD) → convex
hull. Each step is tried only if the package is installed. The report says which one ran.

**Settling.** After registration, each pickable is dropped onto its support surface with
a 1 mm clearance, so t = 0 has no interpenetration. The vertical correction is reported. If
it is larger than 1.5 cm, the report flags a registration problem.

## 5. Validation (`cleanscan validate <scene>`)

MuJoCo, timestep 2 ms, 5 s:

| check | failure condition |
|---|---|
| finite | any NaN/Inf in `qpos`/`qvel` |
| fall-through | a pickable ends below its support surface by more than 2 cm, or below z = −0.05 |
| at rest | a pickable moves more than 2 cm (t = 0 → 5 s) with no external force |
| interpenetration at t = 0 | any contact with penetration deeper than 3 mm between different bodies (parent and child links of a joint excluded) |
| joint limits reachable | each joint, driven alone by a bounded generalized force for 2 s towards each limit, reaches it within 5 % (or 2°/1 cm), checked from both ends |

USD (when `pxr` / `usd-core` is installed): stage opens, `metersPerUnit = 1`, `upAxis = Z`,
every rigid body has positive mass and collision, joint `body0`/`body1` targets exist,
and every articulation has exactly one root. Results go to `report.json → validation`, and
the exit code is non-zero on failure.

## 6. Depth module (`module/`)

**Hardware:** ESP32-S3 (DevKitC-1 or XIAO ESP32S3) + Sipeed MaixSense A010 (100×100 ToF,
UART), powered from a USB power bank. **Firmware:** PlatformIO with the **ESP-IDF** framework
(Apache-2.0; this avoids linking the LGPL-2.1 Arduino core). It runs a soft AP named
`CleanScanDepth`, with the password set at build time or over USB serial (NVS). UDP 5600
uses the rig's JSON control messages (`hello`/`info`/`ping`/`pong` with the same clock-sync
maths) plus a new frame packet:

```
"CSD1" | u8 kind (3 = depth frame) | u8 fmt (1 = u16 mm) | u16 width | u16 height
| u32 frame_id | u16 chunk | u16 n_chunks | f64 t_capture (module clock) | payload (≤ 1360 B)
```

A 100×100 u16 frame takes 15 chunks. The app reassembles frames, drops incomplete ones,
converts the time to the phone clock and writes `tof/` + `tof.jsonl`. The A010 driver
configures the sensor over UART (`AT+…`: higher baud rate, binning 1, frame rate, unit) and
parses its frame packets. Its 8-bit depth codes are converted to mm on the ESP32.
Protocol/parsing code lives in a portable C++ library, unit-tested natively (`pio test -e
native`) and cross-checked against the Python decoder in CI.
**Arducam option:** a new `arducam_tof` sensor in `rig/cleanscan_rig/sensors.py`
(240×180, via Arducam's SDK) sends the same `CSD1` frames from a Pi Zero 2 W. It is set up
with `install.sh arducam_tof`.

**Docs/hardware:** `module/README.md` (wiring, flashing, LED states, troubleshooting),
`module/hardware/clip.scad` (parametric clip: phone thickness/width, A010 and ESP32 board
mounts, the ToF lens kept close to the phone camera), `module/hardware/BOM.md`, and
`module/LICENSE` files (firmware MIT, hardware CERN-OHL-P-2.0).

**Pipeline:** `depthmodule.py` reads ToF frames and estimates the module→camera
extrinsic by generalising `lidar.py`'s crispness objective. ToF frames are just dense
"LiDAR" points with phone-clock timestamps, so the same `_Problem` (90° orientation
candidates, neighbourhood plane residuals between different times, ARKit planes) applies.
A small ToF→camera time offset is also refined (±30 ms). **Upsampling:** ToF points are
projected into each colour frame (pose interpolated at the ToF timestamp), then an
RGB-guided densification at depth-map resolution (joint-bilateral normalised convolution,
then a guided filter). The ToF samples are kept as hard metric anchors, and results are
restricted to the ToF field of view plus a small margin. The rest of the frame falls back to
stereo depth when available. The resulting `depth_est/` frames feed the unchanged TSDF.

## 7. Apps

**Shared:** a class → mode table, the 80-class COCO label list, and a tracker (greedy IoU
on boxes plus 3D distance when depth is available, tracks expire after 1.5 s unseen).

**iOS (SwiftUI):** the decoder runs on all classes. Living-being masking keeps its current
class set. New views: `ObjectSelectionView` (grouped list, thumbnails, checkboxes, a tap on a
keyframe adds an object with a typed name, a mode picker per object) and `GuidedCaptureView`.
For pickables it shows a coverage ring (36 azimuth bins around the object centre from an
ARKit raycast, done at ≥ 80 %). For articulated objects it walks through "Scan it closed" →
"Open it fully" → "Scan again". For transparent objects it shows a level indicator from the
gravity vector, then asks for the top-down shot. `DepthModuleClient` generalises `RigClient`
and adds frame reassembly. Settings: "External depth module". **Discovery:** unicast `hello`
to known addresses (the Wi-Fi gateway, `192.168.4.1`, `10.42.0.1`). Broadcast or multicast
discovery on iOS needs Apple's multicast entitlement, which a free-Apple-ID sideload cannot
get.
**Android (Kotlin, Views):** the same screens as Activities, a TFLite decoder for all classes,
a `DepthModuleClient` (UDP broadcast discovery with a `MulticastLock`, plus the gateway
probe), and writing the same files.

## 8. What can and cannot be verified here

| verified in CI / this environment | needs testing on a real phone or hardware |
|---|---|
| all Python: synthetic kitchen end to end, MuJoCo validation, USD validation with `usd-core` | anything the apps do at runtime: detection quality, tracking, coverage ring, ARKit/ARCore behaviour, sharing |
| iOS and Android **compile** (CI builds) | real ToF noise and A010 UART settings (baud rate, unit encoding), Wi-Fi throughput |
| firmware **compiles** for ESP32-S3; protocol/parser unit tests run natively | flashing and running on an ESP32-S3; the Arducam SDK on a Pi |
| scene files load in MuJoCo and pass `pxr` checks | loading in Isaac Sim / Isaac Lab and Gazebo (not installable here). The files follow the schemas, and an example Isaac Lab config is included |

Synthetic results are an upper bound: real scans will be worse, and the docs will say so.

## 9. Tests

* All existing tests stay green: 7 pipeline tests, 7 rig tests. A new
  `test_process_unchanged` checks that `process` output (file list, triangle counts) does not
  change.
* `test_kitchen_e2e.py`: `synth-kitchen` (iPhone-11-style camera-only phone + simulated
  A010, person walking through) → `process-rl` → `validate`, asserting: fridge door
  revolute and drawer prismatic, axis error < 5°, origin error < 3 cm (revolute: distance
  between the axis lines; prismatic: origin isn't defined, so the check is axis + limit
  error < 3 cm), pickable position error < 2 cm, masses within 30 % of ground truth, MuJoCo
  validation passes, USD validates if `pxr` is installed (it is installed in CI).
* Unit tests: inertia of analytic solids, joint fitting on clean synthetic transforms,
  surface of revolution, MJCF/USD/SDF parse, `CSD1` encode/decode round trip, A010 parser.
* CI time budget: the kitchen test is kept under about 6 min on a GitHub runner (low
  resolution, about 120 room frames, about 40 frames per capture).

## 10. CI and releases

* `ios.yml`, `android.yml`: run on every push to `main` (no path filter, as requested;
  note that macOS minutes cost 10×). Assets are uploaded with `--clobber` to a rolling
  `latest` release (created if missing, tag moved to the commit): `CleanScan-unsigned.ipa`,
  `CleanScan.apk`. The old `ios-latest` / `android-latest` releases keep being updated so
  existing links and the Sideloadly / Linux Sideloader steps keep working (see question 2).
* APK signing: the debug key differs on every runner, so updating needs an uninstall
  first (the same problem exists today). If a repository secret `ANDROID_KEYSTORE_B64` exists,
  CI signs a release APK with it instead. Documented.
* New `pipeline.yml`: rig and pipeline tests (Python 3.11 and 3.12, with `mujoco` and
  `usd-core`), firmware build (`pio run -e esp32s3`) plus native firmware tests, and
  publishing `cleanscan-depth-esp32s3.bin` (+ merged `-factory.bin` for one-shot flashing with
  `esptool`) to `latest`. It also writes the synthetic kitchen run's renders (MuJoCo snapshot,
  before/after images) used in the README.

## 11. Licensing (`LICENSES.md`)

| item | license | flag | swap via |
|---|---|---|---|
| bundled YOLO11n-seg (`LivingSeg.mlpackage`, `living_seg.tflite`) and `ultralytics` (optional `seg` extra, `--resegment`) | **AGPL-3.0** | ⚠ app distribution | app `ModelConfig` (file name, class list, output layout); `--resegment MODEL`. Candidate permissive replacement: RTMDet-Ins (Apache-2.0), **not yet exported or tested** |
| Sideloader CLI (built by `sideloader.yml`) | GPL-3.0 | separate tool, not linked | — |
| Arduino-ESP32 core | LGPL-2.1 | avoided: firmware uses ESP-IDF | — |
| Arducam ToF SDK | to verify | ⚠ check before release | `--sensor` |
| Open3D (MIT), OpenCV (Apache-2.0), NumPy/SciPy (BSD), MuJoCo (Apache-2.0), usd-core (Tomorrow Open Source Technology License, Apache-2.0-based), CoACD (MIT), vhacdx (BSD-3), trimesh (MIT), OWLv2 / Grounding DINO / SAM 2 (Apache-2.0), TFLite (Apache-2.0), ESP-IDF (Apache-2.0) | permissive | — | — |
| ARCore SDK | Google APIs terms | note | — |

No non-commercial weights are used. Every model is chosen through config.

## 12. Milestones (each: tests green, a report back to you, small commits)

| # | milestone | deliverable | test gate |
|---|---|---|---|
| 1 | scene bundle + MJCF/USD export, existing room | `process-rl` on an old session (room only) + objects declared by hand in `scene.json`; `cleanscan export` | existing 14 + `process` unchanged + export parse tests |
| 2 | synthetic kitchen + validation | `synth-kitchen`, `validate`, GT objects inserted directly → validation passes | validation unit tests (good scene passes, broken scenes fail each check) |
| 3 | pickable objects | track lifting, capture → mesh → registration, physics, carving, settling | bottle pose < 2 cm, mass ±30 % |
| 4 | articulated objects | closed/open → joint fit, links | axis < 5°, origin < 3 cm |
| 5 | transparent objects | silhouette lathe + fallback | glass mass ±30 %, pose < 2 cm |
| 6 | app screens (iOS + Android) | detection on all classes, tracker, selection, guided capture, new files | CI compiles both; format writer checked against the Python loader with a fixture |
| 7 | depth module | firmware, Arducam sensor, app clients, `depthmodule.py` calibration + upsampling, A010 simulation | extrinsic recovered (< 1°, < 1 cm) on synthetic data; full kitchen E2E on camera-only + A010 |
| 8 | CI release + docs | workflows, README "Install in 5 minutes" + "Scan your kitchen for RL" with images from the synthetic run, `LICENSE`, `LICENSES.md` | all workflows green on the PR |

Ordering note: the E2E test asks for the iPhone 11 + A010 configuration, which arrives in
milestone 7. Milestones 3–5 therefore run the kitchen with the simulated sensor depth
(the "LiDAR phone" path) first, and milestone 7 switches the E2E test to camera-only + A010
and keeps the sensor-depth variant as a second, faster test.

## 13. Risks

* **iPhone 11 without the module:** small objects reconstructed from stereo alone will
  often be incomplete. The pipeline then falls back to the room-pass OBB (box or cylinder by
  class) and says so in the report. The module is the intended setup for pickables.
* **Pickables are often only partly visible in the room pass**, which makes registration
  ambiguous. The pipeline uses the support plane + yaw search, and symmetric objects ignore yaw.
* **A010 specifics** (UART speed needed for full frame rate, unit encoding) come from
  Sipeed's docs and are implemented but not verified on hardware.
* **Doors opened less than about 30°** make axis estimation weak, so the app asks the user to
  open them fully. Below a 10° opening, the pipeline falls back to the class prior (hinge at
  the nearest vertical edge) and warns.
