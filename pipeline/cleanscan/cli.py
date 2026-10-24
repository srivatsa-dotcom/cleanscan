"""Command line: ``cleanscan process SESSION -o OUT`` (see README)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path

from .fusion import FusionConfig
from .pipeline import PipelineConfig, run
from .session import Session
from .texture import TextureConfig


def _unpack(path: Path) -> Path:
    """Accept the .zip the phone app shares as well as an extracted folder."""
    if path.suffix.lower() != ".zip":
        return path
    dest = path.with_suffix("")
    if not dest.exists():
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
    inner = [p for p in dest.rglob("session.json")]
    return inner[0].parent if inner else dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cleanscan", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="session -> clean mesh, Gazebo world, nav map")
    p.add_argument("session", type=Path, help="capture folder or .zip exported by the app")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--name", default=None, help="Gazebo model name (default: folder name)")
    p.add_argument("--voxel", type=float, default=0.02, help="TSDF voxel size in m (0.01 = finer, slower)")
    p.add_argument("--max-depth", type=float, default=5.0)
    p.add_argument("--min-confidence", type=float, default=None,
                   help="0..1 depth confidence cut-off (default: the value recommended by the app, else 0.5)")
    p.add_argument("--mask-dilate", type=int, default=4, help="grow people/animal masks by N depth px")
    p.add_argument("--no-ghost-removal", action="store_true")
    p.add_argument("--resegment", metavar="MODEL", default=None,
                   help="re-run people/animal segmentation on the desktop, e.g. yolo11x-seg.pt")
    p.add_argument("--texture", choices=["photo", "none"], default="photo")
    p.add_argument("--texture-size", type=int, default=4096)
    p.add_argument("--hole-size", type=float, default=0.15)
    p.add_argument("--map-resolution", type=float, default=0.05)
    p.add_argument("--robot-height", type=float, default=1.8,
                   help="obstacles below this height go into the 2D nav map")
    p.add_argument("--no-center", action="store_true", help="keep the capture origin instead of centering")

    s = sub.add_parser("synth", help="write a synthetic demo session (room + walking person + dog)")
    s.add_argument("out", type=Path)
    s.add_argument("--frames", type=int, default=160)
    s.add_argument("--no-masks", action="store_true", help="omit masks to test ghost removal alone")
    s.add_argument("--no-depth", action="store_true",
                   help="camera-only capture like a phone without LiDAR (depth by stereo)")
    s.add_argument("--trajectory", choices=["spin", "walk"], default=None,
                   help="camera path (default: spin with depth, walk without)")

    i = sub.add_parser("inspect", help="print a summary of a session")
    i.add_argument("session", type=Path)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if a.cmd == "synth":
        from .synthetic import make_session
        traj = a.trajectory or ("walk" if a.no_depth else "spin")
        print(make_session(a.out, n_frames=a.frames, with_masks=not a.no_masks,
                           with_depth=not a.no_depth, trajectory=traj))
        return 0

    session_dir = _unpack(a.session)
    if a.cmd == "inspect":
        ses = Session.load(session_dir)
        masked = sum(1 for f in ses.frames if f.mask_path is not None)
        print(json.dumps({**ses.meta, "frames": len(ses.frames), "frames_with_masks": masked,
                          "duration_s": ses.frames[-1].timestamp - ses.frames[0].timestamp,
                          "color_size": ses.frames[0].color_size,
                          "depth_size": ses.frames[0].depth_size}, indent=2))
        return 0

    name = a.name or session_dir.name.replace(" ", "_")
    cfg = PipelineConfig(
        name=name,
        fusion=FusionConfig(voxel_size=a.voxel, max_depth=a.max_depth,
                            min_confidence=a.min_confidence, mask_dilate_px=a.mask_dilate),
        texture=TextureConfig(size=a.texture_size),
        texture_mode=a.texture,
        ghost_removal=not a.no_ghost_removal,
        resegment_model=a.resegment,
        hole_size=a.hole_size,
        map_resolution=a.map_resolution,
        map_obstacle_max_z=a.robot_height,
        center_xy=not a.no_center,
    )
    report = run(session_dir, a.out, cfg)
    print(json.dumps({k: report[k] for k in ("name", "frames", "triangles_visual",
                                             "ghost_triangles_removed", "extent_m", "outputs")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
