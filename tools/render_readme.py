"""Renders the README images from a synthetic kitchen run (needs MuJoCo + an off-screen GL:
MUJOCO_GL=egl on Linux).

    cleanscan synth-kitchen kitchen --depth-module a010
    cleanscan process-rl kitchen -o kitchen_scene
    MUJOCO_GL=egl python tools/render_readme.py kitchen kitchen_scene docs/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image


def input_sheet(session: Path, out: Path) -> None:
    """What the phone recorded: room pass frames and the guided captures."""
    sel = json.loads((session / "selection.json").read_text())["objects"]
    caps = {o["name"]: o["captures"] for o in sel}
    picks = [session / "color/000005.jpg", session / "color/000030.jpg", session / "color/000060.jpg",
             session / caps["bottle_a"]["scan"] / "color/000005.jpg",
             session / caps["fridge"]["open"] / "color/000015.jpg",
             session / caps["cupboard"]["open"] / "color/000015.jpg",
             session / caps["glass"]["profile"] / "color/000004.jpg",
             session / caps["glass"]["top"] / "color/000000.jpg"]
    labels = ["room pass", "room pass", "room pass (person walks through)", "pickable: walk around",
              "articulated: open", "articulated: open", "glass: side profile", "glass: from the top"]
    tiles = []
    for p, label in zip(picks, labels):
        im = cv2.resize(cv2.imread(str(p)), (320, 240))
        cv2.rectangle(im, (0, 0), (320, 24), (0, 0, 0), -1)
        cv2.putText(im, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


class SceneRenderer:
    def __init__(self, scene: Path, w: int = 960, h: int = 540):
        self.m = mujoco.MjModel.from_xml_path(str(scene / "export/scene.xml"))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.r = mujoco.Renderer(self.m, h, w)
        self.cam = mujoco.MjvCamera()
        sc = json.loads((scene / "scene.json").read_text())
        self.objects = {o["name"]: o for o in sc["objects"]}
        centers = np.array([o["pose"]["position"] for o in sc["objects"]])
        self.cam.lookat[:] = centers.mean(0) + [0, 0, 0.6]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 4.2, 90 + 25, -22
        floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.m.geom_rgba[floor] = [0.55, 0.55, 0.58, 1.0]  # the collision floor, shown for the picture
        self.m.geom_group[floor] = 1
        self.opt = mujoco.MjvOption()

    def frame(self) -> np.ndarray:
        self.r.update_scene(self.d, self.cam, self.opt)
        return self.r.render()

    def joint(self, obj: str) -> tuple[int, int]:
        o = self.objects[obj]
        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{o['id']}_{o['joints'][0]['name']}")
        return self.m.jnt_qposadr[jid], self.m.jnt_dofadr[jid]

    def body(self, obj: str) -> int:
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, self.objects[obj]["id"])


def actions_gif(scene: Path, out: Path, still: Path) -> None:
    """Fridge door opens, the drawer slides out, a bottle is lifted: physics in MuJoCo."""
    s = SceneRenderer(scene)
    Image.fromarray(s.frame()).save(still)
    door, drawer, bottle = s.joint("fridge"), s.joint("cupboard"), s.body("bottle_a")
    up_door = s.objects["fridge"]["joints"][0]["upper"]
    up_drawer = s.objects["cupboard"]["joints"][0]["upper"]
    mass = s.m.body_subtreemass[bottle]
    z0 = float(s.d.xpos[bottle][2])
    frames = []
    steps_per_frame = 20  # 25 fps at 2 ms
    for k in range(110):
        t = k / 110
        targets = [(door, up_door * min(1, 2 * t)), (drawer, up_drawer * min(1, max(0, 2 * t - 0.4)))]
        for _ in range(steps_per_frame):
            for (qa, da), q in targets:  # PD drive, like a robot pulling the handle
                s.d.qfrc_applied[da] = 400 * (q - s.d.qpos[qa]) - 40 * s.d.qvel[da]
            if t > 0.55:  # lift the bottle 15 cm and hold it there, like a gripper would
                z, vz = s.d.xpos[bottle][2], s.d.cvel[bottle][5]
                s.d.xfrc_applied[bottle, 2] = mass * (9.81 + 60 * (z0 + 0.15 - z) - 12 * vz)
            mujoco.mj_step(s.m, s.d)
        if k % 3 == 0:
            frames.append(Image.fromarray(s.frame()).resize((480, 270)).quantize(colors=96))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True)


if __name__ == "__main__":
    session, scene, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    input_sheet(session, out / "rl_kitchen_input.jpg")
    actions_gif(scene, out / "rl_kitchen_actions.gif", out / "rl_kitchen_scene.png")
    print("written to", out)
