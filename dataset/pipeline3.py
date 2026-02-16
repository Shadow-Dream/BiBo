from infinigen.tools import export
import os
import bpy
from pathlib import Path
import sys

offset = int(sys.argv[1])
scenes = os.listdir("scenes")
scenes = list(sorted(scenes))

for scene in scenes[offset::8]:
    blendfile = os.path.join("scenes",scene)
    bpy.ops.wm.open_mainfile(filepath=str(blendfile))

    folder = export.export_scene(
        Path(blendfile),
        Path("assets"),
        format="obj",
        image_res=1024,
        vertex_colors=False,
        individual_export=True,
        omniverse_export=False,
    )