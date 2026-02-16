import os
import json
import trimesh
import shutil
import re
from tqdm import tqdm
import numpy as np
import subprocess as sp
import sys

BBOX_TEMPLATE = """
            <collision>
                <origin xyz="{pos}"/>
                <geometry>
				    <box size="{size}"/>
                </geometry>
			</collision>
"""

GEOM_TEMPLATE = """
            <collision>
				<origin xyz="0 0 0"/>
                <geometry>
				    <mesh filename="./{id}/{geom_id}.obj"/>
                </geometry>
			</collision>
                    
"""

URDF_TEMPLATE = """<?xml version="0.0"?>
	<robot name="object">
		<link name="object">
			<visual>
				<origin xyz="0 0 0"/>
				<geometry>
					<mesh filename="./{id}/{id}.obj"/>
				</geometry>
			</visual>
			{geom}
		</link>

	</robot>
"""

process_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
assets_path = "../infinigen/assets"
scenes = os.listdir(assets_path)
scenes = list(sorted(list(scenes)))
scenes = scenes[process_id::8]
base_dir = "scenes"
for scene in scenes:
    export_path = os.path.join(assets_path, scene)
    metadata_path = os.path.join(export_path, "metadata.json")

    with open(metadata_path, "r") as json_file:
        metadata = json.load(json_file)

    scene_dir = os.path.join(base_dir,scene.replace("export","").replace("blend","").replace("_","").replace(".",""))
    room_type = "living-room" if "livingroom" in export_path else "bedroom"

    for name,translation in tqdm(metadata.items()):
        if room_type + "_0_0_wall" not in name:
            continue
        
        dst_path = os.path.join(scene_dir, f"{name}")
        mesh_path = os.path.join(dst_path, f"{name}.obj")
        urdf_path = os.path.join(scene_dir, f"{name}.urdf")
        sp.run(["python", "vhacd.py", mesh_path])
        bboxes = os.path.join(dst_path,"bboxes.npy")
        bboxes = np.load(bboxes)
        geoms = []
        for bbox in bboxes:
            bbox = [[bbox[0],bbox[2],bbox[4]],[bbox[1],bbox[3],bbox[5]]]
            bbox = np.array(bbox)
            size = np.abs(bbox[0] - bbox[1])
            pos = (bbox[0] + bbox[1]) / 2
            size = [str(i) for i in size]
            pos = [str(i) for i in pos]
            size = " ".join(size)
            pos = " ".join(pos)
            geoms.append(
                BBOX_TEMPLATE
                .replace("{size}",size)
                .replace("{pos}",pos)
            )
        geom = "".join(geoms)

        urdf = (
            URDF_TEMPLATE
            .replace("{id}", name)
            .replace("{geom}",geom)
        )

        with open(urdf_path, "w") as urdf_file:
            urdf_file.write(urdf)

