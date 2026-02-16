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

def process_obj(file):
    input_file = output_file = file
    with open(input_file, "r") as infile:
        lines = infile.readlines()

    with open(output_file, "w") as outfile:
        for line in lines:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    new_line = f"v {x} {-z} {y}\n"
                    outfile.write(new_line)
                else:
                    outfile.write(line)
            else:
                outfile.write(line)

def filter_mesh(mesh):
    z_coords = mesh.vertices[:, 2]
    valid_indices = (z_coords >= -1) & (z_coords <= 4)
    filtered_vertices = mesh.vertices[valid_indices]
    valid_faces = []
    for face in mesh.faces:
        if all(valid_indices[face]):
            valid_faces.append(face)
    new_mesh = trimesh.Trimesh(vertices=filtered_vertices, faces=valid_faces)
    return new_mesh

def type_from_name(name):
    for predefined_type in predefined_types:
        if predefined_type in name.lower():
            return predefined_type
    
    name = name.split("Factory")[0]
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()
    return name

def filter_obj(input_path, output_path, z_min=-0.2-0.4317423105239868, z_max=3-0.4317423105239868):
    obj = trimesh.load(input_path, force='scene')
    
    # If the OBJ is a Scene, process each geometry individually
    if isinstance(obj, trimesh.Scene):
        new_scene = trimesh.Scene()
        for name, geom in obj.geometry.items():
            filtered_geom = filter_geometry(geom, z_min, z_max)
            if filtered_geom is not None:
                new_scene.add_geometry(filtered_geom, node_name=name)
        new_scene.export(output_path)
    else:
        # If it's a single mesh, process it directly
        filtered_mesh = filter_geometry(obj, z_min, z_max)
        if filtered_mesh is not None:
            filtered_mesh.export(output_path)

def filter_geometry(mesh, z_min, z_max):
    # Get the z-coordinates of all vertices
    vertices = mesh.vertices
    z_coords = vertices[:, 2]
    
    # Find the indices of vertices to keep
    keep_indices = (z_coords >= z_min) & (z_coords <= z_max)
    
    # If no vertices are left, return None
    if not keep_indices.any():
        return None
    
    # Map old vertex indices to new indices
    new_vertex_indices = -1 * np.ones(len(vertices), dtype=int)
    new_vertex_indices[keep_indices] = np.arange(keep_indices.sum())
    
    # Filter vertices
    new_vertices = vertices[keep_indices]
    
    # Filter faces: keep only faces where all vertices are in the new vertex set
    faces = mesh.faces
    mask_faces = keep_indices[faces].all(axis=1)
    new_faces = new_vertex_indices[faces[mask_faces]]
    
    # Create a new mesh with the filtered vertices and faces
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    
    # Copy material information if available
    if hasattr(mesh, 'visual') and mesh.visual.kind != 'none':
        new_mesh.visual = mesh.visual.copy()
    
    return new_mesh

def is_type(name,type_set):
    return any([t in name for t in type_set])

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
    if os.path.exists(scene_dir):
        shutil.rmtree(scene_dir)
    os.makedirs(scene_dir)
    config = {}
    discards = {"hoof"}

    predefined_types = {
        "ceiling",
        "exterior",
        "floor",
        "wall",
        "support",
        "rug",
        "window",
        "pillar",
        "wall art",
        "door"
    }

    takable_types = {
        "large plant container"
    }

    room_type = "living-room" if "livingroom" in export_path else "bedroom"

    for name,translation in tqdm(metadata.items()):
        if "_0_0_" in name and room_type not in name:
            continue
        ignore = False
        for discard in discards:
            if discard in name:
                ignore = True
                break
        if ignore:
            continue
        
        src_path = os.path.join(export_path, f"{name}")
        dst_path = os.path.join(scene_dir, f"{name}")
        shutil.copytree(src_path, dst_path)
        mesh_path = os.path.join(dst_path, f"{name}.obj")
        process_obj(mesh_path)

        urdf_path = os.path.join(scene_dir, f"{name}.urdf")
        
        config[name] = {}
        config[name]["position"] = translation
        config[name]["type"] = type_from_name(name)
        
        if not is_type(config[name]["type"], predefined_types):
            try:
                filter_obj(mesh_path, mesh_path, z_min=-0.2-translation[2], z_max=3-translation[2])
            except:
                continue
            
            if not is_type(config[name]["type"],takable_types):
                sp.run(["python", "vhacd.py", mesh_path])

            mesh = trimesh.load_mesh(mesh_path)

            bbox_oriented = mesh.bounding_box_oriented.vertices.tolist()
            bbox = mesh.bounding_box.vertices
            min_point,max_point = bbox.min(0),bbox.max(0)
            x1,y1,z1 = min_point
            x2,y2,z2 = max_point
            bbox = [[x1,y1,z1],[x1,y1,z2],[x1,y2,z1],[x1,y2,z2],[x2,y2,z1],[x2,y2,z2],[x2,y1,z1],[x2,y1,z2]]
            if is_type(config[name]["type"],takable_types):
                bbox_np = np.array(bbox)
                bbox_center = bbox_np.mean(0)
                bbox_center[2] = bbox_np[:,2].min()
                bbox_centralized = bbox_np - bbox_center
                bbox_centralized[:, :2] = np.clip(bbox_centralized[:, :2], -0.22, 0.22)
                bbox_centralized[:, 2] = np.clip(bbox_centralized[:, 2], 0, 0.5)
                bbox_np = bbox_centralized
                bbox_np[:,2] += bbox_center[2]
                bbox = bbox_np.tolist()
                ix,iy,iz = bbox_np.min(0)
                ax,ay,az = bbox_np.max(0)
                bbox_np = np.array([ix,ax,iy,ay,iz,az])[None]

            config[name]["bbox_oriented"] = bbox_oriented
            config[name]["bbox"] = bbox
        
        if is_type(config[name]["type"],takable_types):
            bboxes = bbox_np
        else:
            bboxes = os.path.join(dst_path,"bboxes.npy")
            if os.path.exists(bboxes):
                bboxes = np.load(bboxes)
            else:
                bboxes = None

        if bboxes is not None:
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
        else:
            geom = GEOM_TEMPLATE.replace("{id}",name).replace("{geom_id}",name)

        urdf = (
            URDF_TEMPLATE
            .replace("{id}", name)
            .replace("{geom}",geom)
        )

        with open(urdf_path, "w") as urdf_file:
            urdf_file.write(urdf)

    if "livingroom" in export_path:
        config = {"objects":config,"room":"living-room"}
    else:
        config = {"objects":config,"room":"bedroom"}
    config_path = os.path.join(scene_dir, "config.json")
    with open(config_path, "w") as config_file:
        json.dump(config, config_file)

