import bpy
import mathutils
import math
import random
import argparse
import numpy as np
import OpenEXR
import Imath
import time
import cv2 as cv
import os
from PIL import Image, ImageDraw, ImageFont
import multiprocessing as mp
import logging
import sys
from tqdm import tqdm
log = logging.getLogger('werkzeug')
log.disabled = True
process_id = int(sys.argv[1])
output_dir = "outputs/blender"
scenes = os.listdir(output_dir)
scenes = [scene for scene in scenes if scene.startswith("bedroom")]
scenes = sorted(list(scenes))
scenes = scenes[process_id::16]
for scene in scenes:
    blender_path = os.path.join(output_dir,scene,"scene.blend")
    if not os.path.exists(blender_path):
        continue 
    decimated_path = os.path.join("scenes",scene + ".blend")
    if os.path.exists(decimated_path):
        continue
    bpy.ops.wm.open_mainfile(filepath=blender_path)
    print(f"[Start decimation] {scene}")
    decimated_count = 0
    # Iterate over all objects in the scene.
    for obj in bpy.context.scene.objects:   
        # Process mesh objects only.
        if obj.type == 'MESH':
            if "0/0" in obj.name:
                continue
            
            # Skip objects that were already decimated.
            if "decimated" in obj and obj["decimated"]:
                print(f"Skip {obj.name} (already decimated)")
                continue
            
            # Set current object as active.
            bpy.context.view_layer.objects.active = obj
            
            # Get current face count.
            face_count = len(obj.data.polygons)
            
            # Adjust decimation ratio by face count.
            if face_count > 100000:
                ratio = 10000 / face_count
            elif face_count > 10000:
                ratio = 1 - (face_count - 10000) / 100000
            else:
                continue

            print(f"Decimating {obj.name}")
            decimated_count += 1

            # Add Decimate modifier.
            decimate_modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
            decimate_modifier.ratio = ratio
            
            # Apply modifier.
            bpy.ops.object.modifier_apply(modifier=decimate_modifier.name)
            
            # Mark object as decimated.
            obj["decimated"] = True

            print(f"Decimation applied to {obj.name}, faces: {len(obj.data.polygons)}")

    if decimated_count:
        # Save scene so decimation state is preserved.
        saved_blend_path = decimated_path
        bpy.ops.wm.save_as_mainfile(filepath=saved_blend_path)
        print(f"Scene saved to {saved_blend_path}")

    print(f"[Finish decimation] {scene}")
import bpy
import mathutils
import math
import random
import argparse
import numpy as np
import OpenEXR
import Imath
import time
import cv2 as cv
import os
from PIL import Image, ImageDraw, ImageFont
import multiprocessing as mp
import logging
import sys
from tqdm import tqdm
log = logging.getLogger('werkzeug')
log.disabled = True
process_id = int(sys.argv[1])
output_dir = "outputs/blender"
scenes = os.listdir(output_dir)
scenes = [scene for scene in scenes if scene.startswith("bedroom")]
scenes = sorted(list(scenes))
scenes = scenes[process_id::16]
for scene in scenes:
    blender_path = os.path.join(output_dir,scene,"scene.blend")
    if not os.path.exists(blender_path):
        continue 
    decimated_path = os.path.join("scenes",scene + ".blend")
    if os.path.exists(decimated_path):
        continue
    bpy.ops.wm.open_mainfile(filepath=blender_path)
    print(f"[Start decimation] {scene}")
    decimated_count = 0
    # Iterate over all objects in the scene.
    for obj in bpy.context.scene.objects:   
        # Process mesh objects only.
        if obj.type == 'MESH':
            if "0/0" in obj.name:
                continue
            
            # Skip objects that were already decimated.
            if "decimated" in obj and obj["decimated"]:
                print(f"Skip {obj.name} (already decimated)")
                continue
            
            # Set current object as active.
            bpy.context.view_layer.objects.active = obj
            
            # Get current face count.
            face_count = len(obj.data.polygons)
            
            # Adjust decimation ratio by face count.
            if face_count > 100000:
                ratio = 10000 / face_count
            elif face_count > 10000:
                ratio = 1 - (face_count - 10000) / 100000
            else:
                continue

            print(f"Decimating {obj.name}")
            decimated_count += 1

            # Add Decimate modifier.
            decimate_modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
            decimate_modifier.ratio = ratio
            
            # Apply modifier.
            bpy.ops.object.modifier_apply(modifier=decimate_modifier.name)
            
            # Mark object as decimated.
            obj["decimated"] = True

            print(f"Decimation applied to {obj.name}, faces: {len(obj.data.polygons)}")

    if decimated_count:
        # Save scene so decimation state is preserved.
        saved_blend_path = decimated_path
        bpy.ops.wm.save_as_mainfile(filepath=saved_blend_path)
        print(f"Scene saved to {saved_blend_path}")

    print(f"[Finish decimation] {scene}")
