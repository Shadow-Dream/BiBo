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
    print(f"【开始减面】{scene}")
    decimated_count = 0
    # 遍历场景中的所有物体
    for obj in bpy.context.scene.objects:   
        # 只处理网格类型的物体
        if obj.type == 'MESH':
            if "0/0" in obj.name:
                continue
            
            # 检查是否已经减面
            if "decimated" in obj and obj["decimated"]:
                print(f"跳过 {obj.name}（已减面）")
                continue
            
            # 激活物体
            bpy.context.view_layer.objects.active = obj
            
            # 获取当前物体的面数
            face_count = len(obj.data.polygons)
            
            # 根据面数设置不同的减面比例
            if face_count > 100000:
                ratio = 10000 / face_count
            elif face_count > 10000:
                ratio = 1 - (face_count - 10000) / 100000
            else:
                continue

            print(f"正在对 {obj.name} 进行减面")
            decimated_count += 1

            # 添加 Decimate Modifier
            decimate_modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
            decimate_modifier.ratio = ratio
            
            # 应用修改器
            bpy.ops.object.modifier_apply(modifier=decimate_modifier.name)
            
            # 添加标记，表示已减面
            obj["decimated"] = True

            print(f"减面操作已应用于 {obj.name}, 面数{len(obj.data.polygons)}")

    if decimated_count:
        # 2. 保存场景，以便下次加载时保留减面状态
        saved_blend_path = decimated_path
        bpy.ops.wm.save_as_mainfile(filepath=saved_blend_path)
        print(f"场景已保存至 {saved_blend_path}")

    print(f"【完成减面】{scene}")
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
    print(f"【开始减面】{scene}")
    decimated_count = 0
    # 遍历场景中的所有物体
    for obj in bpy.context.scene.objects:   
        # 只处理网格类型的物体
        if obj.type == 'MESH':
            if "0/0" in obj.name:
                continue
            
            # 检查是否已经减面
            if "decimated" in obj and obj["decimated"]:
                print(f"跳过 {obj.name}（已减面）")
                continue
            
            # 激活物体
            bpy.context.view_layer.objects.active = obj
            
            # 获取当前物体的面数
            face_count = len(obj.data.polygons)
            
            # 根据面数设置不同的减面比例
            if face_count > 100000:
                ratio = 10000 / face_count
            elif face_count > 10000:
                ratio = 1 - (face_count - 10000) / 100000
            else:
                continue

            print(f"正在对 {obj.name} 进行减面")
            decimated_count += 1

            # 添加 Decimate Modifier
            decimate_modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
            decimate_modifier.ratio = ratio
            
            # 应用修改器
            bpy.ops.object.modifier_apply(modifier=decimate_modifier.name)
            
            # 添加标记，表示已减面
            obj["decimated"] = True

            print(f"减面操作已应用于 {obj.name}, 面数{len(obj.data.polygons)}")

    if decimated_count:
        # 2. 保存场景，以便下次加载时保留减面状态
        saved_blend_path = decimated_path
        bpy.ops.wm.save_as_mainfile(filepath=saved_blend_path)
        print(f"场景已保存至 {saved_blend_path}")

    print(f"【完成减面】{scene}")
