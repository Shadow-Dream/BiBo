import bpy
import mathutils
import math
import numpy as np
import OpenEXR
import Imath
import sys
import cv2 as cv
import os
from PIL import Image, ImageDraw, ImageFont
import multiprocessing as mp
from scipy.spatial import ConvexHull
import shapely as sl
import logging
import time
log = logging.getLogger('werkzeug')
log.disabled = True

blender_path, port, instance_index = sys.argv[1:]
blender_path = str(blender_path)
port = int(port)
instance_index = int(instance_index)

blender_path = f"./scenes/{blender_path}.blend"
scene_name = blender_path.split("/")[-2]

'''
Setting Up Scene
'''

bpy.ops.wm.open_mainfile(filepath=blender_path)
point_lights = [obj for obj in bpy.data.objects if obj.type == 'LIGHT' and obj.data.type == 'POINT']
for light in point_lights:
    bpy.data.objects.remove(light, do_unlink=True)
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

cam_name = "Render_Camera"
if cam_name not in bpy.data.objects:
    cam = bpy.data.cameras.new(cam_name)
    cam_obj = bpy.data.objects.new(cam_name, cam)
    bpy.context.scene.collection.objects.link(cam_obj)
else:
    cam_obj = bpy.data.objects[cam_name]

for obj in bpy.data.objects:
    if obj.type == 'MESH':
        quat = obj.rotation_euler.to_quaternion()
        obj["origin_rotation"] = [quat.w, quat.x, quat.y, quat.z]
        obj["origin_position"] = np.array(obj.matrix_world.translation).tolist()

'''
Setting Up Render Pipeline
'''

fov = 75.0
pitch = 30
margin = 1.2
circle_radius = 15
circle_color = (255, 0, 0)
text_color = (255, 255, 255)
font_size = 20
front_direction = np.array([1,0,0],dtype=np.float32)
right_direction = np.array([0,1,0],dtype=np.float32)
pitch = np.radians(pitch)

bpy.context.scene.camera = cam_obj
cam_obj.data.angle = math.radians(fov)

bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.use_simplify = True

bpy.context.scene.eevee.taa_render_samples = 16
bpy.context.scene.eevee.taa_samples = 2
bpy.context.scene.eevee.use_soft_shadows = False
bpy.context.scene.eevee.use_bloom = False
bpy.context.scene.eevee.use_volumetric_lights = False
bpy.context.scene.eevee.motion_blur_steps = 0
bpy.context.scene.eevee.gi_diffuse_bounces = 0
bpy.context.scene.eevee.use_taa_reprojection = False

bpy.context.scene.use_nodes = True
bpy.context.view_layer.use_pass_cryptomatte_object = True
bpy.context.view_layer.use_pass_z = True
tree = bpy.context.scene.node_tree

for node in tree.nodes:
    tree.nodes.remove(node)

render_layer = tree.nodes.new(type="CompositorNodeRLayers")

file_output_color = tree.nodes.new(type="CompositorNodeOutputFile")
file_output_color.base_path = "./"
file_output_color.file_slots[0].path = f"render_color_{instance_index}"
file_output_color.format.file_format = "PNG"

file_output_depth = tree.nodes.new(type="CompositorNodeOutputFile")
file_output_depth.base_path = "./"
file_output_depth.file_slots[0].path = f"render_depth_{instance_index}"
file_output_depth.format.file_format = "OPEN_EXR"  # 32-bit

file_output_crypto = tree.nodes.new(type="CompositorNodeOutputFile")
file_output_crypto.base_path = "./"
file_output_crypto.file_slots[0].path = f"render_crypto_{instance_index}"
file_output_crypto.format.file_format = 'PNG'

cryptomatte_node = tree.nodes.new(type="CompositorNodeCryptomatte")
tree.links.new(render_layer.outputs['Image'], cryptomatte_node.inputs['Image'])
tree.links.new(render_layer.outputs['CryptoObject00'], cryptomatte_node.inputs['Crypto 00'])
tree.links.new(render_layer.outputs['CryptoObject01'], cryptomatte_node.inputs['Crypto 01'])
tree.links.new(render_layer.outputs['CryptoObject02'], cryptomatte_node.inputs['Crypto 02'])

tree.links.new(render_layer.outputs["Image"], file_output_color.inputs[0])
tree.links.new(cryptomatte_node.outputs["Image"], file_output_crypto.inputs[0])
tree.links.new(render_layer.outputs["Depth"], file_output_depth.inputs[0])

size = 512
x,y = np.meshgrid(np.arange(size), np.arange(size))
x = (x - size//2) * 2 * np.tan(np.radians(fov/2))
y = (size//2 - y) * 2 * np.tan(np.radians(fov/2))
grid = np.stack((x,y,np.ones_like(x)*size), axis=-1)
grid = grid / size

'''
Start First Render for Initialize
'''
position = np.array([0,0,0]).astype(np.float32)
roll = np.radians(0)
direction = (np.cos(roll) * front_direction + np.sin(roll) * right_direction) * np.cos(pitch)
direction = np.array([direction[0],direction[1],np.sin(pitch)],dtype=np.float32)
offset = direction
camera_position = position + offset
cam_obj.location = camera_position

direction = cam_obj.location - mathutils.Vector(position)
cam_obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()

bpy.ops.render.render(write_still=True)


while True:
    while not os.path.exists(f"capture/{instance_index}.lock"):
        time.sleep(0.1)

    data = np.load("capture/data.npy",allow_pickle=True)[None][0]

    objects = data["objects"]
    parents = data["parents"]
    index_dict = data["index_dict"]
    start_point = data["start_point"]
    env_id = data["env_id"]
    target = data["target"]
    anchor = data["anchor"]
    
    save_dir = os.path.join("capture",str(env_id))
    target_name = target
    anchor_name = anchor

    start_point = np.array(start_point)
    for instance in objects.values():
        instance["position"] = np.array(instance["position"],dtype=np.float32)
        instance["bbox"] = np.array(instance["bbox"],dtype=np.float32)

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            name = obj.name.replace("/","_").replace(".","_")
            if name in index_dict:
                if "mattress" in name.lower():
                    continue
                instance = objects[index_dict[name]]
                position = instance["position"]
                rotation = instance["rotation"]
                origin_quat = mathutils.Quaternion(obj["origin_rotation"])
                quat = mathutils.Quaternion(tuple(rotation))
                quat = origin_quat @ quat
                location = mathutils.Vector(position)
                scale = obj.matrix_world.to_scale()
                new_matrix = mathutils.Matrix.Translation(location) @ quat.to_matrix().to_4x4()
                new_matrix[0][0] *= scale.x
                new_matrix[1][1] *= scale.y
                new_matrix[2][2] *= scale.z
                obj.matrix_world = new_matrix
            else:
                if "origin_position" in obj:
                    obj.matrix_world.translation = mathutils.Vector(obj["origin_position"]) - mathutils.Vector(start_point)

    instance = objects[target]
    children = set([child for child,parent in parents.items() if parent==target] + [target])
    
    position = instance["position"]
    bbox = instance["bbox"]
    position = position + bbox.mean(0)
    bbox = bbox - bbox.mean(0)
    radius = np.linalg.norm(bbox,axis = -1).max()
    radius *= 1.6
    half_fov = fov / 2
    distance = radius / np.tan(np.radians(half_fov))

    target_object_scene_name = ""
    anchor_object_scene_name = ""

    hidden_objects = []
    for obj in bpy.context.scene.objects:
        name = obj.name.replace("/","_").replace(".","_")
        if name not in index_dict or index_dict[name] not in children:
            obj.hide_viewport = True
            obj.hide_render = True
            hidden_objects.append(obj)
        if name in index_dict:
            if index_dict[name] == target:
                target_object_scene_name = obj.name
            if index_dict[name] == anchor:
                anchor_object_scene_name = obj.name

    os.makedirs(f"{save_dir}/{target}",exist_ok=True)

    roll_list = [0,45,90,135,180,225,270,315]

    '''
    Render Template Image
    '''
    roll = roll_list[instance_index]
    camera_index = instance_index

    roll = np.radians(roll)
    direction = (np.cos(roll) * front_direction + np.sin(roll) * right_direction) * np.cos(pitch)
    direction = np.array([direction[0],direction[1],np.sin(pitch)],dtype=np.float32)
    offset = direction * distance
    camera_position = position + offset
    cam_obj.location = camera_position

    direction = cam_obj.location - mathutils.Vector(position)
    cam_obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    
    bpy.ops.render.render(write_still=True)
    exr_file = OpenEXR.InputFile(f"render_depth_{camera_index}0001.exr")
    depth_str = exr_file.channel("V", Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape((512, 512)).copy()[...,None]
    mask = depth < 100
    camera_grid = depth * grid
    camera_grid = camera_grid.reshape(-1,3)

    forward = np.array(-direction)
    forward /= np.linalg.norm(forward)
    up = np.array([0,0,1])
    right = np.cross(forward,up)
    right = right / np.linalg.norm(right)
    up = np.cross(right,forward)
    rotation = np.array([right,up,forward])
    points = camera_grid @ rotation
    points += camera_position
    points_grid = points.reshape(size,size,3)

    os.rename(f"render_color_{camera_index}0001.png",f"{save_dir}/{target}/{camera_index}_tpl.png")
    np.save(f"{save_dir}/{target}/{camera_index}_depth.npy",points_grid)
    np.save(f"{save_dir}/{target}/{camera_index}_tpl_position.npy",position)
    mask = mask.astype(np.uint8)*255
    cv.imwrite(f"{save_dir}/{target}/{camera_index}_mask.png",mask)

    
    if os.path.exists(f"capture/{camera_index}.lock"):
        os.remove(f"capture/{camera_index}.lock")
    '''
    Done Render Template Image
    '''

    for obj in hidden_objects:
        obj.hide_viewport = False
        obj.hide_render = False

    parent = parents[target]
    parent = objects[parent]

    parent_position = parent["position"]
    parent_rotation = parent["rotation"]
    parent_bbox = parent["bbox"]
    parent_shape = parent["shape"]

    anchor = objects[anchor]
    anchor_position = anchor["position"]
    anchor_rotation = anchor["rotation"]
    anchor_bbox = anchor["bbox"]
    anchor_shape = anchor["shape"]

    def qrot_numpy(q, v):
        """
        Rotate vector(s) v about the rotation described by quaternion(s) q.

        q: ndarray of shape (..., 4), quaternion (w, x, y, z)
        v: ndarray of shape (..., 3), vector to be rotated
        Returns: ndarray of shape (..., 3), rotated vector
        """
        assert q.shape[-1] == 4
        assert v.shape[-1] == 3
        assert q.shape[:-1] == v.shape[:-1]

        original_shape = q.shape[:-1]  # batch shape
        q_flat = q.reshape(-1, 4)
        v_flat = v.reshape(-1, 3)

        qvec = q_flat[:, 1:]  # (x, y, z)
        uv = np.cross(qvec, v_flat)
        uuv = np.cross(qvec, uv)
        rotated = v_flat + 2 * (q_flat[:, :1] * uv + uuv)

        return rotated.reshape(*original_shape, 3)

    def get_convex_hull(points):
        try:
            hull = ConvexHull(points)
            contour = sl.Polygon(points[hull.vertices])
            return contour
        except:
            return None

    def get_ray_intersections(polygon, point, ray_length=1e5):
        """
        Given a Shapely Polygon and a Point inside it,
        return a numpy array of shape (8, 2) containing intersection points between 
        the polygon and rays cast from the point in 8 directions:
        [+x, +xy, +y, -xy, -x, -x-y, -y, x-y]
        
        If no intersection found in a direction, (0, 0) is used.
        """
        assert polygon.contains(point), "Point must be inside the polygon"
        
        cx, cy = point.x, point.y
        directions = [
            (1, 0),     # +x
            (1, 1),     # +xy
            (0, 1),     # +y
            (-1, 1),    # -xy
            (-1, 0),    # -x
            (-1, -1),   # -x-y
            (0, -1),    # -y
            (1, -1),    # x-y
        ]
        
        results = []

        for dx, dy in directions:
            norm = math.hypot(dx, dy)
            dx /= norm
            dy /= norm

            ray = sl.LineString([
                (cx, cy),
                (cx + dx * ray_length, cy + dy * ray_length)
            ])

            intersection = ray.intersection(polygon.boundary)

            if intersection.is_empty:
                results.append([point.x, point.y])
            elif intersection.geom_type == 'Point':
                results.append([intersection.x, intersection.y])
            elif intersection.geom_type == 'MultiPoint':
                closest = min(intersection.geoms, key=lambda p: p.distance(point))
                results.append([closest.x, closest.y])
            else:
                results.append([point.x, point.y])

        return np.array(results)  # shape (8, 2)

    anc_position = np.array(anchor_position)
    anc_rotation = np.array(anchor_rotation)
    anc_shape = np.array(anchor_shape)
    quat = anc_rotation[None][None]
    quat = np.broadcast_to(quat,(anc_shape.shape[0],anc_shape.shape[1],4))
    anc_shape = qrot_numpy(quat, anc_shape)
    anc_shape += anc_position[None][None]
    anc_shape = anc_shape[:,:,:2]
    object_polygon = None
    for box in anc_shape:
        polygon = get_convex_hull(box)
        if polygon is None:
            continue
        if object_polygon is None:
            object_polygon = polygon
        else:
            object_polygon = object_polygon.union(polygon)
    object_polygon = object_polygon.buffer(0.4, join_style="mitre")
            
    position_point = sl.Point(position[:2])
    if object_polygon.contains(position_point):
        label_positions = get_ray_intersections(object_polygon,position_point)
        anchor_bbox = anchor_bbox + anchor_position
        anchor_position = anchor_bbox.mean(0)
        anchor_bbox -= position
    else:
        position_point = sl.Point(object_polygon.centroid)
        label_positions = get_ray_intersections(object_polygon,position_point)
        anchor_bbox = anchor_bbox + anchor_position
        anchor_position = anchor_bbox.mean(0)
        anchor_bbox -= anchor_position

    radius = anchor_bbox
    radius = radius - radius.mean(0)
    radius += 0.3
    radius = radius.max(0)
    radius = np.linalg.norm(radius)
    radius *= 1.2
    half_fov = fov / 2
    distance = radius / np.tan(np.radians(half_fov))
    distance = max(distance,1)

    hidden_objects = []
    for obj in bpy.context.scene.objects:
        if "rug" in obj.name.lower():
            obj.hide_viewport = True
            obj.hide_render = True
            hidden_objects.append(obj)

    label_position_dict = {}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for i,label_position in enumerate(label_positions):
        hit, hit_location, hit_normal, hit_index, hit_object, matrix = bpy.context.scene.ray_cast(
            depsgraph,
            mathutils.Vector([label_position[0],label_position[1],1.5]),
            mathutils.Vector([0,0,-1])
        )
        if hit and "floor" in hit_object.name:
            label_position_dict[i + 1] = hit_location

    checked_label_indices = set()
    images = []

    def draw_index_on_image(image_path, coordinates, output_path, bbox_info):
        # Load the image
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        # Load a default font
        try:
            font = ImageFont.truetype("/root/infinigen/Arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        for index,(x,y) in coordinates.items():
            x=int(x)
            y=int(y)
            text_size = draw.textbbox((0, 0), str(index), font=font)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]
            draw.ellipse((x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius), fill=circle_color, width=3)
            text_x = x - text_width // 2
            # text_y = y - text_height // 2
            text_y = y - text_height // 2 - 4
            # Draw text at the center of the circle
            draw.text((text_x, text_y), str(index), fill=text_color, font=font)
        if bbox_info is not None:
            draw.rectangle(bbox_info, outline="red", width=3)
        # Save the modified image
        image.save(output_path)
        return output_path

    depsgraph = bpy.context.evaluated_depsgraph_get()

    '''
    Begin Render Object
    '''
    roll = roll_list[instance_index]
    camera_index = instance_index
    roll = np.radians(roll)
    direction = (np.cos(roll) * front_direction + np.sin(roll) * right_direction) * np.cos(pitch)
    direction = np.array([direction[0],direction[1],np.sin(pitch)],dtype=np.float32)
    offset = direction * distance
    camera_position = anchor_position + offset
    cam_obj.location = camera_position

    direction = cam_obj.location - mathutils.Vector(anchor_position)
    cam_obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()

    forward = np.array(-direction)
    forward /= np.linalg.norm(forward)
    up = np.array([0,0,1])
    right = np.cross(forward,up)
    right = right / np.linalg.norm(right)
    up = np.cross(right,forward)
    camera_matrix = np.array([right,up,forward])

    def world_to_camera_view(world_coords):
        camera_coords = np.array(world_coords) - camera_position
        camera_coords = camera_coords @ camera_matrix.T
        x,y,z = camera_coords[:,0],camera_coords[:,1],camera_coords[:,2]

        ratio = 1/(2*np.tan(np.radians(fov/2)))
        x_ndc = (x / z) * ratio
        y_ndc = (y / z) * ratio

        pixel_x = (x_ndc + 0.5) * 512
        pixel_y = (0.5 - y_ndc) * 512
        return np.stack([pixel_x, pixel_y],axis = -1)

    cryptomatte_node.matte_id = target_object_scene_name + "," + anchor_object_scene_name
    bpy.ops.render.render(write_still=True)

    def get_depth_map():
        exr_file = OpenEXR.InputFile(f"render_depth_{camera_index}0001.exr")
        depth_str = exr_file.channel("V", Imath.PixelType(Imath.PixelType.FLOAT))
        depth = np.frombuffer(depth_str, dtype=np.float32).reshape((512, 512)).copy()[...,None]
        
        camera_grid = depth * grid
        camera_grid = camera_grid.reshape(-1,3)

        forward = np.array(-direction)
        forward /= np.linalg.norm(forward)
        up = np.array([0,0,1])
        right = np.cross(forward,up)
        right = right / np.linalg.norm(right)
        up = np.cross(right,forward)
        rotation = np.array([right,up,forward])
        points = camera_grid @ rotation
        points += camera_position
        points_grid = points.reshape(size,size,3)

        np.save(f"{save_dir}/{target}/{camera_index}_rgb_depth.npy",points_grid)

    get_depth_map()

    mask = cv.imread(f"render_crypto_{camera_index}0001.png",cv.IMREAD_UNCHANGED)[...,-1]
    mask = mask > 0
    if mask.sum() < 16:
        render_metadata = {"images":[],"labels":{}}
        np.save(f"{save_dir}/{target}/{camera_index}_meta.npy",render_metadata)
        continue
    
    #save depth mask
    mask = mask.astype(np.uint8)*255
    cv.imwrite(f"{save_dir}/{target}/{camera_index}_rgb_mask.png",mask)

    has_label = {}

    current_checked_label_indices = []
    for label_index,label_position in label_position_dict.items():
        hit, hit_location, hit_normal, hit_index, hit_object, matrix = bpy.context.scene.ray_cast(
            depsgraph,
            mathutils.Vector(camera_position),
            (mathutils.Vector(label_position) - mathutils.Vector(camera_position)).normalized()
        )

        if hit and np.linalg.norm(np.array(hit_location - label_position))<0.1:
            
            label_position_2d = world_to_camera_view(np.array(label_position)[None])[0]
            x,y = label_position_2d
            x=int(x)
            y=int(y)
            occupation_sum = mask[y - circle_radius:y + circle_radius,x-circle_radius:x+circle_radius].sum()
            if occupation_sum/(circle_radius**2)<0.2:
                has_label[label_index] = label_position_2d
                current_checked_label_indices.append(label_index)
    
    if len(has_label)==0:
        render_metadata = {"images":[],"labels":{}}
        np.save(f"{save_dir}/{target}/{camera_index}_meta.npy",render_metadata)
        continue

    bbox_info = None
    if target_name != anchor_name:
        target_bbox = instance["bbox"] + instance["position"]
        camera_bbox = world_to_camera_view(target_bbox)
        (x1,y1),(x2,y2) = camera_bbox.min(0),camera_bbox.max(0)
        bbox_info = (x1-10,y1-10,x2+10,y2+10)
    
    images.append(
        draw_index_on_image(
            f"render_color_{camera_index}0001.png",
            has_label,
            f"{save_dir}/{target}/{camera_index}_rgb.png",bbox_info
        )
    )
    checked_label_indices.update(current_checked_label_indices)
    '''
    End Render Images
    '''
    labels = {k:np.array(v).tolist()[:2] for k,v in label_position_dict.items() if k in checked_label_indices}
    render_metadata = {
        "images":list(images),
        "labels":labels
    }
    np.save(f"{save_dir}/{target}/{camera_index}_meta.npy",render_metadata)

    os.remove(f"render_color_{camera_index}0001.png")
    os.remove(f"render_crypto_{camera_index}0001.png")
    os.remove(f"render_depth_{camera_index}0001.exr")