import os
import json
import random
import trimesh
import re
import numpy as np
import networkx as nx
import shapely as sl
from scipy.spatial import ConvexHull
import shutil
from tqdm import tqdm
import copy
import base64
from openai import OpenAI
import cv2 as cv
minn = 100
maxx = 0
tol = 0
simple_task_num = 20
client = OpenAI(
    api_key=""
)

def get_convex_hull(points):
    try:
        hull = ConvexHull(points)
        contour = sl.Polygon(points[hull.vertices])
        contour = contour.buffer(0.3, join_style="mitre")
        return contour
    except:
        return None

def _build_start_point(scene_dir):
    scene_config = os.path.join(scene_dir, "config.json")
    with open(scene_config, "r") as json_file:
        metadata = json.load(json_file)
    objects = metadata["objects"]
    room = metadata["room"]
    floor_name = None
    for name in objects:
        if room in name and "floor" in name:
            floor_name = name
            break
    floor_config = objects[floor_name]
    floor_urdf = os.path.join(scene_dir, floor_name + ".urdf")
    with open(floor_urdf,"r") as f:
        urdf = f.read()
    
    mesh_file = re.search(r'filename="([^"]+)"', urdf).groups()[0]
    mesh_file = os.path.join(os.path.dirname(floor_urdf),mesh_file)
    mesh = trimesh.load(mesh_file)

    floor_position = floor_config["position"]
    floor_position = np.array(floor_position)

    vertices = mesh.vertices
    vertices += floor_position
    vertices = vertices[:,:2]

    faces = mesh.faces
    floor = None
    for face in faces:
        subpoly = vertices[face]
        if floor is None:
            floor = sl.Polygon(subpoly)
        else:
            floor = floor.union(sl.Polygon(subpoly))
    floor = floor.buffer(-0.3, join_style="mitre")
    return floor

def _build_objects(scene_dir, parents):
    scene_config = os.path.join(scene_dir, "config.json")
    with open(scene_config, "r") as json_file:
        metadata = json.load(json_file)
    objects = metadata["objects"]
    ignore_types = {
        "ceiling",
        "ceiling light",
        "exterior"
    }
    layout_types = {
        
    }
    decoration_types = {
        "rug",
        "window",
        "floor",
        "pillar",
        "ceiling",
        "exterior",
        "wall",
        "support",
        "wall art",
        "mirror",
        "lite door",
        "panel door",
        "glass door",
        "door"
    }

    name_dict = {}
    index_dict = {}
    object_polygon = None
    object_positions = {}
    for object_id, setting in objects.items():
        name = setting["type"]
        if name not in ignore_types and name not in layout_types and name not in decoration_types:
            name_dict[name] = name_dict.get(name,0) + 1
            name_index = name_dict[name]
            object_ind = f"{name}{name_index}"
            index_dict[object_id] = object_ind
            bbox = np.array(setting["bbox"])
            bbox = get_convex_hull(bbox)
            if object_polygon is None:
                object_polygon = bbox
            else:
                object_polygon = object_polygon.union(bbox)
            object_positions[object_ind] = np.array(setting["position"]) + np.array(setting["bbox"]).mean(0)

    parents = {index_dict[k]:index_dict[v] for k,v in parents.items() if k in index_dict and v in index_dict}
    reversed_index_dict = {v:k for k,v in index_dict.items()}
    return index_dict, reversed_index_dict, object_polygon, parents, object_positions
    
def _build_parents(scene_dir):
    scene_config = os.path.join(scene_dir, "config.json")
    with open(scene_config, "r") as json_file:
        metadata = json.load(json_file)
    objects = metadata["objects"]

    ignore_types = {
        "ceiling",
        "ceiling light",
        "exterior"
    }
    layout_types = {
    }
    decoration_types = {
        "rug",
        "window",
        "floor",
        "pillar",
        "ceiling",
        "exterior",
        "wall",
        "support",
        "wall art",
        "mirror",
        "lite door",
        "panel door",
        "glass door",
        "door"
    }

    objects = {name:setting for name, setting in objects.items() 
                if setting["type"] not in ignore_types
                and setting["type"] not in layout_types
                and setting["type"] not in decoration_types}
    
    object_list = list(objects.keys())
    bboxes = [np.array(objects[name]["bbox"])[None,:] for name in object_list]
    bboxes = np.concatenate(bboxes,0)
    positions = [np.array(objects[name]["position"])[None,None,:] for name in object_list]
    positions = np.concatenate(positions,0)
    bboxes += positions
    height = bboxes[:,:,2].min(axis=-1)
    bboxes = bboxes[:,:,:2]
    min_axis,max_axis = bboxes.min(1),bboxes.max(1)
    pairs = np.stack(np.meshgrid(np.arange(len(object_list)),np.arange(len(object_list))),-1)
    pair_min_axis = min_axis[pairs]
    pair_max_axis = max_axis[pairs]
    pair_min_axis = pair_min_axis.max(-2)
    pair_max_axis = pair_max_axis.min(-2)
    pair_length = np.maximum(pair_max_axis - pair_min_axis,0)
    pair_intersection = pair_length.prod(-1)
    area = (max_axis - min_axis).prod(-1)
    pair_min_area = area[pairs].min(-1)
    pairs = (pair_intersection / pair_min_area) > 0.75
    start,end = np.where(pairs)
    graph = nx.Graph()
    graph.add_edges_from(list(zip(start,end)))
    object_sets = list(nx.connected_components(graph))
    parents = {}
    
    for object_set in object_sets:
        object_set = list(object_set)
        parent = object_set[height[object_set].argmin()]
        for object_index in object_set:
            parents[object_list[object_index]] = object_list[parent]
    return parents

def format_coordinate_string(coordinate):
    return f"({coordinate[0]:.1f}, {coordinate[1]:.1f})"

def format_objects(objects,object_dict,parent_labels):
    rev = {v:k for k,v in parent_labels.items()}
    objects = []
    for name,contain in object_dict.items():
        if name == "origin":
            continue
        i = rev[name]
        object_string = f"- {i}: {name}"

        if contain:
            contain_string = ", ".join(contain)
            object_string += f", containing [{contain_string}]"
        objects.append(object_string)
    objects = sorted(objects,key = lambda x:int(x[2:x.index(":")]))
    object_string = "\n".join(objects)
    return object_string

scenes = list(sorted(list(os.listdir("scenes"))))
has = True
for scene in tqdm(scenes):
    if scene == "bedroom27":
        has = False
    if has:
        continue
    simple_tasks = []
    simple_missions = []
    scene_dir = os.path.join("scenes",scene)
    image_dir = os.path.join("images",scene)
    try:
        parent_labels = np.load(os.path.join(image_dir,"labels.npy"),allow_pickle=True)[None][0]
    except:
        continue
    scene_config = os.path.join(scene_dir, "config.json")
    with open(scene_config, "r") as json_file:
        metadata = json.load(json_file)
    metadata = metadata["objects"]

    parents = _build_parents(scene_dir)
    floor = _build_start_point(scene_dir)
    index_dict, reversed_index_dict, object_polygon, parents, object_positions = _build_objects(scene_dir,parents)
    objects = list(parents.keys())
    main_objects = list(set(parents.values()))
    object_dict = {name:[] for name in main_objects}
    for name,parent in parents.items():
        if name != parent:
            object_dict[parent].append(name)
    object_string = format_objects(objects, object_dict,parent_labels)
    images = os.listdir(image_dir)
    images = [os.path.join(image_dir,image) for image in images if image.endswith(".png")]
    images = [cv.imread(i) for i in images]
    # 获取图像高度（假设所有图像高度相同）
    height = images[0].shape[0]

    # 统一高度并生成黑色间隔
    black_gap = np.zeros((height, 16, 3), dtype=np.uint8)

    # 构造带间隔的图像列表
    padded_images = []
    for idx, img in enumerate(images):
        padded_images.append(img)
        if idx < len(images) - 1:
            padded_images.append(black_gap)

    # 横向拼接
    image = np.hstack(padded_images)

    def get_base64(image):
        _, buffer = cv.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    image = get_base64(image)
    image_url = {"url":f"data:image/jpeg;base64,{image}"}
    
    system = """You are an experienced humanoid robot tester. Given a scene, you can design corresponding tasks for the humanoid robot to complete based on the layout of the environment.

Specifically, the scene contains main objects and sub-objects. Each main object may contain multiple sub-objects. The scene layout is provided as multi-view images, where each main object is marked with a numerical label shown in the image using a red circle with a white number.

Your task is to generate an instruction based on the given scene, and define a set of test checkpoints to verify whether the humanoid robot has correctly followed the instruction. The instruction can be either explicit or implicit, but it should not be overly complex.

Please format your response using the following JSON structure:

{
    "prompt": "The instruction for the robot",
    "mission": [
        [
            {
                "type": "The type of checkpoint",
                // Other relevant details
            }
        ]
    ]
}

- The outer array in mission represents a sequence of time steps. The robot must complete all checkpoints in one time step before moving on to the next.

- The inner array at each time step contains checkpoints that must be completed simultaneously. All checkpoints within the same inner array must be satisfied at the same time for that step to be considered successful.

Currently supported checkpoint types:

- reach: Checks whether the robot is standing at a specific location facing a target object.
    - Fields: 
        - target (the target object)
        - direction (one of front, back, left, right)
    - Note: Currently, the target must be an object with a clearly defined orientation such as a sofa, bed, or a cabinet placed against a wall, as orientation detection relies on predefined rules.

- sit: Checks whether the robot is sitting on a specific seat.
    - Fields: 
        - target (the seat, can be sofa and bed)

- sleep: Checks whether the robot is lying on a specific bed.
    - Fields: 
        - target (the bed)

- touch: Checks whether the robot is touching a specific object.
    - Fields: 
        - target (the object)

- watch: Checks whether the robot is facing a specific object.
    - Fields: 
        - target (the object)

- place: Checks whether a specific object has been placed at a designated location, used to verify object transporting tasks.
    - Fields:
        - target: the object to be moved (only support plant container and nature shelf trinket currently)
        - at: the destination

If target or at refers to a general type of object, you can omit the numeric suffix and specify only the object category to indicate any object of that type (e.g. when "book stack1" and "book column2" are both ok, then target can be "book").

Please make sure the number of checkpoints is evenly distributed among types. In particular, AVOIDING having too many "place" tasks ( <= 1). 

Here's some examples:

[start of example A]
Current Have Checkpoints:
- reach: 0
- sit: 0
- sleep: 0
- touch: 0
- watch: 0
- place: 0

Input Scene:
- 1: bed1, containing [mattress1]
- 2: simple bookcase1, containing [book column1, book stack1, nature shelf trinkets1]

Output Plan:
{
    "prompt": "You are tired now. Get some rest. ",
    "mission": [
        [
            {
                "type": "sleep",
                "target": "bed1"
            }
        ]
    ]
}
[end of example A]

[start of example B]
Current Have Checkpoints:
- reach: 0
- sit: 1
- sleep: 1
- touch: 0
- watch: 0
- place: 0

Input Scene:
- 1: simple desk1, containing [monitor1, desk lamp1]
- 2: sofa1
- 3: kitchen cabinet1, containing [book column1, book stack1, nature shelf trinkets1, desk lamp2]

Output Plan:
{
    "prompt": "The room is too dark. ",
    "mission": [
        [
            {
                "type": "touch",
                "target": "desk lamp"
            }
        ],
        [
            {
                "type": "touch",
                "target": "desk lamp"
            }
        ]
    ]
}
[end of example B]

[start of example C]
Current Have Checkpoints:
- reach: 1
- sit: 2
- sleep: 1
- touch: 0
- watch: 0
- place: 2

Input Scene:
- 1: sofa1
- 2: large shelf1, containing [book column1, book stack1, book column2, book stack2]

Output Plan:
{
    "prompt": "Organize the bookshelf. ",
    "mission": [
        [
            {
                "type": "touch",
                "target": "book"
            }
        ],
        [
            {
                "type": "touch",
                "target": "book"
            }
        ]
    ]
}
[end of example C]

[start of example D]
Current Have Checkpoints:
- reach: 1
- sit: 1
- sleep: 1
- touch: 0
- watch: 0
- place: 1

Input Scene:
- 1: sofa1
- 2: sofa2

Output Plan:
{
    "prompt": "Wipe down the sofas. ",
    "mission": [
        [
            {
                "type": "touch",
                "target": "sofa"
            },
            {
                "type": "touch",
                "target": "sofa"
            }
        ]
    ]
}
[end of example D]

[start of example E]
Current Have Checkpoints:
- reach: 1
- sit: 0
- sleep: 1
- touch: 0
- watch: 1
- place: 1

Input Scene: // Here, assume the scene images are provided and the side table is near the sofa.
- 1: sofa1
- 2: side table1, containing [desk lamp1]
- 3: cabinet1, containing [book column1, nature shelf trinkets1, desk lamp2]

Output Plan:
{
    "prompt": "Sit on the sofa and turn on the lamp next to it at the same time. ",
    "mission": [
        [
            {
                "type": "sit",
                "target": "sofa1"
            },
            {
                "type": "touch",
                "target": "desk lamp1"
            }
        ]
    ]
}
[end of example E]
"""
    valid_plan = 0
    counts = {
        "reach": 0,
        "sit": 0,
        "sleep": 0,
        "touch": 0,
        "watch": 0,
        "place": 0
    }
    for turn in ["Alpha", "Zeta", "Gamma", "Epsilon", "Theta"]:
        if valid_plan >= 3:
            break

        user = "Current Have Checkpoints:\n" + "\n".join([f"- {k}: {v}" for k,v in counts.items()]) + f"\n\nScene ID: {turn}\n\nInput Scene:\n" + object_string + "\n\nOutput Plan:"
        
        user = [{"type": "text","text": user},
                {"type": "image_url", "image_url": image_url}]
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        response = completion.choices[0].message.content
        try:
            start = response.find('{')
            end = response.rfind('}') + 1

            if start == -1 or end == -1:
                raise ValueError("No JSON object found in response.")

            json_str = response[start:end]

            # 尝试解析为 JSON
            parsed_json = json.loads(json_str)
            for timestep in parsed_json["mission"]:
                for mission in timestep:
                    counts[mission["type"]] += 1

            os.makedirs(f"task_mixed/{scene}",exist_ok=True)
            plan_id = len(os.listdir(f"task_mixed/{scene}"))
            with open(f"task_mixed/{scene}/{plan_id}.json","w") as f:
                json.dump(parsed_json,f,indent=2)
            chat = user[0]["text"] + "\n" + response
            with open(f"task_mixed/{scene}/{plan_id}.txt","w") as f:
                f.write(chat)
            valid_plan += 1
        except Exception as e:
            continue
        