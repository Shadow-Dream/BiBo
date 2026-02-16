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

max_length_composite = 5
min_length_composite = 2

simple_task_num = 10
composite_task_num = 5

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
    for object_id, setting in objects.items():
        name = setting["type"]
        if name not in ignore_types and name not in layout_types and name not in decoration_types:
            name_dict[name] = name_dict.get(name,0) + 1
            name_index = name_dict[name]
            index_dict[object_id] = f"{name}{name_index}"
            bbox = np.array(setting["bbox"])
            bbox = get_convex_hull(bbox)
            if object_polygon is None:
                object_polygon = bbox
            else:
                object_polygon = object_polygon.union(bbox)

    parents = {index_dict[k]:index_dict[v] for k,v in parents.items() if k in index_dict and v in index_dict}
    reversed_index_dict = {v:k for k,v in index_dict.items()}
    return index_dict, reversed_index_dict, object_polygon, parents
    
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

scenes = list(sorted(list(os.listdir("scenes"))))
for scene in tqdm(scenes):
    simple_tasks = []
    simple_missions = []
    composite_tasks = []
    composite_missions = []
    scene_dir = os.path.join("scenes",scene)

    scene_config = os.path.join(scene_dir, "config.json")
    with open(scene_config, "r") as json_file:
        metadata = json.load(json_file)
    metadata = metadata["objects"]

    parents = _build_parents(scene_dir)
    floor = _build_start_point(scene_dir)
    index_dict, reversed_index_dict, object_polygon, parents = _build_objects(scene_dir,parents)
    objects = list(parents.keys())
    main_objects = list(set(parents.values()))
    
    sofas = [o for o in objects if "sofa" in o]
    random.shuffle(sofas)
    sofas = sofas[:3]
    for sofa in sofas:
        simple_tasks.append(f"Sitting on {sofa}.")
        simple_missions.append({"type":"sit","target":sofa})

    lamps = [o for o in objects if "desk lamp" in o]
    random.shuffle(lamps)
    lamps = lamps[:3]
    for lamp in lamps:
        simple_tasks.append(f"Turning on the {lamp}.")
        simple_missions.append({"type":"touch","target":lamp})

    tvs = [o for o in objects if ("t v" in o and "stand" not in o) or "monitor" in o]
    random.shuffle(tvs)
    tvs = tvs[:3]
    for tv in tvs:
        simple_tasks.append(f"Watching {tv}.")
        simple_missions.append({"type":"look","target":tv})

    action_num = int((simple_task_num - len(simple_tasks))*random.random())
    touch_num = simple_task_num - len(simple_tasks) - action_num

    containers = [o for o in objects 
    if ("desk" in o and "lamp" not in o) 
    or ("t v stand" in o)
    or ("table" in o)
    or ("cabinet" in o)]

    children = []
    for container in containers:
        children += [k for k in objects if parents[k] == container and "lamp" not in k and "t v" not in k and "monitor" not in k and container != k]
    random.shuffle(children)
    children = children[:touch_num]
    action_num = simple_task_num - len(simple_tasks) - len(children)

    for child in children:
        if "trinkets" in child:
            simple_tasks.append(f"Playing with {child} with hand.")
        elif "book" in child:
            simple_tasks.append(f"Wiping off the dust from {child}.")
        else:
            simple_tasks.append(f"Putting the hand on {child}.")
        simple_missions.append({"type":"touch","target":child})
    
    targets = list(set(parents.values()))
    random.shuffle(targets)
    targets = targets[:action_num]

    for target in targets:
        simple_tasks.append(f"Standing still besides {target}.")
        simple_missions.append({"type":"stand","target":target})

    simple_missions = [[m] for m in simple_missions]



    sofas = [o for o in objects if "sofa" in o]
    for sofa in sofas:
        composite_tasks.append(f"Sitting on {sofa}.")
        composite_missions.append({"type":"sit","target":sofa})

    tvs = [o for o in objects if ("t v" in o and "stand" not in o) or "monitor" in o]
    for tv in tvs:
        composite_tasks.append(f"Watching {tv}.")
        composite_missions.append({"type":"look","target":tv})

    containers = [o for o in objects 
    if ("desk" in o and "lamp" not in o) 
    or ("t v stand" in o)
    or ("table" in o)
    or ("cabinet" in o)]

    children = []
    for container in containers:
        children += [k for k in objects if parents[k] == container and "t v" not in k and "monitor" not in k and container != k]
    ck = max(len(composite_tasks),3)
    random.shuffle(children)
    children = children[:ck]

    for child in children:
        if "trinkets" in child:
            composite_tasks.append(f"Playing with {child} with hand.")
        elif "book" in child:
            composite_tasks.append(f"Wiping off the dust from {child}.")
        elif "lamp" in child:
            composite_tasks.append(f"Turning on the {child}.",)
        else:
            composite_tasks.append(f"Putting the hand on {child}.")
        composite_missions.append({"type":"touch","target":child})
    
    targets = list(set(parents.values()))
    random.shuffle(targets)
    targets = targets[:ck]

    for target in targets:
        composite_tasks.append(f"Standing still besides {target}.")
        composite_missions.append({"type":"stand","target":target})

    if len(composite_tasks) < max_length_composite:
        targets = list(set(parents.values()))
        random.shuffle(targets)
        targets = targets[:max_length_composite - len(composite_tasks)]
        for target in targets:
            if random.random() > 0.5:
                composite_tasks.append(f"Putting the hand on {target}.")
                composite_missions.append({"type":"touch","target":target})
            else:
                composite_tasks.append(f"Standing still besides {target}.")
                composite_missions.append({"type":"stand","target":target})
    
    final_composite_tasks = []
    final_composite_missions = []
    for _ in range(composite_task_num):
        moves = min(random.randint(min_length_composite,max_length_composite),len(composite_tasks))
        col = random.sample(list(zip(composite_tasks,composite_missions)),moves)
        tasks,missions = zip(*col)
        final_composite_tasks.append(" ".join(tasks))
        final_composite_missions.append(list(missions))
    
    composite_tasks = final_composite_tasks
    composite_missions = final_composite_missions
    tasks = simple_tasks + composite_tasks
    missions = simple_missions + composite_missions

    tasks = [{"prompt":task,"mission":mission} for task,mission in zip(tasks,missions)]
    task_dir = os.path.join("tasks",scene)
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    os.makedirs(task_dir)
    for env_id,task in enumerate(tasks):
        with open(os.path.join(task_dir,f"{env_id}.json"),"w") as f:
            json.dump(task,f)