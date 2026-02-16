import argparse
import base64
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import networkx as nx
import numpy as np
import shapely as sl
import trimesh
from openai import OpenAI
from scipy.spatial import ConvexHull
from tqdm import tqdm


ALLOWED_TYPES = {"watch", "sit", "sleep", "touch", "lift"}
TYPE_ALIASES = {"look": "watch"}
TARGET_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]*(?:\d+|\[\d+\])?\*?$")
SCENE_IDS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Theta"]

TASK_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prompt", "mission"],
    "properties": {
        "prompt": {"type": "string", "minLength": 1},
        "mission": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "target"],
                    "properties": {
                        "type": {"type": "string", "enum": sorted(list(ALLOWED_TYPES))},
                        "target": {
                            "type": "string",
                            "pattern": r"^[A-Za-z][A-Za-z0-9 _-]*(?:\d+|\[\d+\])?\*?$",
                        },
                    },
                },
            },
        },
    },
}

IGNORE_TYPES = {"ceiling", "ceiling light", "exterior"}
LAYOUT_TYPES = set()
DECORATION_TYPES = {
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
    "door",
}

SYSTEM_PROMPT = """You are an intelligent task generator.
Your goal is to create interaction tasks that a humanoid agent can perform within a given scene.

Each task must be JSON with two fields:
- prompt: natural language instruction of the task, can be both abstract and concrete.
- mission: a two-level array of interaction objectives. The first level is sequential, while the second level is simultaneous.

Each interaction objective contains:
- type: one of [watch, sit, sleep, touch, lift]
- target: target object in "category[index][*]" style.
  - category is required.
  - [index] is optional (if omitted, it means all uninteracted objects of that category).
  - [*] marks an object that should not be omitted in subsequent matching.

Task difficulty criteria:
- simple: contain <4 steps (including navigating to another object between two interactions).
- medium: 4-10 steps, or contains dynamic object manipulation (e.g., lift, transport).
- hard: >10 steps, or contains simultaneous interactions with multiple objects.

Output rules:
- Analyze first, then output final task.
- Final answer must be enclosed in >>> and <<<.
- Output only one task JSON in the final answer.
"""

SIMPLE_EXAMPLE = """{
  "prompt": "Take a short break.",
  "mission": [
    [
      {"type": "sit", "target": "sofa1"}
    ],
    [
      {"type": "watch", "target": "monitor1"}
    ]
  ]
}"""

MEDIUM_EXAMPLE = """{
  "prompt": "Tidy the desk corner.",
  "mission": [
    [
      {"type": "lift", "target": "nature shelf trinket1*"}
    ],
    [
      {"type": "touch", "target": "desk1"}
    ],
    [
      {"type": "watch", "target": "monitor1"}
    ]
  ]
}"""

HARD_EXAMPLE = """{
  "prompt": "Relax while checking the room setup.",
  "mission": [
    [
      {"type": "sit", "target": "sofa1"},
      {"type": "touch", "target": "side table1*"}
    ],
    [
      {"type": "watch", "target": "monitor1"},
      {"type": "touch", "target": "desk lamp1"}
    ]
  ]
}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=str, default="scenes")
    parser.add_argument("--image-root", type=str, default="images")
    parser.add_argument("--output-root", type=str, default="task_mixed_new")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tasks-per-scene", type=int, default=3)
    parser.add_argument("--max-attempts-per-task", type=int, default=5)
    parser.add_argument("--scene-prefix", type=str, default="")
    parser.add_argument("--seed", type=int, default=2026)

    # Request specification
    parser.add_argument("--difficulty", type=str, default="simple", choices=["simple", "medium", "hard"])
    parser.add_argument("--prompt-style", type=str, default="abstract", choices=["abstract", "concrete"])
    parser.add_argument("--interactions", type=int, default=2)
    parser.add_argument("--allow-simultaneous", action="store_true")
    parser.add_argument("--allow-dynamic-object-manipulation", action="store_true")
    parser.add_argument("--structured-output", action="store_true")
    return parser.parse_args()


def get_convex_hull(points: np.ndarray):
    try:
        hull = ConvexHull(points)
        contour = sl.Polygon(points[hull.vertices])
        contour = contour.buffer(0.3, join_style="mitre")
        return contour
    except Exception:
        return None


def _build_parents(scene_dir: str) -> Dict[str, str]:
    scene_config = os.path.join(scene_dir, "config.json")
    with open(scene_config, "r") as json_file:
        metadata = json.load(json_file)
    objects = metadata["objects"]

    objects = {
        name: setting
        for name, setting in objects.items()
        if setting["type"] not in IGNORE_TYPES
        and setting["type"] not in LAYOUT_TYPES
        and setting["type"] not in DECORATION_TYPES
    }
    if not objects:
        return {}

    object_list = list(objects.keys())
    bboxes = [np.array(objects[name]["bbox"])[None, :] for name in object_list]
    bboxes = np.concatenate(bboxes, 0)
    positions = [np.array(objects[name]["position"])[None, None, :] for name in object_list]
    positions = np.concatenate(positions, 0)
    bboxes += positions
    height = bboxes[:, :, 2].min(axis=-1)
    bboxes = bboxes[:, :, :2]
    min_axis, max_axis = bboxes.min(1), bboxes.max(1)
    pairs = np.stack(np.meshgrid(np.arange(len(object_list)), np.arange(len(object_list))), -1)
    pair_min_axis = min_axis[pairs]
    pair_max_axis = max_axis[pairs]
    pair_min_axis = pair_min_axis.max(-2)
    pair_max_axis = pair_max_axis.min(-2)
    pair_length = np.maximum(pair_max_axis - pair_min_axis, 0)
    pair_intersection = pair_length.prod(-1)
    area = (max_axis - min_axis).prod(-1)
    pair_min_area = area[pairs].min(-1)
    pairs = (pair_intersection / pair_min_area) > 0.75
    start, end = np.where(pairs)
    graph = nx.Graph()
    graph.add_edges_from(list(zip(start, end)))
    object_sets = list(nx.connected_components(graph))
    parents: Dict[str, str] = {}

    for object_set in object_sets:
        object_set = list(object_set)
        parent = object_set[height[object_set].argmin()]
        for object_index in object_set:
            parents[object_list[object_index]] = object_list[parent]
    return parents


def _build_objects(scene_dir: str, parents: Dict[str, str]):
    scene_config = os.path.join(scene_dir, "config.json")
    with open(scene_config, "r") as json_file:
        metadata = json.load(json_file)
    objects = metadata["objects"]

    name_dict: Dict[str, int] = {}
    index_dict: Dict[str, str] = {}
    object_polygon = None
    object_positions: Dict[str, np.ndarray] = {}
    for object_id, setting in objects.items():
        name = setting["type"]
        if name in IGNORE_TYPES or name in LAYOUT_TYPES or name in DECORATION_TYPES:
            continue
        name_dict[name] = name_dict.get(name, 0) + 1
        name_index = name_dict[name]
        object_ind = f"{name}{name_index}"
        index_dict[object_id] = object_ind

        bbox = np.array(setting["bbox"])
        convex = get_convex_hull(bbox)
        if convex is not None:
            if object_polygon is None:
                object_polygon = convex
            else:
                object_polygon = object_polygon.union(convex)

        object_positions[object_ind] = (
            np.array(setting["position"]) + np.array(setting["bbox"]).mean(0)
        )

    parents = {
        index_dict[k]: index_dict[v]
        for k, v in parents.items()
        if k in index_dict and v in index_dict
    }
    reversed_index_dict = {v: k for k, v in index_dict.items()}
    return index_dict, reversed_index_dict, object_polygon, parents, object_positions


def load_parent_labels(image_dir: str) -> Dict[str, str]:
    labels_path = os.path.join(image_dir, "labels.npy")
    labels = np.load(labels_path, allow_pickle=True)
    if isinstance(labels, np.ndarray):
        if labels.shape == ():
            labels = labels.item()
        elif labels.size > 0:
            labels = labels.reshape(-1)[0]
    if not isinstance(labels, dict):
        return {}
    return {str(k): str(v) for k, v in labels.items()}


def label_sort_key(label: str) -> Tuple[int, str]:
    try:
        return int(label), label
    except Exception:
        return 10**9, label


def format_coordinate(coordinate: np.ndarray) -> str:
    return f"({coordinate[0]:.1f}, {coordinate[1]:.1f})"


def format_scene_objects(
    object_dict: Dict[str, List[str]],
    object_positions: Dict[str, np.ndarray],
    parent_labels: Dict[str, str],
) -> str:
    reverse = {v: str(k) for k, v in parent_labels.items()}
    entries: List[Tuple[str, str]] = []
    fallback_idx = 1

    for parent_name in sorted(object_dict.keys()):
        if parent_name == "origin":
            continue
        label = reverse.get(parent_name)
        if label is None:
            label = f"U{fallback_idx}"
            fallback_idx += 1

        position = object_positions.get(parent_name)
        position_text = f" {format_coordinate(position)}" if position is not None else ""
        contain = sorted(object_dict[parent_name])

        line = f"- {label}: {parent_name}{position_text}"
        if contain:
            line += f", containing: [{', '.join(contain)}]"
        entries.append((label, line))

    entries.sort(key=lambda x: label_sort_key(x[0]))
    return "\n".join([line for _, line in entries])


def build_multiview_payload(image_dir: str) -> Optional[Dict[str, str]]:
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
    )
    images = [cv.imread(os.path.join(image_dir, f)) for f in image_files]
    images = [img for img in images if img is not None]
    if not images:
        return None

    min_height = min(img.shape[0] for img in images)
    resized = []
    for img in images:
        if img.shape[0] != min_height:
            width = int(img.shape[1] * (min_height / img.shape[0]))
            img = cv.resize(img, (width, min_height))
        resized.append(img)

    black_gap = np.zeros((min_height, 16, 3), dtype=np.uint8)
    padded_images: List[np.ndarray] = []
    for idx, img in enumerate(resized):
        padded_images.append(img)
        if idx < len(resized) - 1:
            padded_images.append(black_gap)

    merged = np.hstack(padded_images)
    ok, buffer = cv.imencode(".png", merged)
    if not ok:
        return None
    b64 = base64.b64encode(buffer).decode("utf-8")
    return {"url": f"data:image/png;base64,{b64}"}


def build_user_prompt(scene_id: str, object_string: str, spec: Dict[str, object]) -> str:
    difficulty = str(spec["difficulty"])
    prompt_style = str(spec["prompt_style"])
    interactions = int(spec["interactions"])
    allow_simultaneous = bool(spec["allow_simultaneous"])
    allow_dynamic = bool(spec["allow_dynamic_object_manipulation"])

    simultaneous_text = (
        "Simultaneous interaction is allowed."
        if allow_simultaneous
        else "No simultaneous interaction."
    )
    dynamic_text = (
        "Dynamic object manipulation is allowed."
        if allow_dynamic
        else "No dynamic object manipulation."
    )
    article = "an" if difficulty[0].lower() in "aeiou" else "a"

    return f"""Example of simple task:
{SIMPLE_EXAMPLE}

Example of medium task:
{MEDIUM_EXAMPLE}

Example of hard task:
{HARD_EXAMPLE}

The multi-view scene images are provided, each image contain object labels, corresponding to:
{object_string}

Scene ID: {scene_id}

You are currently generating {article} {difficulty} task, with {prompt_style} prompt and {interactions} interactions. {simultaneous_text} {dynamic_text}
Please analyze before outputting the final task, and enclose your final answer in >>> and <<<.
"""


def extract_json_from_response(response: str) -> Optional[dict]:
    marker_match = re.search(r">>>\s*(.*?)\s*<<<", response, flags=re.S)
    candidate = marker_match.group(1).strip() if marker_match else response.strip()

    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
        candidate = candidate.strip()

    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = candidate[start : end + 1]

    try:
        return json.loads(candidate)
    except Exception:
        return None


def normalize_target(target: str) -> str:
    target = target.strip()
    # Fix common pattern produced by some models: sofa[1][*] -> sofa[1]*
    target = re.sub(r"\[(\d+)\]\[\*\]$", r"[\1]*", target)
    # Fix category1[*] -> category1*
    target = re.sub(r"(\d+)\[\*\]$", r"\1*", target)
    return target


def normalize_plan(plan: dict) -> Optional[dict]:
    if not isinstance(plan, dict):
        return None
    prompt = plan.get("prompt", "")
    mission = plan.get("mission", [])

    if not isinstance(mission, list):
        return None

    normalized_mission: List[List[Dict[str, str]]] = []
    for step in mission:
        if isinstance(step, dict):
            step = [step]
        if not isinstance(step, list):
            return None
        normalized_step: List[Dict[str, str]] = []
        for objective in step:
            if not isinstance(objective, dict):
                return None
            obj_type = str(objective.get("type", "")).strip().lower()
            obj_type = TYPE_ALIASES.get(obj_type, obj_type)
            target = normalize_target(str(objective.get("target", "")))
            normalized_step.append({"type": obj_type, "target": target})
        normalized_mission.append(normalized_step)
    return {"prompt": str(prompt).strip(), "mission": normalized_mission}


def count_interactions(mission: List[List[Dict[str, str]]]) -> int:
    return sum(len(step) for step in mission)


def validate_plan(plan: dict, spec: Dict[str, object]) -> Tuple[bool, str]:
    prompt = plan.get("prompt")
    mission = plan.get("mission")
    if not isinstance(prompt, str) or not prompt.strip():
        return False, "Invalid prompt."
    if not isinstance(mission, list) or len(mission) == 0:
        return False, "Mission must be a non-empty list."

    allow_simultaneous = bool(spec["allow_simultaneous"])
    allow_dynamic = bool(spec["allow_dynamic_object_manipulation"])
    interactions_target = int(spec["interactions"])

    for step in mission:
        if not isinstance(step, list) or len(step) == 0:
            return False, "Each mission step must be a non-empty list."
        if not allow_simultaneous and len(step) > 1:
            return False, "Simultaneous interaction is disabled."
        for objective in step:
            if not isinstance(objective, dict):
                return False, "Invalid mission objective."
            obj_type = objective.get("type", "")
            target = objective.get("target", "")
            if obj_type not in ALLOWED_TYPES:
                return False, f"Unsupported objective type: {obj_type}"
            if not allow_dynamic and obj_type == "lift":
                return False, "Dynamic manipulation is disabled, but lift is present."
            if not isinstance(target, str) or not target.strip():
                return False, "Target must be a non-empty string."
            if not TARGET_PATTERN.match(target.strip()):
                return False, f"Invalid target format: {target}"

    interactions = count_interactions(mission)
    if interactions != interactions_target:
        return False, f"Expected {interactions_target} interactions, got {interactions}."
    return True, ""


def count_existing_json(output_dir: str) -> int:
    if not os.path.exists(output_dir):
        return 0
    return len([f for f in os.listdir(output_dir) if f.endswith(".json")])


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    client = OpenAI()

    spec = {
        "difficulty": args.difficulty,
        "prompt_style": args.prompt_style,
        "interactions": args.interactions,
        "allow_simultaneous": args.allow_simultaneous,
        "allow_dynamic_object_manipulation": args.allow_dynamic_object_manipulation,
    }

    os.makedirs(args.output_root, exist_ok=True)
    scenes = sorted(os.listdir(args.scene_root))
    if args.scene_prefix:
        scenes = [scene for scene in scenes if scene.startswith(args.scene_prefix)]

    for scene in tqdm(scenes):
        scene_dir = os.path.join(args.scene_root, scene)
        image_dir = os.path.join(args.image_root, scene)
        if not os.path.isdir(scene_dir) or not os.path.isdir(image_dir):
            continue
        if not os.path.exists(os.path.join(image_dir, "labels.npy")):
            continue

        try:
            parent_labels = load_parent_labels(image_dir)
            parents = _build_parents(scene_dir)
            if not parents:
                continue
            _, _, _, parents, object_positions = _build_objects(scene_dir, parents)
            if not parents:
                continue

            main_objects = sorted(set(parents.values()))
            object_dict = {name: [] for name in main_objects}
            for name, parent in parents.items():
                if name != parent:
                    object_dict[parent].append(name)
            object_string = format_scene_objects(object_dict, object_positions, parent_labels)
            image_payload = build_multiview_payload(image_dir)
            if image_payload is None:
                continue
        except Exception as exc:
            print(f"[{scene}] Failed to build scene context: {exc}")
            continue

        out_dir = os.path.join(args.output_root, scene)
        os.makedirs(out_dir, exist_ok=True)
        existing = count_existing_json(out_dir)
        if existing >= args.tasks_per_scene:
            continue

        valid_plans = existing
        max_attempts = args.tasks_per_scene * args.max_attempts_per_task
        attempts = 0
        while valid_plans < args.tasks_per_scene and attempts < max_attempts:
            attempts += 1
            scene_id = f"{random.choice(SCENE_IDS)}-{attempts}"
            user_text = build_user_prompt(scene_id, object_string, spec)
            user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": image_payload},
            ]

            request_kwargs = {
                "model": args.model,
                "temperature": args.temperature,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            }
            if args.structured_output:
                request_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "scene_task",
                        "strict": True,
                        "schema": TASK_JSON_SCHEMA,
                    },
                }

            try:
                completion = client.chat.completions.create(**request_kwargs)
                response = completion.choices[0].message.content or ""
            except Exception as exc:
                if args.structured_output:
                    # Fallback for providers that do not support response_format/json_schema.
                    print(f"[{scene}] Structured output unavailable, fallback to plain mode: {exc}")
                    request_kwargs.pop("response_format", None)
                    try:
                        completion = client.chat.completions.create(**request_kwargs)
                        response = completion.choices[0].message.content or ""
                    except Exception as retry_exc:
                        print(f"[{scene}] OpenAI request failed: {retry_exc}")
                        continue
                else:
                    print(f"[{scene}] OpenAI request failed: {exc}")
                    continue

            parsed = extract_json_from_response(response)
            parsed = normalize_plan(parsed) if parsed is not None else None
            if parsed is None:
                continue

            ok, reason = validate_plan(parsed, spec)
            if not ok:
                print(f"[{scene}] Invalid task skipped: {reason}")
                continue

            plan_id = count_existing_json(out_dir)
            json_path = os.path.join(out_dir, f"{plan_id}.json")
            txt_path = os.path.join(out_dir, f"{plan_id}.txt")

            with open(json_path, "w") as f:
                json.dump(parsed, f, indent=2)

            with open(txt_path, "w") as f:
                f.write("[SYSTEM]\n")
                f.write(SYSTEM_PROMPT)
                f.write("\n\n[USER]\n")
                f.write(user_text)
                f.write("\n\n[RAW_RESPONSE]\n")
                f.write(response)

            valid_plans += 1


if __name__ == "__main__":
    main()
