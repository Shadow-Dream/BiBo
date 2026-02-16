import os
import numpy as np
import sys
import cv2 as cv

import multiprocessing as mp
import subprocess
import time
import requests
from flask import Flask, request, jsonify, send_from_directory
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from orient_anything.vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor
from orient_anything.utils import *
from orient_anything.inference import *
from orient_anything.paths import *

def write_log(text):
    print(text,flush=True)
    with open("render_backend_remote.log","a") as f:
        f.write(f"{text}\n")

# '''
# Pre Clean Up
# '''
# subprocess.run("bash ./render_backend_remote_cleanup.sh",text=True,shell=True)

'''
Init Orient Anything Network
'''
ckpt_path = './checkpoints/models--Viglong--Orient-Anything/snapshots/5249ecae5cf2b8371874a88e9ab766ce81760242/croplargeEX2/dino_weight.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dino = DINOv2_MLP(
    dino_mode   = 'large',
    in_dim      = 1024,
    out_dim     = 360+180+180+2,
    evaluate    = True,
    mask_dino   = False,
    frozen_back = False
)

dino.eval()
dino.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
dino = dino.to(device)
val_preprocess = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir='./checkpoints', local_files_only=True)

def calculate_object_orientation(env_id,target):
    def resize_foreground(image, ratio = 0.85):
        alpha = np.where(image[..., 3] > 0)
        y1, y2, x1, x2 = (
            alpha[0].min(),
            alpha[0].max(),
            alpha[1].min(),
            alpha[1].max(),
        )
        fg = image[y1:y2, x1:x2]
        size = max(fg.shape[0], fg.shape[1])
        ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
        ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
        new_image = np.pad(
            fg,
            ((ph0, ph1), (pw0, pw1), (0, 0)),
            mode="constant",
            constant_values=((0, 0), (0, 0), (0, 0)),
        )

        new_size = int(new_image.shape[0] / ratio)
        ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
        ph1, pw1 = new_size - size - ph0, new_size - size - pw0
        new_image = np.pad(
            new_image,
            ((ph0, ph1), (pw0, pw1), (0, 0)),
            mode="constant",
            constant_values=((0, 0), (0, 0), (0, 0)),
        )
        return new_image
    
    def get_orientation(image):
        origin_image = Image.fromarray(image).convert('RGB')
        angles = get_3angle(origin_image, dino, val_preprocess, device)
        azimuth     = float(angles[0])
        polar       = float(angles[1])
        rotation    = float(angles[2])
        confidence  = float(angles[3])
        return azimuth, polar, rotation, confidence

    images = os.listdir(f"capture/{env_id}/{target}")
    masks = [image for image in images if "mask.png" in image and "rgb" not in image]
    tpls = [image for image in images if "tpl.png" in image]

    masks = sorted(masks)
    tpls = sorted(tpls)

    rolls = [45 * i for i in range(8)]
    roll_weights = [0 for _ in range(8)]
    for roll, mask, tpl in zip(rolls, masks, tpls):
        debug = mask.replace("mask","debug")
        mask = cv.imread(os.path.join("capture",str(env_id), target, mask))
        tpl = cv.imread(os.path.join("capture",str(env_id), target, tpl))
        border = cv.GaussianBlur(mask, (5, 5), 0)
        alpha = mask[..., :1]
        border = border[..., :1]
        alpha[(alpha == 0) & (border > 0)] = border[(alpha == 0) & (border > 0)]
        tpl = np.concatenate([tpl, alpha], axis=-1)
        tpl = resize_foreground(tpl)
        cv.imwrite(os.path.join("capture",str(env_id), target, debug), tpl)
        
        azimuth, polar, rotation, confidence = get_orientation(tpl)
        confidence = confidence * max(20 - abs(rotation),0) / 20
        confidence = confidence * max(10 - max(abs(polar - 20) - 20,0),0) / 10
        roll = 360 + roll - azimuth
        roll = round(roll / 45) % 8
        roll_weights[roll] += confidence

    instance = {}
    max_weight = max(roll_weights)
    max_index = roll_weights.index(max_weight)
    submax_weight = max(roll_weights[:max_index] + roll_weights[max_index+1:])
    max_ratio = max_weight / (sum(roll_weights) + 1e-5)
    if max_weight > 2 and max_weight / (submax_weight + 1e-5) > 1.5 and max_ratio > 0.5:
        instance["front"] = max_index * 45
        instance["end"] = None
    else:
        side_weights = [roll_weights[i] + roll_weights[(i + 4)%8] for i in range(4)]
        max_weight = max(side_weights)
        max_index = side_weights.index(max_weight)
        submax_weight = max(side_weights[:max_index] + side_weights[max_index+1:])
        max_ratio = max_weight / (sum(side_weights) + 1e-5)
        if max_weight > 3 and max_weight / (submax_weight + 1e-5) > 2 and max_ratio > 0.5:
            instance["end"] = max_index * 45
            instance["front"] = None
        else:
            instance["end"] = None
            instance["front"] = None
    return instance

'''
Init Backend Server
'''
if len(sys.argv) > 1:
    blender_path =str(sys.argv[1])
else:
    blender_path = "demo"

docker_password = os.getenv("RENDER_BACKEND_PASSWORD", "change_me")
if docker_password == "change_me":
    write_log("RENDER_BACKEND_PASSWORD is not set. Using default non-sensitive password.")

request_queue = mp.Queue()
result_queue = mp.Queue()
token = 0
tasks = {}

app = Flask(__name__)

@app.route('/info', methods=['GET'])
def info():
    response = {"name":"blender"}
    return jsonify(response)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    filename = filename.replace("<bar>","/").replace("<space>"," ")
    return send_from_directory("capture",filename,as_attachment=True)

@app.route('/list/<filename>', methods=['GET'])
def list_files(filename):
    filename = filename.replace("<bar>","/").replace("<space>"," ")
    files = os.listdir(os.path.join("capture",filename))
    return jsonify({"files":files})

@app.route('/poll', methods=['POST'])
def poll():
    data = request.json
    while not result_queue.empty():
        result = result_queue.get()
        token = result["token"]
        labels = result.get("labels",{})
        orientations = result.get("orientations",{})
        tasks[token] = {
            "labels":labels,
            "orientations":orientations
        }
    token = data["token"]
    if tasks.get(token,None) is not None:
        response = {"done":True}
        response.update(tasks[token])
    else:
        response = {"done":False}
    return jsonify(response)

@app.route('/capture', methods=['POST'])
def capture():
    global token
    data = request.json
    token += 1
    tasks[token] = None
    request_queue.put({
        "token":token,
        "method":"capture",
        "data":data
    })
    response = {"token":token}
    return jsonify(response)

def backend():
    app.run(host='127.0.0.1', port=14000)

'''
Start Render Processes
'''

process = mp.Process(target=backend)
process.start()

instance_processes = []
urls = []
for instance_index in range(8):
    port = 14001 + instance_index
    gpus = instance_index % 3
    command = [
        "docker", "run",
        "--entrypoint", "bash",
        "--privileged",
        "--gpus", f"device={gpus}",
        "--tmpfs", "/dev/shm:rw",
        "-e", "TZ=UTC",
        "-e", "DISPLAY_SIZEW=1",
        "-e", "DISPLAY_SIZEH=1",
        "-e", "DISPLAY_REFRESH=1",
        "-e", "DISPLAY_DPI=1",
        "-e", "DISPLAY_CDEPTH=1",
        "-e", f"PASSWD={docker_password}",
        "-e", "SELKIES_ENCODER=nvh264enc",
        "-e", "SELKIES_VIDEO_BITRATE=8000",
        "-e", "SELKIES_FRAMERATE=60",
        "-e", "SELKIES_AUDIO_BITRATE=128000",
        "-e", f"SELKIES_BASIC_AUTH_PASSWORD={docker_password}",
        "-v", "/root/infinigen/:/root/infinigen",
        "-v", "/usr/local/cuda-11.8:/usr/local/cuda-11.8",
        "-v", "/root/miniconda3:/root/miniconda3",
        "-p", f"{port}:{port}",
        "ghcr.io/selkies-project/nvidia-egl-desktop:latest",
        "-c", f"/root/infinigen/render_backend_remote_instance.sh {blender_path} {port} {instance_index}"
    ]
    with open(f"logs/render_backend_remote_instance_{instance_index}.txt", "w") as log_file:
        instance_process = subprocess.Popen(command, stdout=log_file, stderr=log_file)
    instance_processes.append(instance_process)
    urls.append(f"http://localhost:{port}")

def fetch(pair):
    url,data = pair
    try:
        response = requests.post(url + "/capture", json = data, timeout=60)
        return response.json()
    except Exception as e:
        write_log(e)
        return {"success":False}

'''
Main Loop
'''

while True:
    if request_queue.empty():
        time.sleep(0.1)
        continue
    request = request_queue.get()
    method = request["method"]
    token = request["token"]
    data = request["data"]
    env_id = data["env_id"]
    target = data["target"]
    save_dir = os.path.join("capture",str(env_id))
    os.makedirs(f"{save_dir}/{target}",exist_ok=True)
    np.save("capture/data.npy",data)
    subprocess.run("chmod -R 777 capture",text = True,shell = True)

    for i in range(8):
        if os.path.exists(f"{save_dir}/{target}/{i}_meta.npy"):
            os.remove(f"{save_dir}/{target}/{i}_meta.npy")

    '''
    Start Render
    '''
    for i in range(8):
        with open(f"capture/{i}.lock","w") as f:
            f.write("")
    write_log(f"Render {target}")
    
    '''
    Calculate Orientation
    '''
    while True:
        files = os.listdir(f"capture")
        files = [f for f in files if f.endswith(".lock")]
        if len(files) == 0:
            break
        time.sleep(0.1)
    
    orientations = calculate_object_orientation(env_id,target)
    
    '''
    Done Render Object Images
    '''
    while True:
        files = os.listdir(f"{save_dir}/{target}")
        files = [f for f in files if f.endswith("_meta.npy")]
        if len(files) == 8:
            break
        time.sleep(0.1)
    
    results = []
    for i in range(8):
        result = np.load(f"{save_dir}/{target}/{i}_meta.npy",allow_pickle=True)[None][0]
        results.append(result)
    
    images = sum([result["images"] for result in results],[])
    labels = {}
    for result in results:
        labels.update(result["labels"])

    if images:
        if len(images) > 1 and len(images) < 8:
            indices = [int(image.split("/")[-1].split("_rgb")[0]) for image in images]
            max_index = indices[0]
            min_index = indices[0]
            while (max_index + 1) % 8 in indices:
                    max_index = (max_index + 1) % 8
                
            while (min_index + 7) % 8 in indices:
                    min_index = (min_index + 7) % 8

            images = []
            if (max_index - min_index + 8) % 8 + 1 == len(images):
                for i in range((max_index - min_index + 8) % 8 + 1):
                    i = (i + min_index) % 8
                    images.append(f"{save_dir}/{target}/{i}_rgb.png")
            else:
                for i in range(8):
                    i = (i + min_index) % 8
                    if i in indices:
                        images.append(f"{save_dir}/{target}/{i}_rgb.png")
        else:
            images = sorted(images)

        buffer = []
        for image in images:
            img = cv.imread(image)
            if buffer:
                img = np.concatenate([np.ones([512,8,3])*255, img], axis=1)
            buffer.append(img)
        image = np.concatenate(buffer, axis=1)
        cv.imwrite(f"{save_dir}/{target}_rgb.png", image)

    result_queue.put({"token":token,"labels":labels,"orientations":orientations})
    write_log("Done")
