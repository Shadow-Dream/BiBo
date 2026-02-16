import os
import shutil
import subprocess
import json
import time
import signal
import random
import traceback
random.seed(2025)

scene_dst = "closd/data/assets/scene"
task_dst = "closd/data/assets/task"
result_dir = "results"
final_result_dir = "results_mixed"
if os.path.exists(final_result_dir):
    shutil.rmtree(final_result_dir)
os.makedirs(final_result_dir,exist_ok=True)

max_env = 4

counts = {}
successes = {}

random.seed(2025)
scenes = os.listdir("task_mixed")
random.shuffle(scenes)

for scene_name in scenes:
    while True:
        result = subprocess.check_output(["ps", "aux"], text=True)
        has = False
        for line in result.splitlines():
            if ("render_backend.py" in line or "closd/run.py" in line) and "python" in line and "grep" not in line:
                has = True
                parts = line.split()
                pid = int(parts[1])
                print(f"Killing render_backend process, PID={pid}")
                os.kill(pid, signal.SIGKILL)  # 或者 signal.SIGTERM
                time.sleep(0.1)  # 等一下
        if not has:
            break

    print(f"="*50,flush=True)
    print(f"="*50,flush=True)
    print(f"="*50,flush=True)
    print(f"Running {scene_name}",flush=True)
    print(f"="*50,flush=True)
    print(f"="*50,flush=True)
    print(f"="*50,flush=True)

    current_final_result_dir = os.path.join(final_result_dir,scene_name)
    os.makedirs(current_final_result_dir,exist_ok=True)
    if os.path.exists("capture"):
        shutil.rmtree("capture")
    os.makedirs("capture",exist_ok=True)

    scene_src = os.path.join("scenes",scene_name)
    task_src = os.path.join("task_mixed",scene_name)
    image_src = os.path.join("images",scene_name)
    if os.path.exists("closd/assets/data/images"):
        shutil.rmtree("closd/assets/data/images")
    shutil.copytree(image_src,"closd/assets/data/images")
    
    task_files = []
    task_filenames = os.listdir(task_src)
    task_filenames = [f for f in task_filenames if f.endswith(".json")]
    task_files = sorted(task_filenames)
    task_length = len(task_files)
    if task_length == 0:
        continue

    if os.path.exists(scene_dst):
        shutil.rmtree(scene_dst)
    shutil.copytree(scene_src,scene_dst)
    
    command = [
        "/root/anaconda3/envs/infinigen/bin/python", 
        "/root/infinigen/render_backend.py",
        f"{scene_name}"
    ]
    render_process = subprocess.Popen(command, 
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL
                                      )
    
    start_task_id = 0
    current_max_env = max_env
    while start_task_id < task_length:
        try:
            end_task_id = min(task_length,start_task_id + current_max_env)
            current_task_length = end_task_id - start_task_id
            if os.path.exists(task_dst):
                shutil.rmtree(task_dst)
            os.makedirs(task_dst)
            for dst_task_id,src_task_id in enumerate(range(start_task_id,end_task_id)):
                src_taskid_path = os.path.join(task_src,task_files[src_task_id])
                dst_taskid_path = os.path.join(task_dst,f"{dst_task_id}.json")
                shutil.copy(src_taskid_path,dst_taskid_path)
            
            if os.path.exists(result_dir):
                shutil.rmtree(result_dir)
            os.makedirs(result_dir)

            # 要执行的命令
            command = [
                "python", "closd/run.py",
                "learning=im_big",
                "robot=smpl_humanoid",
                "epoch=-1",
                "test=True",
                "no_virtual_display=True",
                "headless=True",
                f"env.num_envs={current_task_length}",
                "env=closd_planner",
                "exp_name=CLoSD_multitask_finetune"
            ]
            # 启动子进程
            process = subprocess.Popen(command, 
                                    #   stdout=subprocess.DEVNULL,
                                    #   stderr=subprocess.DEVNULL
                                      )
            process.wait()
            results = os.listdir(result_dir)
            if len(results) != current_task_length:
                raise Exception("Execution Failed.")
            for result in results:
                resultp = os.path.join(result_dir,result)
                with open(resultp,"r") as f:
                    result = json.load(f)

                counts[result["total"]] = counts.get(result["total"],0) + 1
                if result["success"] > 0:
                    successes[result["total"]] = successes.get(result["total"],0) + 1
                else:
                    successes[result["total"]] = successes.get(result["total"],0)
                
                result_id = len(os.listdir(current_final_result_dir)) + 1
                resultd = os.path.join(current_final_result_dir,f"{result_id}.json")
                shutil.copy(resultp,resultd)
            
            with open("pipeline5_mixed_result.log","w") as f:
                for key in counts:
                    count = counts[key]
                    success = successes[key]
                    f.write(f"{key}: {success}/{count}, {success/count}\n")
                count = sum(counts.values())
                success = sum(successes.values())
                if count > 0:
                    f.write(f"Total: {success}/{count}, {success/count}\n")
            start_task_id = end_task_id
        except Exception as e:
            print(f"="*50,flush=True)
            print(f"="*50,flush=True)
            print(f"="*50,flush=True)
            print(f"Error occurred: {str(e)}")
            traceback.print_exc()
            current_max_env = current_max_env // 2
            if current_max_env == 0:
                current_max_env = max_env
                start_task_id += 1
                print("Execution Failed! Skip This Env")
            else:
                print("Execution Failed! Retrying...")
            print(f"="*50,flush=True)
            print(f"="*50,flush=True)
            print(f"="*50,flush=True)

    while True:
        result = subprocess.check_output(["ps", "aux"], text=True)
        has = False
        for line in result.splitlines():
            if ("render_backend.py" in line or "closd/run.py" in line) and "python" in line and "grep" not in line:
                has = True
                parts = line.split()
                pid = int(parts[1])
                print(f"Killing render_backend process, PID={pid}")
                os.kill(pid, signal.SIGKILL)  # 或者 signal.SIGTERM
                time.sleep(0.1)  # 等一下
        if not has:
            break

print("All Ok!")

    
