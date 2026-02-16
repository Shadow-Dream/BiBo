import os
import random
import subprocess

num_rooms = 50

# Output root directory
output_root = "outputs/blender"

# Common arguments
common_args = (
    "--task coarse "
    "-g fast_solve.gin singleroom.gin "
    "-p compose_indoors.terrain_enabled=False"
)

# Room types to generate
room_types = ["Bedroom", "LivingRoom"]

for room_type in room_types:
    for i in range(num_rooms):
        seed = random.randint(1, 999999)
        output_folder = os.path.join(output_root, f"{room_type.lower()}{i}")
        
        cmd = (
            f"python -m infinigen_examples.generate_indoors "
            f"--seed {seed} "
            f"{common_args} "
            f"--output_folder {output_folder} "
            f"-p restrict_solving.restrict_parent_rooms=['{room_type}']"
        )
        
        print(f"\n>>> Running: {cmd}\n")
        subprocess.run(cmd, shell=True, check=False)
