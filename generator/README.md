## Generator

This module contains the diffusion-based motion generation models used in BiBo.

- **CALDM**: pure text-to-motion diffusion (T2M).
- **CACLDM**: controllable motion diffusion (text + control parameters).
- **CAVAE**: motion VAE backbone used by both diffusion models.

This code is implemented with reference to
[MotionLCM](https://github.com/GuyTevet/CLoSD) and
[CLoSD](https://github.com/Dai-Wenxun/MotionLCM).

### Assets

- Download [metadata and checkpoints]((https://huggingface.co/Behavia/BEHAVIA/tree/main/generator)) to `./data` and `./checkpoints`.
- Prepare dependencies and tiny data by running scripts under `./prepare`
- Refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D) and move the dataset under `./datasets/humanml3d`

### Training

- **CAVAE:** Run `python train_cavae.py --cfg configs/cavae.yaml`
    
- **CALDM:**

    1. Set `TRAIN.PRETRAINED` in `configs/caldm.yaml` to your trained CAVAE checkpoint.
    2. Run `python train_caldm.py --cfg configs/caldm.yaml`

- **CACLDM:**

    1. Set `TRAIN.PRETRAINED` in `configs/caldm.yaml` to your trained CAVAE checkpoint.
    2. Run: `python train_cacldm.py --cfg configs/cacldm.yaml`

### Inference

- **CALDM:**

    1. Set `TRAIN.PRETRAINED` in `configs/caldm.yaml` to the CALDM checkpoint you want to use.
    2. Edit prompt settings in `inf_caldm.py` (e.g., `texts`, `gen_len`) if needed.
    3. Run: `python inf_caldm.py --cfg configs/caldm.yaml
`

- **CACLDM:**
    1. Set `TRAIN.PRETRAINED` in `configs/cacldm.yaml` to the CACLDM checkpoint you want to use.
    2. Edit prompt and controls in `inf_cacldm.py`.
   - Text prompt: `texts = [...]`
   - Control params: `compute_cond({...})`
    3. Run: `python inf_cacldm.py --cfg configs/cacldm.yaml`