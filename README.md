## [ICML 2026] Endowing GPT-4 with a Humanoid Body: Building the Bridge Between Off-the-Shelf VLMs and the Physical World

## 👋 Introduction

BiBo is a natural language-driven humanoid agent capable of autonomously performing diverse scene interactions. Rather than relying on costly human-scene interaction datasets, BiBo manages scene interactions by leveraging a general Vision-Language Model (such as GPT-4o). The VLM translates high-level user instructions into structured low-level commands, which are then used to drive a diffusion model that generates full-body human motion. During execution, the system also incorporates physical feedback from the environment to ensure that the motion remains controllable and realistic. In experiments, BiBo demonstrates strong task success in randomly generated scenes and enhances the quality of text-guided physical motion generation.

## 📂 Structure

- `dataset/`: Scene-interaction dataset generation and annotation pipelines, used to generate random scenes and random human-scene interaction tasks.
- `generator/`: Motion generation model that produces full-body human motion from text instructions and control parameters.
- `renderer/`: Scene renderer for multi-view rendering of scenes and objects.
- `visualizer/` (visualier): Unity scripts for motion visualization.
- `simulator/`: Simulator and planner-related code.

##  📜 TODO List

- [x] Refactor and release the dataset construction pipeline.
- [x] Refactor and release the motion generator.
- [x] Refactor and release the scene renderer.
- [x] Refactor and release the Unity visualization code.
- [ ] Refactor and release the simulator and VLM planner.
- [ ] Write an environment setup and program execution documentation.
- [ ] Organize and release the randomly generated scene-interaction dataset.

## 📝 Citation
If our paper has inspired your research or our code has been helpful in your work, we would greatly appreciate it if you could kindly cite our paper!

The ICLR version will soon be released; you can first cite our arXiv version.

```bibtex
@article{jian2025endowing,
  title={Endowing GPT-4 with a Humanoid Body: Building the Bridge Between Off-the-Shelf VLMs and the Physical World},
  author={Jian, Yingzhao and Wang, Zhongan and Yang, Yi and Fan, Hehe},
  journal={arXiv preprint arXiv:2511.00041},
  year={2025}
}
```
