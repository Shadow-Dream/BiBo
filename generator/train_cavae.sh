export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM="false"

python train_cavae.py --cfg configs/cavae.yaml --wandb_name CAVAE