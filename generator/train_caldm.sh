export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM="false"

python train_caldm.py --cfg configs/caldm.yaml --wandb_name CALDM