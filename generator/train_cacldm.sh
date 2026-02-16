export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM="false"

python train_cacldm.py --cfg configs/cacldm.yaml --wandb_name CACLDM