import os
import sys

import torch
import numpy as np

import wandb
import logging
import datetime
from tqdm.auto import tqdm
from omegaconf import OmegaConf

import transformers
import diffusers
from diffusers.optimization import get_scheduler

from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.caldm import CALDM
from mld.utils.utils import print_table, set_seed, move_batch_to_device


def main():
    cfg = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(cfg.SEED_VALUE)

    name_time_str = os.path.join(
        cfg.NAME, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    )
    cfg.output_dir = os.path.join(cfg.FOLDER, name_time_str)
    os.makedirs(cfg.output_dir, exist_ok=False)
    os.makedirs(f"{cfg.output_dir}/checkpoints", exist_ok=False)
    if cfg.TRAIN.model_ema:
        os.makedirs(f"{cfg.output_dir}/checkpoints_ema", exist_ok=False)

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(os.path.join(cfg.output_dir, "output.log"))
    handlers = [file_handler, stream_handler]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)

    wandb.init(
        project="CALDM",
        config=dict(cfg),
        name=cfg.wandb_name + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
        settings=wandb.Settings(code_dir="."),
    )

    OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))

    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    dataset = get_dataset(cfg)
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()

    model = CALDM(cfg, dataset)

    assert cfg.TRAIN.PRETRAINED, "cfg.TRAIN.PRETRAINED must not be None."
    logger.info(f"Loading pre-trained model: {cfg.TRAIN.PRETRAINED}")
    state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]
    logger.info(model.load_state_dict(state_dict, strict=False))

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.vae.eval()
    model.text_encoder.eval()
    model.to(device)

    logger.info("learning_rate: {}".format(cfg.TRAIN.learning_rate))
    optimizer = torch.optim.AdamW(
        model.denoiser.parameters(),
        lr=cfg.TRAIN.learning_rate,
        betas=(cfg.TRAIN.adam_beta1, cfg.TRAIN.adam_beta2),
        weight_decay=cfg.TRAIN.adam_weight_decay,
        eps=cfg.TRAIN.adam_epsilon,
    )

    if cfg.TRAIN.max_train_steps == -1:
        assert cfg.TRAIN.max_train_epochs != -1
        cfg.TRAIN.max_train_steps = cfg.TRAIN.max_train_epochs * len(train_dataloader)

    if cfg.TRAIN.checkpointing_steps == -1:
        assert cfg.TRAIN.checkpointing_epochs != -1
        cfg.TRAIN.checkpointing_steps = cfg.TRAIN.checkpointing_epochs * len(
            train_dataloader
        )

    if cfg.TRAIN.validation_steps == -1:
        assert cfg.TRAIN.validation_epochs != -1
        cfg.TRAIN.validation_steps = cfg.TRAIN.validation_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        cfg.TRAIN.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.lr_warmup_steps,
        num_training_steps=cfg.TRAIN.max_train_steps,
    )

    # EMA
    model_ema = None
    if cfg.TRAIN.model_ema:
        alpha = 1.0 - cfg.TRAIN.model_ema_decay
        logger.info(f"EMA alpha: {alpha}")
        model_ema = torch.optim.swa_utils.AveragedModel(
            model, device, lambda p0, p1, _: (1 - alpha) * p0 + alpha * p1
        )

    logger.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logging.info(f"  Num Epochs = {cfg.TRAIN.max_train_epochs}")
    logging.info(f"  Instantaneous batch size per device = {cfg.TRAIN.BATCH_SIZE}")
    logging.info(f"  Total optimization steps = {cfg.TRAIN.max_train_steps}")

    global_step = 0

    @torch.no_grad()
    def validation(target_model, ema=False):
        target_model.denoiser.eval()
        for val_batch in tqdm(val_dataloader):
            val_batch = move_batch_to_device(val_batch, device)
            target_model.allsplit_step(split="val", batch=val_batch)
        metrics = target_model.allsplit_epoch_end()

        max_val_rp1 = metrics["Metrics/R_precision_top_1"]
        min_val_fid = metrics["Metrics/FID"]
        print_table(f"Validation@Step-{global_step}", metrics)
        wandb.log(metrics, step=global_step)

        target_model.denoiser.train()
        return max_val_rp1, min_val_fid

    max_rp1, min_fid = validation(model)
    if cfg.TRAIN.model_ema:
        validation(model_ema.module, ema=True)

    progress_bar = tqdm(range(0, cfg.TRAIN.max_train_steps), desc="Steps")
    while True:
        for step, batch in enumerate(train_dataloader):
            batch = move_batch_to_device(batch, device)
            loss_dict = model.allsplit_step("train", batch)

            diff_loss = loss_dict["diff_loss"]
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.denoiser.parameters(), cfg.TRAIN.max_grad_norm
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            progress_bar.update(1)
            global_step += 1

            if cfg.TRAIN.model_ema and global_step % cfg.TRAIN.model_ema_steps == 0:
                model_ema.update_parameters(model)

            if global_step % cfg.TRAIN.checkpointing_steps == 0:
                save_path = os.path.join(
                    cfg.output_dir, "checkpoints", f"checkpoint-{global_step}.ckpt"
                )
                ckpt = dict(state_dict=model.state_dict())
                model.on_save_checkpoint(ckpt)
                torch.save(ckpt, save_path)
                logger.info(f"Saved state to {save_path}")

                if cfg.TRAIN.model_ema:
                    save_path = os.path.join(
                        cfg.output_dir,
                        "checkpoints_ema",
                        f"checkpoint-{global_step}.ckpt",
                    )
                    ckpt = dict(state_dict=model_ema.module.state_dict())
                    model_ema.module.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    logger.info(f"Saved EMA state to {save_path}")

            if global_step % cfg.TRAIN.validation_steps == 0:
                cur_rp1, cur_fid = validation(model)
                if cfg.TRAIN.model_ema:
                    validation(model_ema.module, ema=True)

                if cur_rp1 > max_rp1:
                    max_rp1 = cur_rp1
                    save_path = os.path.join(
                        cfg.output_dir,
                        "checkpoints",
                        f"checkpoint-{global_step}-rp1-{round(cur_rp1, 3)}.ckpt",
                    )
                    ckpt = dict(state_dict=model.state_dict())
                    model.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    logger.info(
                        f"Saved state to {save_path} with rp1:{round(cur_rp1, 3)}"
                    )

                if cur_fid < min_fid:
                    min_fid = cur_fid
                    save_path = os.path.join(
                        cfg.output_dir,
                        "checkpoints",
                        f"checkpoint-{global_step}-fid-{round(cur_fid, 3)}.ckpt",
                    )
                    ckpt = dict(state_dict=model.state_dict())
                    model.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    logger.info(
                        f"Saved state to {save_path} with fid:{round(cur_fid, 3)}"
                    )

            logs = {
                "loss": loss.item(),
                "diff_loss": diff_loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            wandb.log(logs, step=global_step)

            if global_step >= cfg.TRAIN.max_train_steps:
                save_path = os.path.join(
                    cfg.output_dir, "checkpoints", f"checkpoint-last.ckpt"
                )
                ckpt = dict(state_dict=model.state_dict())
                model.on_save_checkpoint(ckpt)
                torch.save(ckpt, save_path)

                if cfg.TRAIN.model_ema:
                    save_path = os.path.join(
                        cfg.output_dir, "checkpoints_ema", f"checkpoint-last.ckpt"
                    )
                    ckpt = dict(state_dict=model_ema.module.state_dict())
                    model_ema.module.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)

                exit(0)


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(5678)
    # debugpy.wait_for_client()
    main()
