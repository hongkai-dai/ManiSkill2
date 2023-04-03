import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
import wandb


def get_trainer(cfg, logger):
    # Hydra runs in the output directory.
    dirpath = os.getcwd()
    checkpoint_dir = os.path.join(dirpath, "checkpoints")
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(1),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="checkpoint",
            save_top_k=2,
            save_last=True,
            monitor="val/loss",
            mode="min",
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=500,
            mode="min",
        ),
    ]
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    return trainer


def initialize_wandb_and_logger(cfg):
    dirpath = os.getcwd()
    wandb.init(
        dir=dirpath,
        config=OmegaConf.to_container(cfg, resolve=True),
        **cfg.logging,
    )
    wandb.config.update(
        {
            "output_dir": dirpath,
        }
    )

    wandb_logger = WandbLogger(
        name=cfg.logging.name,
        save_dir=dirpath,
        project=cfg.logging.project,
    )
    return wandb_logger


@hydra.main(config_path="config", config_name="film_unet")
def main(cfg: DictConfig):
    logging.info(f"Starting training. Output dir: {os.getcwd()}")
    wandb_logger = initialize_wandb_and_logger(cfg)
    module = hydra.utils.instantiate(cfg.module)
    train_loader = hydra.utils.instantiate(cfg.data.train_loader)
    val_loader = hydra.utils.instantiate(cfg.data.val_loader)
    trainer = get_trainer(cfg, logger=wandb_logger)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logging.info("Training complete!")


if __name__ == "__main__":
    main()
