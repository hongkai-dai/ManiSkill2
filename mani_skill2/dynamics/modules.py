"""Pytorch lightning training modules."""
from typing import Dict

import pytorch_lightning as pl
import torch

from mani_skill2.dynamics.network_dynamics_model import NetworkDynamicsModel


class DynamicsPLModule(pl.LightningModule):
    """LightningModule for training network dynamics models."""

    def __init__(self, dynamics_model: NetworkDynamicsModel, lr: float):
        """
        Args:
            dynamics_model: The dynamics model.
            lr: The learning rate.
        """
        super().__init__()
        self.dynamics_model = dynamics_model
        self.lr = lr

    def _compute_loss(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # Dims to sum over.
        sum_dims = list(range(1, y.ndim))

        diff = y_pred - y
        l2_loss = torch.square(diff).sum(dim=sum_dims).mean(dim=0)
        l1_loss = torch.abs(diff).sum(dim=sum_dims).mean(dim=0)
        loss = l2_loss * 0.5 + l1_loss * 0.5
        return loss

    def _step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        obs, action, next_obs = batch["obs"], batch["actions"], batch["new_obs"]
        pred_next_obs, _ = self.dynamics_model.step(obs, action)
        loss = self._compute_loss(next_obs, pred_next_obs)
        return loss

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self._step(batch, batch_idx)
        self.log(
            "train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self._step(batch, batch_idx)
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
