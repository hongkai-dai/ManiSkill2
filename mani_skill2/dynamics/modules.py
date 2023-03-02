"""Pytorch lightning training modules."""
from typing import Dict, Optional

import pytorch_lightning as pl
import torch

from mani_skill2.dynamics.network_dynamics_model import NetworkDynamicsModel
from mani_skill2.dynamics.visualizers import DynamicsTrainingPLVisualizer


class DynamicsPLModule(pl.LightningModule):
    """LightningModule for training network dynamics models."""

    def __init__(
        self,
        dynamics_model: NetworkDynamicsModel,
        lr: float,
        visualizer: Optional[DynamicsTrainingPLVisualizer] = None,
    ):
        """
        Args:
            dynamics_model: The dynamics model.
            lr: The learning rate.
            visualizer: Visualizes transitions and logs them.
        """
        super().__init__()
        self.dynamics_model = dynamics_model
        self.lr = lr
        self.visualizer = visualizer

    def _compute_loss(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # Dims to sum over.
        sum_dims = list(range(1, y.ndim))

        diff = y_pred - y
        l2_loss = torch.square(diff).sum(dim=sum_dims).mean(dim=0)
        l1_loss = torch.abs(diff).sum(dim=sum_dims).mean(dim=0)
        loss = l2_loss * 0.5 + l1_loss * 0.5
        return loss

    def _step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        split: str,
    ) -> torch.Tensor:
        state, action, next_state = batch["obs"], batch["actions"], batch["new_obs"]
        pred_next_state, _, _ = self.dynamics_model.step(state, action)
        loss = self._compute_loss(next_state, pred_next_state)
        if self.visualizer is not None:
            self.visualizer(
                self.logger,
                state,
                next_state,
                pred_next_state,
                split=split,
                global_step=self.global_step,
            )

        self.log(
            f"{split}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self._step(batch, batch_idx, split="train")
        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self._step(batch, batch_idx, split="val")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
