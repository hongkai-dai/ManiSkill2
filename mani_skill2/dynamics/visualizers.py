import abc
from typing import List

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import Logger
import torch


class DynamicsTrainingPLVisualizer(abc.ABC):
    """Base class for dynamics training visualizers.

    Subclasses implement different types of visualization
    that is used specifically during pytorch lightning training
    of dynamics modules.
    """

    @abc.abstractmethod
    def __call__(
        self,
        logger: Logger,
        state: torch.Tensor,
        next_state: torch.Tensor,
        pred_next_state: torch.Tensor,
        split: str,
        global_step: int,
    ) -> None:
        """Generates a visualization and logs it.

        Args:
            logger: The logger to visualize to.
            state: Ground truth current state.
            next_state: Ground truth next state.
            pred_next_state: Predicted next state.
            split: The training / validation split.
            global_step: Step in training to log at.
        """


class HeightMapDynamicsPLVisualizer(DynamicsTrainingPLVisualizer):
    """Visualizes images of height map transitions."""

    def __init__(self, max_num_to_visualize: int = 4):
        """
        Args:
            max_num_to_visualize: Max number of transitions to visualize.
        """
        self.max_num_to_visualize = max_num_to_visualize

    def _format_height_map(
        self,
        height_map: torch.Tensor,
        max_height: float,
    ) -> torch.Tensor:
        x = height_map.clone()
        x[x < 0] = 0
        x = x / max_height * 255
        x = x.to(torch.uint8)
        return x

    def _format_transition(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        pred_next_state: torch.Tensor,
    ) -> List:
        max_height = state.max() + 1e-12
        state = self._format_height_map(state, max_height)
        next_state = self._format_height_map(next_state, max_height)
        pred_next_state = self._format_height_map(pred_next_state, max_height)
        return [state, next_state, pred_next_state]

    def __call__(
        self,
        logger: Logger,
        state: torch.Tensor,
        next_state: torch.Tensor,
        pred_next_state: torch.Tensor,
        split: str,
        global_step: int,
    ) -> None:
        images = []
        for i in range(self.max_num_to_visualize):
            images += self._format_transition(
                state[i],
                next_state[i],
                pred_next_state[i],
            )
        images = torch.stack(images)[:, None]

        key = f"{split}/images"
        if isinstance(logger, WandbLogger):
            logger.log_image(
                key=key,
                images=list(images.to(torch.float32)),
                step=global_step,
            )
        else:
            logger.experiment.add_images(
                key,
                images,
                global_step=global_step,
                dataformats="NCHW",
            )
