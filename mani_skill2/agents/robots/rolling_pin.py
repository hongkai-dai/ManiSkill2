import numpy as np

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.configs.rolling_pin import defaults


class RollingPin(BaseAgent):
    _config: defaults.RollingPinDefaultConfig

    def get_default_config(self):
        return defaults.RollingPinDefaultConfig()