from copy import deepcopy

class RollingPinDefaultConfig:
    def __init__(self) -> None:
        self.urdf_path = "{description}/rolling_pin.urdf"

    @property
    def controllers(self):
        pass
        