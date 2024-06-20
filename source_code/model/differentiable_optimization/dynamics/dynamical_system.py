# ------------------------------------------------------------------------------
# Copyright:    Technical University Dortmund
# Project:      KISSaF
# Created by:   Institute of Control Theory and System Engineering
# ------------------------------------------------------------------------------

import torch.nn as nn
from abc import abstractmethod
import torch

class DynSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.car_lengths = torch.empty((1, 1))
        self.car_widths = torch.empty((1, 1))

    @abstractmethod
    def state_transition(self, control, current_state, dt):
        pass

    def set_car_geometries(self, car_lengths, car_widths, num_of_modes):
        self.car_lengths = car_lengths.repeat_interleave(num_of_modes, dim=0)
        self.car_widths = car_widths.repeat_interleave(num_of_modes, dim=0)
