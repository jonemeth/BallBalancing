import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from bblib.defs import EnvironmentConfig, Position, Angle, EnvironmentState, VirtualBall, Speed, \
    Observation, VirtualEnvironmentConfig, VirtualEnvironmentNoiseConfig, Action, Reward
from bblib.utils import compute_angle, init_motion_kalman


class Environment(ABC):
    def __init__(self, config: EnvironmentConfig, init_env_state: EnvironmentState):
        self.config = config
        self.state = init_env_state

    def compute_reward(self, observed_pos: Position) -> Reward:
        reward = 0.0

        dx = observed_pos.x / self.config.limits.max_x
        dy = observed_pos.y / self.config.limits.max_y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        reward += math.sqrt(2.0) - distance

        if abs(observed_pos.x) > 0.8*self.config.limits.max_x or \
                abs(observed_pos.y) > 0.8*self.config.limits.max_y:
            reward -= 10.0

        # rot_diff_dist = ((previous_observation.last_action.x - observation.last_action.x) ** 2 +
        #                  (previous_observation.last_action.y - observation.last_action.y) ** 2)
        # reward -= rot_diff_dist / 10.0
        #
        # rot_dist = abs(observation.last_action.x) + abs(observation.last_action.y)
        # reward -= rot_dist / 10.0

        return reward

    @abstractmethod
    def update(self, action: Action) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def observe_position(self) -> Position:
        raise NotImplementedError

    @abstractmethod
    def observe_angle(self) -> Angle:
        raise NotImplementedError

    def get_state(self) -> EnvironmentState:
        return self.state

    def get_config(self):
        return self.config
