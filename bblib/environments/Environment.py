import math
from abc import ABC, abstractmethod
from typing import List

from PIL import Image

from bblib.defs import EnvironmentConfig, Position, Angle, EnvironmentState, Observation, Action, Reward


class Environment(ABC):
    def __init__(self, config: EnvironmentConfig, init_env_state: EnvironmentState):
        self.config = config
        self.state = init_env_state
        self.actions: List[Action] = []

    def _compute_reward(self, observed_pos: Position) -> Reward:
        reward = 0.0

        dx = observed_pos.x / self.config.limits.max_x
        dy = observed_pos.y / self.config.limits.max_y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        reward += math.sqrt(2.0) - distance

        if abs(observed_pos.x) > 0.8*self.config.limits.max_x or \
                abs(observed_pos.y) > 0.8*self.config.limits.max_y:
            reward -= 5.0 * math.sqrt(2.0)

        if 2 <= len(self.actions):
            action1 = self.actions[-2]
            action2 = self.actions[-1]
            rot_diff_dist = (action1.x - action2.x) ** 2 + (action1.y - action2.y) ** 2
            reward -= rot_diff_dist / 20.0
        #
        # if 1 <= len(self.actions):
        #     action = self.actions[-1]
        #     rot_dist = action.x**2 + action.y**2
        #     reward -= rot_dist / 20.0

        return reward

    def get_state(self) -> EnvironmentState:
        return self.state

    def get_config(self):
        return self.config

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

    @abstractmethod
    def render(self, observation: Observation) -> Image.Image:
        raise NotImplementedError


class EnvironmentFactory(ABC):
    def __init__(self, env_config: EnvironmentConfig):
        self.env_config = env_config

    def get_env_config(self):
        return self.env_config

    @abstractmethod
    def create(self) -> Environment:
        raise NotImplementedError
