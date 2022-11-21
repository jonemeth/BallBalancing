import math
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from bblib.defs import EnvironmentConfig, Position, Angle, EnvironmentState, Observation, Action, Reward, Speed
from bblib.utils import draw_spot, init_motion_kalman


class Environment(ABC):
    def __init__(self, config: EnvironmentConfig, init_env_state: EnvironmentState):
        self.config = config
        self.state = init_env_state
        self.actions: List[Action] = []
        self.kalman: Optional[KalmanFilter] = None

    def observe(self) -> Observation:
        """
        observed_pos = self.observe_position()

        if not self.kalman:
            self.kalman = init_motion_kalman(observed_pos, self.config.d_t)
        else:
            self.kalman.predict()
            self.kalman.update([observed_pos.x, observed_pos.y])

        estimated_pos = Position(float(self.kalman.x[0]), float(self.kalman.x[1]))
        estimated_speed = Speed(float(self.kalman.x[2]), float(self.kalman.x[3]))
        """
        a, b, x, y, sx, sy = self.sensor.get()
        observed_pos = Position(float(a), float(b))
        estimated_pos = Position(float(x), float(y))
        estimated_speed = Speed(float(sx), float(sy))
        
        real_pos = self.observe_real_position()

        angle = self.observe_angle()
        action = self.actions[-1] if 1 <= len(self.actions) else None

        return Observation(estimated_pos, estimated_speed, angle, action, observed_pos,
                           real_pos, self._compute_reward(estimated_pos, estimated_speed), False)

    def _compute_reward(self, estimated_position: Position, estimated_speed: Speed) -> Reward:
        reward = 0.0

        dx = estimated_position.x / self.config.limits.max_x
        dy = estimated_position.y / self.config.limits.max_y
        # distance = math.sqrt(dx ** 2 + dy ** 2)
        # reward += math.sqrt(2.0) - distance
        # distance = (dx ** 2 + dy ** 2)

        if abs(dx) > 0.5 or abs(dy) > 0.5:
            reward -= 10.0  # 4.0* math.sqrt(2.0)
        else:
            sx = estimated_speed.x / self.config.limits.max_x
            sy = estimated_speed.y / self.config.limits.max_y
            speed = math.sqrt(sx**2 + sy**2)
            reward = 1.0 - 10.0*speed
            
            #distance = max(abs(dx), abs(dy))
            #reward += 1.0 - distance**2.0
        
        # OLD REWARD:
        # if abs(observed_pos.x) > 0.5*self.config.limits.max_x or \
        #         abs(observed_pos.y) > 0.5*self.config.limits.max_y:
        #     reward -= 10 #2.0  # 4.0* math.sqrt(2.0)
        # else:
        #     speed = math.sqrt(estimated_speed.x ** 2 + estimated_speed.y ** 2)
        #     reward += 1.0 - 10.0 * speed

        # if 2 <= len(self.actions):
        #     action1 = self.actions[-2]
        #     action2 = self.actions[-1]
        #     rot_diff_dist = (action1.x - action2.x) ** 2 + (action1.y - action2.y) ** 2
        #     reward -= rot_diff_dist / 20.0
        #
        # if 1 <= len(self.actions):
        #     action = self.actions[-1]
        #     rot_dist = action.x**2 + action.y**2
        #     reward -= rot_dist / 20.0

        return reward

    def render(self, observation: Observation) -> np.ndarray:
        height = 200
        width = round(height * (self.config.limits.max_y / self.config.limits.max_x))
        size = (height, width, 3)
        angle_scale = math.sin(max(self.config.max_angle.x, self.config.max_angle.y))

        img = np.zeros(size, dtype=np.uint8)

        draw_spot(img, 0.5, 0.5, (64, 64, 64))

        x = (observation.estimated_pos.x + self.config.limits.max_x) / (2.0 * self.config.limits.max_x)
        y = (observation.estimated_pos.y + self.config.limits.max_y) / (2.0 * self.config.limits.max_y)
        if observation.reward == -10.0:
            draw_spot(img, x, y, (0, 0, 255))
        else:
            draw_spot(img, x, y, (255, 255, 255))
        

        x = 0.5 + math.sin(observation.angle.x) / (2.0*angle_scale)
        y = 0.5 + math.sin(observation.angle.y) / (2.0*angle_scale)
        draw_spot(img, x, y, (255, 0, 255))

        sx = (observation.estimated_pos.x + observation.estimated_speed.x + self.config.limits.max_x) / \
             (2.0 * self.config.limits.max_x)
        sy = (observation.estimated_pos.y + observation.estimated_speed.y + self.config.limits.max_y) / \
             (2.0 * self.config.limits.max_y)
        draw_spot(img, sx, sy, (255, 0, 0))

        x = (observation.observed_pos.x + self.config.limits.max_x) / (2.0 * self.config.limits.max_x)
        y = (observation.observed_pos.y + self.config.limits.max_y) / (2.0 * self.config.limits.max_y)
        draw_spot(img, x, y, (0, 255, 0))

        if observation.real_pos is not None:
            x = (observation.real_pos.x + self.config.limits.max_x) / (2.0 * self.config.limits.max_x)
            y = (observation.real_pos.y + self.config.limits.max_y) / (2.0 * self.config.limits.max_y)
            draw_spot(img, x, y, (0, 0, 255))

        return img

    def get_state(self) -> EnvironmentState:
        return self.state

    def get_config(self):
        return self.config

    @abstractmethod
    def update(self, action: Action) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def observe_position(self) -> Position:
        raise NotImplementedError

    def observe_real_position(self) -> Optional[Position]:
        raise NotImplementedError

    @abstractmethod
    def observe_angle(self) -> Angle:
        raise NotImplementedError


class EnvironmentFactory(ABC):
    def __init__(self, env_config: EnvironmentConfig):
        self.env_config = env_config

    def get_env_config(self) -> EnvironmentConfig:
        return self.env_config

    @abstractmethod
    def create(self) -> Environment:
        raise NotImplementedError
