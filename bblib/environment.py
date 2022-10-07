import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from bblib.defs import EnvironmentConfig, Position, Rotation, Angle, EnvironmentState, VirtualBall, Speed, \
    Observation, VirtualEnvironmentConfig, VirtualEnvironmentNoiseConfig, Action, Reward
from bblib.utils import compute_angle


class Environment(ABC):
    def __init__(self, config: EnvironmentConfig, init_env_state: EnvironmentState):
        self.config = config
        self.state = init_env_state
        self.last_action = Action(0, 0)
        self.kalman: Optional[KalmanFilter] = None

    @abstractmethod
    def update(self, action: Action) -> Observation:
        raise NotImplementedError

    def init_kalman(self, pos: Position):
        f = KalmanFilter(dim_x=4, dim_z=2)
        f.x = np.array([pos.x, pos.y, 0.0, 0.0])  # velocity

        dt = self.config.d_t

        # Transition matrix
        f.F = np.array([[1.0, 0.0, dt, 0.0],
                        [0.0, 1.0, 0.0, dt],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

        # Measurement matrix
        f.H = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0]])

        # Process noise covariance
        magnitude_of_acceleration_noise = 0.1
        f.Q = np.array([[0.25 * dt ** 4, 0, 0.5 * dt ** 3, 0],
                        [0, 0.25 * dt ** 4, 0, 0.5 * dt ** 3],
                        [0.5 * dt ** 3, 0, dt ** 2, 0],
                        [0, 0.5 * dt ** 3, 0, dt ** 2]]) * (magnitude_of_acceleration_noise**2)

        # Measurement noise covariance
        f.R = np.array([[0.01**2, 0.0],
                        [0.0, 0.01**2]])

        # Covariance
        f.P = (0.1**2) * np.eye(4)

        self.kalman = f

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

    def observe(self) -> Observation:
        observed_pos = self.observe_position()

        if not self.kalman:
            self.init_kalman(observed_pos)
        else:
            self.kalman.predict()
            self.kalman.update([observed_pos.x, observed_pos.y])

        estimated_pos = Position(float(self.kalman.x[0]), float(self.kalman.x[1]))
        estimated_speed = Speed(float(self.kalman.x[2]), float(self.kalman.x[3]))

        angle = self.observe_angle()

        return Observation(estimated_pos, estimated_speed, angle, self.last_action, observed_pos,
                           self.get_real_position(), self.compute_reward(observed_pos), False)

    @abstractmethod
    def observe_position(self) -> Position:
        raise NotImplementedError

    @abstractmethod
    def observe_angle(self) -> Angle:
        raise NotImplementedError

    def get_real_position(self) -> Optional[Position]:
        return None

    def get_state(self) -> EnvironmentState:
        return self.state

    def get_config(self):
        return self.config


class VirtualEnvironment(Environment):
    def __init__(self, config: EnvironmentConfig,
                 init_env_state: EnvironmentState,
                 virtual_config: VirtualEnvironmentConfig,
                 ball: VirtualBall,
                 noise_cfg: VirtualEnvironmentNoiseConfig):
        super().__init__(config, init_env_state)

        self.virtual_config = virtual_config
        self.ball = ball
        self.noise_cfg = noise_cfg

    def observe_position(self) -> Position:
        noise = Position(np.random.normal(scale=self.noise_cfg.position),
                         np.random.normal(scale=self.noise_cfg.position))
        return Position(self.ball.pos.x + noise.x, self.ball.pos.y + noise.y)

    def observe_angle(self) -> Angle:
        return compute_angle(self.config, self.state)

    def get_real_position(self) -> Optional[Position]:
        return self.ball.pos

    def update(self, action: Action) -> Observation:
        self.state.rot.x = max(-self.config.max_rotation.x,
                               min(self.config.max_rotation.x, self.state.rot.x + action.x))
        self.state.rot.y = max(-self.config.max_rotation.y,
                               min(self.config.max_rotation.y, self.state.rot.y + action.y))

        self.last_action = action

        angle = compute_angle(self.config, self.state)
        noisy_angle_x = angle.x * self.virtual_config.angle_scale.x + self.virtual_config.angle_offset.x
        noisy_angle_y = angle.y * self.virtual_config.angle_scale.y + self.virtual_config.angle_offset.y

        self.ball.speed = Speed(
            self.ball.speed.x + self.config.d_t * self.virtual_config.gravity * math.sin(noisy_angle_x),
            self.ball.speed.y + self.config.d_t * self.virtual_config.gravity * math.sin(noisy_angle_y)
        )

        self.ball.pos = Position(
            self.ball.pos.x + self.config.d_t * self.ball.speed.x,
            self.ball.pos.y + self.config.d_t * self.ball.speed.y
        )

        # Check limits
        if self.ball.pos.x > self.config.limits.max_x:
            self.ball.pos.x = self.config.limits.max_x
            self.ball.speed.x *= -0.5
        if self.ball.pos.x < -self.config.limits.max_x:
            self.ball.pos.x = -self.config.limits.max_x
            self.ball.speed.x *= -0.5
        if self.ball.pos.y > self.config.limits.max_y:
            self.ball.pos.y = self.config.limits.max_y
            self.ball.speed.y *= -0.5
        if self.ball.pos.y < -self.config.limits.max_y:
            self.ball.pos.y = -self.config.limits.max_y
            self.ball.speed.y *= -0.5

        return self.observe()




class RealEnvironment(Environment):
    def __init__(self, config: EnvironmentConfig, init_env_state: EnvironmentState):
        super().__init__(config, init_env_state)
        pass

    def observe_position(self) -> Position:
        pass

    def observe_angle(self) -> Angle:
        return compute_angle(self.config, self.state)

    def update(self, rotation: Rotation):
        pass
