import math
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from bblib.defs import EnvironmentConfig, Position, Angle, EnvironmentState, VirtualBall, Speed, \
    Observation, VirtualEnvironmentConfig, VirtualEnvironmentNoiseConfig, Action
from bblib.environments.Environment import Environment
from bblib.utils import compute_angle, init_motion_kalman


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
        self.kalman: Optional[KalmanFilter] = None
        self.last_action = Action(0, 0)

    def observe_position(self) -> Position:
        noise = Position(np.random.normal(scale=self.noise_cfg.position),
                         np.random.normal(scale=self.noise_cfg.position))
        return Position(self.ball.pos.x + noise.x, self.ball.pos.y + noise.y)

    def observe_angle(self) -> Angle:
        return compute_angle(self.config, self.state)

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

    def observe(self) -> Observation:
        observed_pos = self.observe_position()

        if not self.kalman:
            self.kalman = init_motion_kalman(observed_pos, self.config.d_t)
        else:
            self.kalman.predict()
            self.kalman.update([observed_pos.x, observed_pos.y])

        estimated_pos = Position(float(self.kalman.x[0]), float(self.kalman.x[1]))
        estimated_speed = Speed(float(self.kalman.x[2]), float(self.kalman.x[3]))

        angle = self.observe_angle()

        return Observation(estimated_pos, estimated_speed, angle, self.last_action, observed_pos,
                           self.ball.pos, self.compute_reward(observed_pos), False)
