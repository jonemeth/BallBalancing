import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from filterpy.kalman import KalmanFilter

from bblib.defs import EnvironmentConfig, Position, Angle, EnvironmentState, VirtualBall, Speed, \
    Observation, VirtualEnvironmentConfig, VirtualEnvironmentNoiseConfig, Action, RandomVirtualEnvironmentConfig
from bblib.environments.Environment import Environment, EnvironmentFactory
from bblib.utils import compute_angle, init_motion_kalman, random_environment_state, \
    random_virtual_environment_config, random_virtual_ball


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
        noise = Position(np.random.normal(scale=self.noise_cfg.position_std),
                         np.random.normal(scale=self.noise_cfg.position_std))
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

    def render(self, observation: Observation) -> Image.Image:
        height = 200
        width = round(height * (self.config.limits.max_y/self.config.limits.max_x))
        size = (height, width, 3)
        angle_scale = math.sin(max(self.config.max_angle.x, self.config.max_angle.y))

        img = np.zeros(size, dtype=np.uint8)

        def draw_spot(u: float, v: float, color: Tuple[int, int, int], spot_radius: Optional[int] = 2):
            """ u and v are between 0.0..+1.0 """
            u = round(size[0]*u)
            v = round(size[1]*v)
            u = int(max(spot_radius, min(size[0] - 1 - spot_radius, u)))
            v = int(max(spot_radius, min(size[1] - 1 - spot_radius, v)))
            img[u - spot_radius:u + spot_radius, v - spot_radius:v + spot_radius, :] = color

        draw_spot(0.5, 0.5, (64, 64, 64))

        x = (observation.estimated_pos.x + self.config.limits.max_x) / (2.0 * self.config.limits.max_x)
        y = (observation.estimated_pos.y + self.config.limits.max_y) / (2.0 * self.config.limits.max_y)
        draw_spot(x, y, (255, 255, 255))

        x = 0.5 + math.sin(observation.angle.x) / angle_scale
        y = 0.5 + math.sin(observation.angle.y) / angle_scale
        draw_spot(x, y, (255, 0, 255))

        sx = (observation.estimated_pos.x + observation.estimated_speed.x + self.config.limits.max_x) / \
             (2.0*self.config.limits.max_x)
        sy = (observation.estimated_pos.y + observation.estimated_speed.y + self.config.limits.max_y) / \
             (2.0*self.config.limits.max_y)
        draw_spot(sx, sy, (255, 0, 0))

        # x = (observation.observed_pos.x + config.limits.max_x) / (2.0 * config.limits.max_x)
        # y = (observation.observed_pos.y + config.limits.max_y) / (2.0 * config.limits.max_y)
        # draw_spot(x, y, (0, 255, 0))

        x = (observation.real_pos.x + self.config.limits.max_x) / (2.0 * self.config.limits.max_x)
        y = (observation.real_pos.y + self.config.limits.max_y) / (2.0 * self.config.limits.max_y)
        draw_spot(x, y, (0, 0, 255))

        x = 0.5 + math.sin(self.virtual_config.angle_offset.x) / angle_scale
        y = 0.5 + math.sin(self.virtual_config.angle_offset.y) / angle_scale
        draw_spot(x, y, (255, 255, 0))

        return Image.fromarray(img)


class RandomVirtualEnvironmentFactory(EnvironmentFactory):
    def __init__(self,
                 env_config: EnvironmentConfig,
                 random_virtual_env_config: RandomVirtualEnvironmentConfig,
                 virtual_env_noise_cfg: VirtualEnvironmentNoiseConfig):
        super().__init__(env_config)
        self.random_virtual_env_config = random_virtual_env_config
        self.virtual_env_noise_cfg = virtual_env_noise_cfg

    def create(self) -> Environment:
        return VirtualEnvironment(self.env_config,
                                  random_environment_state(),
                                  random_virtual_environment_config(self.env_config, self.random_virtual_env_config),
                                  random_virtual_ball(self.env_config),
                                  self.virtual_env_noise_cfg)
