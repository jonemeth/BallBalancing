import math
import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from bblib.defs import EnvironmentConfig, EnvironmentState, Angle, Rotation, VirtualBall, Position, Speed, Observation, \
    VirtualEnvironmentConfigRandomness, GRAVITY, VirtualEnvironmentConfig


def compute_angle(config: EnvironmentConfig, state: EnvironmentState) -> Angle:
    return Angle(
        (float(state.rot.x) / config.max_rotation.x) * config.max_angle.x,
        (float(state.rot.y) / config.max_rotation.y) * config.max_angle.y
    )


def random_environment_state() -> EnvironmentState:
    rot = Rotation(random.randint(0, 9) - 4, random.randint(0, 9) - 4)
    return EnvironmentState(rot)


def random_virtual_environment_config(config: EnvironmentConfig, virtual_config: VirtualEnvironmentConfigRandomness)\
        -> VirtualEnvironmentConfig:
    gravity = GRAVITY + random.uniform(-virtual_config.gravity, virtual_config.gravity)

    angle_offset = Angle(random.uniform(-virtual_config.max_angle_offset, virtual_config.max_angle_offset)
                         * config.max_angle.x,
                         random.uniform(-virtual_config.max_angle_offset, virtual_config.max_angle_offset)
                         * config.max_angle.y)

    angle_scale = Angle(1.0+random.uniform(-virtual_config.max_angle_scale, virtual_config.max_angle_scale),
                        1.0+random.uniform(-virtual_config.max_angle_scale, virtual_config.max_angle_scale))

    return VirtualEnvironmentConfig(gravity, angle_offset, angle_scale)


def random_virtual_ball(config: EnvironmentConfig) -> VirtualBall:
    # direction = random.random() * 2 * 3.141592653
    # ball_x = 10000.0 * math.cos(direction)
    # ball_y = 10000.0 * math.sin(direction)
    # ball_x = max(-config.limits.max_x, min(config.limits.max_x, ball_x))
    # ball_y = max(-config.limits.max_y, min(config.limits.max_y, ball_y))
    ball_x = random.uniform(-config.limits.max_x, config.limits.max_x)
    ball_y = random.uniform(-config.limits.max_y, config.limits.max_y)

    speed_x = 0.0
    speed_y = 0.0

    return VirtualBall(Position(ball_x, ball_y), Speed(speed_x, speed_y))


def draw_state(config: EnvironmentConfig, state: EnvironmentState, observation: Observation,
               virtual_config: Optional[VirtualEnvironmentConfig]) -> Image.Image:
    height = 200
    width = round(height * (config.limits.max_y/config.limits.max_x))
    size = (height, width, 3)
    angle_scale = math.sin(max(config.max_angle.x, config.max_angle.y))

    img = np.zeros(size, dtype=np.uint8)

    def draw_spot(u: float, v: float, color: Tuple[int, int, int], spot_radius: Optional[int] = 2):
        """ u and v are between 0.0..+1.0 """
        u = round(size[0]*u)
        v = round(size[1]*v)
        u = int(max(spot_radius, min(size[0] - 1 - spot_radius, u)))
        v = int(max(spot_radius, min(size[1] - 1 - spot_radius, v)))
        img[u - spot_radius:u + spot_radius, v - spot_radius:v + spot_radius, :] = color

    draw_spot(0.5, 0.5, (64, 64, 64))

    x = (observation.estimated_pos.x + config.limits.max_x) / (2.0 * config.limits.max_x)
    y = (observation.estimated_pos.y + config.limits.max_y) / (2.0 * config.limits.max_y)
    draw_spot(x, y, (255, 255, 255))

    angle = compute_angle(config, state)
    x = 0.5 + math.sin(angle.x) / angle_scale
    y = 0.5 + math.sin(angle.y) / angle_scale
    draw_spot(x, y, (255, 0, 255))

    sx = (observation.estimated_pos.x + observation.estimated_speed.x + config.limits.max_x) / (2.0*config.limits.max_x)
    sy = (observation.estimated_pos.y + observation.estimated_speed.y + config.limits.max_y) / (2.0*config.limits.max_y)
    draw_spot(sx, sy, (255, 0, 0))

    # x = (observation.observed_pos.x + config.limits.max_x) / (2.0 * config.limits.max_x)
    # y = (observation.observed_pos.y + config.limits.max_y) / (2.0 * config.limits.max_y)
    # draw_spot(x, y, (0, 255, 0))

    x = (observation.real_pos.x + config.limits.max_x) / (2.0 * config.limits.max_x)
    y = (observation.real_pos.y + config.limits.max_y) / (2.0 * config.limits.max_y)
    draw_spot(x, y, (0, 0, 255))

    if virtual_config is not None:
        x = 0.5 + math.sin(virtual_config.angle_offset.x) / angle_scale
        y = 0.5 + math.sin(virtual_config.angle_offset.y) / angle_scale
        draw_spot(x, y, (255, 255, 0))

    return Image.fromarray(img)
