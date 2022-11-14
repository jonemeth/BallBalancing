import math
import random
from typing import Tuple, Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from bblib.defs import EnvironmentConfig, EnvironmentState, Angle, Rotation, VirtualBall, Position, Speed, \
    RandomVirtualEnvironmentConfig, GRAVITY, VirtualEnvironmentConfig


def compute_angle(config: EnvironmentConfig, state: EnvironmentState) -> Angle:
    # return Angle(
    #     (float(state.rot.x) / config.max_rotation.x) * config.max_angle.x,
    #     (float(state.rot.y) / config.max_rotation.y) * config.max_angle.y
    # )
    g = 1.2
    base_x = config.max_angle.x / (g**config.max_rotation.x)
    base_y = config.max_angle.y / (g**config.max_rotation.y)
    angle_x = base_x * (g**abs(state.rot.x))
    angle_y = base_y * (g**abs(state.rot.y))
    angle_x = -angle_x if state.rot.x < 0 else angle_x
    angle_y = -angle_y if state.rot.y < 0 else angle_y
    return Angle(angle_x, angle_y)

def random_environment_state() -> EnvironmentState:
    rot = Rotation(random.randint(0, 9) - 4, random.randint(0, 9) - 4)
    return EnvironmentState(rot)


def random_virtual_environment_config(config: EnvironmentConfig, virtual_config: RandomVirtualEnvironmentConfig)\
        -> VirtualEnvironmentConfig:
    gravity = GRAVITY + random.uniform(-virtual_config.max_gravity_offset, virtual_config.max_gravity_offset)

    angle_offset = Angle(random.uniform(-virtual_config.max_angle_offset, virtual_config.max_angle_offset)
                         * config.max_angle.x,
                         random.uniform(-virtual_config.max_angle_offset, virtual_config.max_angle_offset)
                         * config.max_angle.y)

    angle_scale = Angle(1.0+random.uniform(-virtual_config.max_angle_scale, virtual_config.max_angle_scale),
                        1.0+random.uniform(-virtual_config.max_angle_scale, virtual_config.max_angle_scale))

    return VirtualEnvironmentConfig(gravity, angle_offset, angle_scale)


def random_virtual_ball(config: EnvironmentConfig) -> VirtualBall:
    direction = random.random() * 2 * 3.141592653
    ball_x = 10000.0 * math.cos(direction)
    ball_y = 10000.0 * math.sin(direction)
    ball_x = max(-config.limits.max_x, min(config.limits.max_x, ball_x))
    ball_y = max(-config.limits.max_y, min(config.limits.max_y, ball_y))
    # ball_x = random.uniform(-config.limits.max_x, config.limits.max_x)
    # ball_y = random.uniform(-config.limits.max_y, config.limits.max_y)

    speed_x = 0.0
    speed_y = 0.0

    return VirtualBall(Position(ball_x, ball_y), Speed(speed_x, speed_y))


def init_motion_kalman(pos: Position, d_t: float, magnitude_of_acceleration_noise=0.1):
    f = KalmanFilter(dim_x=4, dim_z=2)
    f.x = np.array([pos.x, pos.y, 0.0, 0.0])  # velocity

    # Transition matrix
    f.F = np.array([[1.0, 0.0, d_t, 0.0],
                    [0.0, 1.0, 0.0, d_t],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])

    # Measurement matrix
    f.H = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0]])

    # Process noise covariance
    f.Q = np.array([[0.25 * d_t ** 4, 0, 0.5 * d_t ** 3, 0],
                    [0, 0.25 * d_t ** 4, 0, 0.5 * d_t ** 3],
                    [0.5 * d_t ** 3, 0, d_t ** 2, 0],
                    [0, 0.5 * d_t ** 3, 0, d_t ** 2]]) * (magnitude_of_acceleration_noise**2)

    # Measurement noise covariance
    f.R = np.array([[0.01**2, 0.0],
                    [0.0, 0.01**2]])

    # Covariance
    f.P = (0.1**2) * np.eye(4)

    return f


def draw_spot(img: np.ndarray, u: float, v: float, color: Tuple[int, int, int], spot_radius: Optional[int] = 2):
    """ u and v are between 0.0..+1.0 """
    size = img.shape
    u = round(size[0] * (1.0-u))
    v = round(size[1] * v)
    u = int(max(spot_radius, min(size[0] - 1 - spot_radius, u)))
    v = int(max(spot_radius, min(size[1] - 1 - spot_radius, v)))
    img[u - spot_radius:u + spot_radius, v - spot_radius:v + spot_radius, :] = color
