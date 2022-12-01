from dataclasses import dataclass
from typing import List, Optional


SAVES_ROOT = "saves"
GRAVITY = 9.81  # mm/s^2
Reward = float


@dataclass
class Position:
    x: float
    y: float


@dataclass
class Speed:
    x: float
    y: float


@dataclass
class Rotation:
    x: int
    y: int


@dataclass
class Action:
    x: int
    y: int


@dataclass
class Angle:
    x: float
    y: float


@dataclass
class Limits:
    """
    In meters
    """
    max_x: float
    max_y: float


@dataclass
class EnvironmentConfig:
    d_t: float
    limits: Limits
    max_rotation: Rotation
    max_angle: Angle

    def get_episode_steps(self, episode_secs: float):
        return 1 + round(episode_secs / self.d_t)


@dataclass
class EnvironmentState:
    rot: Rotation


@dataclass
class VirtualBall:
    pos: Position
    speed: Speed


@dataclass
class Observation:
    estimated_pos: Position
    estimated_speed: Speed
    angle: Angle
    last_action: Action
    observed_pos: Position
    real_pos: Optional[Position]
    reward: Reward
    done: bool


@dataclass
class RandomVirtualEnvironmentConfig:
    max_gravity_offset: float  # m/s^2
    max_angle_offset: float  # ratio of max_angle
    max_angle_scale: float  # max distance from 1.0


@dataclass
class VirtualEnvironmentConfig:
    gravity: float
    angle_offset: Angle
    angle_scale: Angle


@dataclass
class VirtualEnvironmentNoiseConfig:
    position_std: float  # meters


Episode = List[Observation]
