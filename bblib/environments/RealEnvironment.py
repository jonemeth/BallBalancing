from PIL import Image

from bblib.defs import EnvironmentConfig, Position, Angle, EnvironmentState, Observation, Action
from bblib.environments.Environment import Environment


class RealEnvironment(Environment):
    def __init__(self, config: EnvironmentConfig, init_env_state: EnvironmentState):
        super().__init__(config, init_env_state)
        raise NotImplementedError

    def observe_position(self) -> Position:
        raise NotImplementedError

    def observe_angle(self) -> Angle:
        raise NotImplementedError

    def update(self, action: Action) -> Observation:
        raise NotImplementedError

    def observe(self) -> Observation:
        raise NotImplementedError

    def render(self, observation: Observation) -> Image.Image:
        raise NotImplementedError
