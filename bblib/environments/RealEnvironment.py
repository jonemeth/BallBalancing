from bblib.defs import EnvironmentConfig, Position, Angle, EnvironmentState, Observation, Action
from bblib.environments.Environment import Environment


class RealEnvironment(Environment):
    def __init__(self, config: EnvironmentConfig, init_env_state: EnvironmentState):
        super().__init__(config, init_env_state)
        pass

    def observe_position(self) -> Position:
        pass

    def observe_angle(self) -> Angle:
        pass

    def update(self, action: Action) -> Observation:
        pass

    def observe(self) -> Observation:
        pass
