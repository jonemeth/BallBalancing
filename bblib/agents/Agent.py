from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from bblib.defs import Observation, Action, EnvironmentConfig


class Agent(ABC):
    def __init__(self, action_counts: List[int]):
        self.action_counts = action_counts

    def get_action_counts(self) -> List[int]:
        return self.action_counts

    @abstractmethod
    def set_env_config(self, env_config: EnvironmentConfig, train_episode_secs: float):
        raise NotImplementedError

    @abstractmethod
    def step(self, observation: Observation) -> Action:
        raise NotImplementedError

    @abstractmethod
    def start_episode(self, is_train: bool):
        raise NotImplementedError

    @abstractmethod
    def finish_episode(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, filename: Path):
        raise NotImplementedError

    @abstractmethod
    def load(self, filename: Path):
        raise NotImplementedError
