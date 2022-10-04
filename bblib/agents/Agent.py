from abc import ABC, abstractmethod
from typing import List

from bblib.defs import Observation, Action


class Agent(ABC):
    def __init__(self, action_counts: List[int]):
        self.action_counts = action_counts

    def get_action_counts(self) -> List[int]:
        return self.action_counts

    # @abstractmethod
    # def add_experiences(self, experiences: ExperienceList):
    #     pass

    @abstractmethod
    def get_action(self, observation: Observation) -> Action:
        pass
    # @abstractmethod
    # def start(self) -> Action:
    #     pass
    #
    # @abstractmethod
    # def finish(self) -> Action:
    #     pass
