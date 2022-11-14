from abc import ABC, abstractmethod


class EpsilonScheduler(ABC):
    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def get_epsilon(self) -> float:
        raise NotImplementedError


class LinearEpsilonScheduler(EpsilonScheduler):
    def __init__(self, start_eps: float, end_eps: float, num_steps: int):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.num_steps = num_steps
        assert self.num_steps > 0
        self.step_count = 0

    def update(self):
        self.step_count += 1

    def get_epsilon(self) -> float:
        assert self.num_steps > 0
        return self.start_eps - (self.start_eps-self.end_eps) * min(1.0, self.step_count/self.num_steps)


class ExponentialEpsilonScheduler(EpsilonScheduler):
    def __init__(self, start_eps: float, power: float):
        self.start_eps = start_eps
        self.power = power
        self.start_eps = start_eps
        self.step_count = 0

    def update(self):
        self.step_count += 1

    def get_epsilon(self) -> float:
        return self.start_eps * (self.power ** self.step_count)
