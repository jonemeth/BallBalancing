import random
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

from bblib.agents.Agent import Agent
from bblib.agents.nn import DefaultNetwork
from bblib.defs import Observation, EnvironmentConfig, Action


@dataclass
class Experience:
    observation: Observation
    action: Action
    next_observation: Observation


ExperienceList = List[Experience]


class ExperienceDataset(Dataset):
    def __init__(self, max_size: int):
        super().__init__()
        self.experiences = []
        self.max_size = max_size

    def __len__(self):
        return len(self.experiences)

    def add_experiences(self, experiences: List[tuple]):
        self.experiences.extend(experiences)
        if len(self.experiences) > self.max_size:
            self.experiences = self.experiences[-self.max_size:]

    def __getitem__(self, idx):
        return self.experiences[idx]


class DQN(Agent):
    def __init__(self, env_config: EnvironmentConfig, action_counts: List[int], mem_size: int):
        super().__init__(action_counts)
        self.env_config = env_config
        self.observation_dims = 8

        self.network = DefaultNetwork(self.observation_dims, [200, 200], self.action_counts, torch.nn.SiLU)

        self.experience_dataset = ExperienceDataset(mem_size)
        self.new_experiences = []

        self.last_observation = None
        self.last_action = None

        self.mem_size = mem_size
        self.batch_size = 64
        self.num_train_iters = 64
        self.discount_factor = 0.95
        self.epsilon = 0.0

        self.solver = torch.optim.Adam(self.network.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.001)

    def start(self):
        self.new_experiences = []
        self.last_observation = None
        self.last_action = None

    def finish(self):
        self.experience_dataset.add_experiences([
            (self._transform_observation(exp.observation),
             self._transform_action(exp.action),
             self._transform_observation(exp.next_observation),
             torch.tensor(exp.observation.reward),
             torch.tensor(exp.observation.done).float())
            for exp in self.new_experiences])
        pass

    def step(self, observation: Observation) -> Action:
        # Save experience
        if self.last_observation is not None:
            assert self.last_action is not None
            self.new_experiences.append(Experience(self.last_observation, self.last_action, observation))

        if random.random() < self.epsilon:
            actions = [random.randint(0, c - 1) for c in self.action_counts]
        else:
            self.network.eval()
            actions = self.network(self._transform_observation(observation).unsqueeze(0))
            actions = [torch.argmax(action, 1).item() for action in actions]

        if 2 == len(self.action_counts):
            action = Action(actions[0] - self.action_counts[0] // 2, actions[1] - self.action_counts[1] // 2)
        elif 1 == len(self.action_counts):
            assert 9 == self.action_counts[0]
            action = Action(actions[0] // 3 - 1, actions[0] % 3 - 1)
        else:
            raise NotImplementedError

        self.last_action = action
        self.last_observation = observation

        return action

    def train(self):
        self.network.train()

        train_dataloader = DataLoader(self.experience_dataset, batch_size=self.batch_size, shuffle=True)

        for ix, (observations, actions, next_observations, rewards, dones) in enumerate(train_dataloader):
            self.solver.zero_grad()

            # next_qs = [q.detach() for q in self.network(next_observations)]
            # max_next_q = torch.cat([values.max(axis=1, keepdims=True)[0] for values in next_qs], dim=1).mean(dim=1)
            # target_q = rewards + self.discount_factor * max_next_q*(1.0-dones)
            max_next_qs = [q.detach().max(dim=1)[0] for q in self.network(next_observations)]
            target_qs = [(rewards + self.discount_factor * q * (1.0 - dones)) for q in max_next_qs]

            qs = self.network(observations)
            qs = [(q * a).sum(1) for q, a in zip(qs, actions)]

            losses = [(q - target_q) ** 2 for q, target_q in zip(qs, target_qs)]
            loss = torch.stack(losses).mean(0).mean(0)

            loss.backward()
            self.solver.step()

            if ix + 1 >= self.num_train_iters:
                break

    def _transform_observation(self, observation: Observation):
        return torch.tensor([observation.estimated_pos.x / self.env_config.limits.max_x,
                             observation.estimated_pos.y / self.env_config.limits.max_y,
                             observation.estimated_speed.x / self.env_config.limits.max_x,
                             observation.estimated_speed.y / self.env_config.limits.max_y,
                             observation.angle.x / self.env_config.max_angle.x,
                             observation.angle.y / self.env_config.max_angle.y,
                             float(observation.last_action.x),
                             float(observation.last_action.y)])

    @staticmethod
    def _one_hot(index: torch.Tensor, num_classes: int) -> torch.Tensor:
        # noinspection PyUnresolvedReferences
        return torch.nn.functional.one_hot(index, num_classes=num_classes)

    def _transform_action(self, action: Action) -> List[torch.Tensor]:
        if 2 == len(self.action_counts):
            actions = [
                action.x + self.action_counts[0] // 2,
                action.y + self.action_counts[1] // 2,
            ]
        elif 1 == len(self.action_counts):
            assert 9 == self.action_counts[0]
            actions = [3 * (1 + action.x) + (1 + action.y)]
        else:
            raise NotImplementedError

        actions = [torch.tensor(actions[i]) for i in range(len(self.action_counts))]
        actions = [self._one_hot(action, count) for action, count in zip(actions, self.action_counts)]
        return actions
