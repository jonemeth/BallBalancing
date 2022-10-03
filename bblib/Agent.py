import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from bblib.defs import Observation, EnvironmentConfig, Action


@dataclass
class Experience:
    observation: Observation
    action: Action
    next_observation: Observation
    reward: float
    done: bool


ExperienceList = List[Experience]


class Agent(ABC):
    def __init__(self, action_counts: List[int]):
        self.action_counts = action_counts

    def get_action_counts(self) -> List[int]:
        return self.action_counts

    @abstractmethod
    def add_experiences(self, experiences: ExperienceList):
        pass

    @abstractmethod
    def get_action(self, observation: Observation) -> Action:
        pass


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


class DefaultNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_layer_sizes: List[int], out_dims: List[int], activation_fn: Callable):
        super().__init__()

        fcs = []
        for i, n in enumerate(hidden_layer_sizes):
            fcs.append(nn.Linear(in_dim if i == 0 else hidden_layer_sizes[i - 1], n))
            fcs.append(activation_fn())
        self.fcs = nn.Sequential(*fcs)

        fc_outs = []
        for d in out_dims:
            fc_outs.append(nn.Linear(hidden_layer_sizes[-1], d))
        self.fc_outs = nn.ModuleList(fc_outs)

    def forward(self, x: torch.Tensor):
        assert 2 == len(x.shape)
        x = self.fcs(x)

        return [out(x) for out in self.fc_outs]


class DQN(Agent):
    def __init__(self, env_config: EnvironmentConfig, action_counts: List[int], mem_size: int):
        super().__init__(action_counts)
        self.env_config = env_config
        self.observation_dims = 8

        self.network = DefaultNetwork(self.observation_dims, [200, 200], self.action_counts, torch.nn.SiLU)

        self.experience_dataset = ExperienceDataset(mem_size)
        self.mem_size = mem_size

        self.batch_size = 64

        self.discount_factor = 0.95
        self.epsilon = 0.0

        self.solver = torch.optim.Adam(self.network.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.001)

    def add_experiences(self, experiences: ExperienceList):
        self.experience_dataset.add_experiences([
            (self.transform_observation(exp.observation),
             self.transform_action(exp.action),
             self.transform_observation(exp.next_observation),
             torch.tensor(exp.reward),
             torch.tensor(exp.done).float())
            for exp in experiences])

    def transform_observation(self, observation: Observation):
        return torch.tensor([observation.estimated_pos.x / self.env_config.limits.max_x,
                             observation.estimated_pos.y / self.env_config.limits.max_y,
                             observation.estimated_speed.x / self.env_config.limits.max_x,
                             observation.estimated_speed.y / self.env_config.limits.max_y,
                             observation.angle.x / self.env_config.max_angle.x,
                             observation.angle.y / self.env_config.max_angle.y,
                             float(observation.last_action.x),
                             float(observation.last_action.y)])

    def get_action(self, observation: Observation) -> Action:
        if random.random() < self.epsilon:
            actions = [random.randint(0, c - 1) for c in self.action_counts]
        else:
            self.network.eval()
            observation = self.transform_observation(observation).unsqueeze(0)
            actions = self.network(observation)
            actions = [torch.argmax(action, 1).item() for action in actions]

        if 2 == len(self.action_counts):
            return Action(actions[0] - self.action_counts[0] // 2, actions[1] - self.action_counts[1] // 2)
        elif 1 == len(self.action_counts):
            assert 9 == self.action_counts[0]
            return Action(actions[0] // 3 - 1, actions[0] % 3 - 1)

        raise NotImplementedError

    @staticmethod
    def one_hot(index: torch.Tensor, num_classes: int) -> torch.Tensor:
        # noinspection PyUnresolvedReferences
        return torch.nn.functional.one_hot(index, num_classes=num_classes)

    def transform_action(self, action: Action) -> List[torch.Tensor]:
        if 2 == len(self.action_counts):
            actions = [
                action.x + self.action_counts[0] // 2,
                action.y + self.action_counts[1] // 2,
            ]
        elif 1 == len(self.action_counts):
            assert 9 == self.action_counts[0]
            actions = [3*(1+action.x) + (1+action.y)]
        else:
            raise NotImplementedError

        actions = [torch.tensor(actions[i]) for i in range(len(self.action_counts))]
        actions = [self.one_hot(action, count) for action, count in zip(actions, self.action_counts)]
        return actions

    def train(self, num_iters: int):
        self.network.train()

        # num_iters = min(num_iters, len(self.experience_dataset.experiences)//self.batch_size)
        # for ix in range(num_iters):
        #     self.solver.zero_grad()
        #
        #     experience_batch = random.sample(self.experience_dataset.experiences, self.batch_size)
        #
        #     observations = torch.stack([exp[0] for exp in experience_batch])
        #     next_observations = torch.stack([exp[2] for exp in experience_batch])
        #
        #     rewards = torch.stack([exp[3] for exp in experience_batch])
        #     dones = torch.stack([exp[4] for exp in experience_batch])
        #     actions = [torch.stack([exp[1][i] for exp in experience_batch])
        #                for i in range(len(self.action_counts))]
        #
        #     max_next_qs = [q.detach().max(dim=1)[0] for q in self.network(next_observations)]
        #     target_qs = [(rewards + self.discount_factor * q * (1.0 - dones)) for q in max_next_qs]
        #     qs = self.network(observations)
        #     qs = [(q * a).sum(1) for q, a in zip(qs, actions)]
        #
        #     losses = [(q - target_q) ** 2 for q, target_q in zip(qs, target_qs)]
        #     loss = torch.stack(losses).mean(0).mean(0)
        #
        #     loss.backward()
        #     self.solver.step()

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

            if ix + 1 >= num_iters:
                break
