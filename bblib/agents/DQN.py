import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from bblib.agents.Agent import Agent
from bblib.agents.EpsilonScheduler import EpsilonScheduler
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
            random.shuffle(self.experiences)
            self.experiences = self.experiences[-self.max_size:]

    def __getitem__(self, idx):
        return self.experiences[idx]


class LRFactory:
    def __init__(self, scheduler: str, **kwargs):
        self.kwargs = kwargs
        self.Scheduler = getattr(torch.optim.lr_scheduler, scheduler)

    def create(self, optimizer: torch.optim.Optimizer):
        return self.Scheduler(optimizer, **self.kwargs)


class DQN(Agent):
    def __init__(self, env_config: EnvironmentConfig,
                 action_counts: List[int],
                 epsilon_scheduler: EpsilonScheduler,
                 episodes_in_memory: int,
                 lr_scheduler_factory: LRFactory):
        super().__init__(action_counts)
        self.env_config = env_config

        self.network = DefaultNetwork(8, [200, 200], self.action_counts, torch.nn.GELU)
        self.epsilon_scheduler = epsilon_scheduler

        self.max_num_experiences = episodes_in_memory*env_config.get_episode_steps()
        self.experience_dataset = ExperienceDataset(self.max_num_experiences)
        self.new_experiences = []

        self.last_observation = None
        self.last_action = None

        self.batch_size = 64
        self.num_train_iters = 128
        self.discount_factor = 0.95

        self.solver = torch.optim.Adam(self.network.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.001)
        self.lr_scheduler = lr_scheduler_factory.create(self.solver)

        self.is_train = False

    def start_episode(self, is_train: bool):
        self.is_train = is_train
        self.new_experiences = []
        self.last_observation = None
        self.last_action = None

    def finish_episode(self):
        if self.is_train:
            self.experience_dataset.add_experiences([
                (self._transform_observation(exp.observation),
                 self._transform_action(exp.action),
                 self._transform_observation(exp.next_observation),
                 torch.tensor(exp.observation.reward),
                 torch.tensor(exp.observation.done).float())
                for exp in self.new_experiences])
            self.epsilon_scheduler.update()

    def step(self, observation: Observation) -> Action:
        # Save experience
        if self.is_train and self.last_observation is not None:
            assert self.last_action is not None
            self.new_experiences.append(Experience(self.last_observation, self.last_action, observation))

        if self.is_train and random.random() < self.epsilon_scheduler.get_epsilon():
            assert 2 == len(self.action_counts)
            # actions = [random.randint(0, c - 1) for c in self.action_counts]

            w1x = (observation.angle.x / self.env_config.max_angle.x + 1) / 2.0 + 0.4
            w2x = (1 - observation.angle.x / self.env_config.max_angle.x) / 2.0 + 0.4
            w1y = (observation.angle.y / self.env_config.max_angle.y + 1) / 2.0 + 0.4
            w2y = (1 - observation.angle.y / self.env_config.max_angle.y) / 2.0 + 0.4

            if 3 == self.action_counts[0] and 3 == self.action_counts[1]:
                probs_x = [w1x/2, 0.2/2, w2x/2]
                probs_y = [w1y/2, 0.2/2, w2y/2]
                values = [0, 1, 2]
            elif 5 == self.action_counts[0] and 5 == self.action_counts[1]:
                probs_x = [w1x/4, w1x/4, 0.2/2, w2x/4, w2x/4]
                probs_y = [w1y/4, w1y/4, 0.2/2, w2y/4, w2y/4]
                values = [0, 1, 2, 3, 4]
            else:
                raise RuntimeError

            actions = [int(np.random.choice(values, p=probs_x)),
                       int(np.random.choice(values, p=probs_y))]
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

    def _loss(self, batch: tuple) -> torch.Tensor:
        observations, actions, next_observations, rewards, dones = batch

        max_next_qs = [q.detach().max(dim=1)[0] for q in self.network(next_observations)]
        target_qs = [(rewards + self.discount_factor * q * (1.0 - dones)) for q in max_next_qs]

        qs = self.network(observations)
        qs = [(q * a).sum(1) for q, a in zip(qs, actions)]

        losses = [(q - target_q) ** 2 for q, target_q in zip(qs, target_qs)]
        loss = torch.stack(losses).mean(0).mean(0)
        return loss

    def _loss2(self, batch: tuple) -> torch.Tensor:
        observations, actions, next_observations, rewards, dones = batch

        next_qs = [q.detach() for q in self.network(next_observations)]
        max_next_q = torch.cat([values.max(axis=1, keepdims=True)[0] for values in next_qs], dim=1).mean(dim=1)
        target_q = rewards + self.discount_factor * max_next_q*(1.0-dones)

        qs = self.network(observations)
        qs = [(q * a).sum(1) for q, a in zip(qs, actions)]

        losses = [(q - target_q) ** 2 for q in qs]
        loss = torch.stack(losses).mean(0).mean(0)
        return loss

    def train(self):
        if len(self.experience_dataset) < self.max_num_experiences // 2:
            return

        self.network.train()
        train_dataloader = DataLoader(self.experience_dataset, batch_size=self.batch_size, shuffle=True)

        for ix, batch in enumerate(train_dataloader):
            self.solver.zero_grad()

            loss = self._loss2(batch)

            loss.backward()
            self.solver.step()

            if ix + 1 >= self.num_train_iters:
                break

        self.lr_scheduler.step()

    def save(self, filename: Path):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename: Path):
        self.network.load_state_dict(torch.load(filename))

    def _transform_observation(self, observation: Observation):
        last_action = observation.last_action if observation.last_action is not None else Action(0, 0)
        return torch.tensor([2.0 * observation.estimated_pos.x / self.env_config.limits.max_x,
                             2.0 * observation.estimated_pos.y / self.env_config.limits.max_y,
                             2.0 * observation.estimated_speed.x / self.env_config.limits.max_x,
                             2.0 * observation.estimated_speed.y / self.env_config.limits.max_y,
                             2.0 * observation.angle.x / self.env_config.max_angle.x,
                             2.0 * observation.angle.y / self.env_config.max_angle.y,
                             float(last_action.x),
                             float(last_action.y)])

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
