import time

import cv2

from bblib.agents.Agent import Agent
from bblib.defs import Episode
from bblib.environments.Environment import Environment


def run_episode(env: Environment, agent: Agent, secs: float, is_train: bool, display_duration: float = -1.0) -> Episode:
    observation = env.observe()
    episode = [observation]

    agent.start_episode(is_train)
    for i in range(env.get_config().get_episode_steps(secs)):
        action = agent.step(observation)
        observation = env.update(action)
        episode.append(observation)

        if display_duration > 0.0:
            cv2.imshow('image', env.render(observation))
            cv2.waitKey(1)
            time.sleep(display_duration)

    agent.finish_episode()

    return episode
