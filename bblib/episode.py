import time
from pathlib import Path

import cv2
from PIL import Image

from bblib.agents.Agent import Agent
from bblib.defs import Episode, Observation
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


def save_episode_to_text(episode: Episode, filename: Path):
    with open(filename, 'wt') as fp:
        for obs in episode:
            fp.write(f"{obs.observed_pos.x} {obs.observed_pos.y} " +
                     f"{obs.estimated_pos.x} {obs.estimated_pos.y} " +
                     f"{obs.estimated_speed.x} {obs.estimated_speed.y} " +
                     f"{obs.angle.x} {obs.angle.y} " +
                     (f"{obs.last_action.x} {obs.last_action.y} " if obs.last_action is not None else "0 0") +
                     f"{obs.reward}\n")


def save_episode_to_gif(episode: Episode, env: Environment, filename: Path):
    frames = []
    for observation in episode:
        frames.append(Image.fromarray(env.render(observation)))

    frames[0].save(filename, format='GIF', append_images=frames[1:],
                   save_all=True, duration=env.get_config().d_t * 1000, loop=0)


def in_region(obs: Observation, rs: float):
    return abs(obs.estimated_pos.x) <= rs / 200 and abs(obs.estimated_pos.y) <= rs / 200


def evaluate_episode(episode: Episode, env: Environment, skip_secs = 2.0) -> dict:
    skip_indices = int(skip_secs / env.get_config().d_t)
    chunk = episode[skip_indices:]
    region_sizes = [1, 2, 3, 4, 5]
    results = {}

    # Evaluate speed
    for rs in region_sizes:
        i = len(episode) - 1
        while i >= 0 and in_region(episode[i], rs):
            i -= 1
        speed = (1 + i) * env.get_config().d_t
        results[f"s{rs}"] = speed

    # Evaluate precision
    for rs in region_sizes:
        n = len([obs for obs in chunk if in_region(obs, rs)])
        results[f"p{rs}"] = n / len(chunk)

    return results
