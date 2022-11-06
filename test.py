import argparse
import os
from datetime import datetime
from pathlib import Path

from PIL import Image

from bblib.agents.Agent import Agent
from bblib.environments.Environment import EnvironmentFactory
from bblib.episode import run_episode
from utils.config import load_config

SAVES_ROOT = "saves"

KEY_ENVIRONMENT_FACTORY = "environmentFactory"
KEY_AGENT = "agent"


def main():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config_file', type=str)
    parser.add_argument('save_folder', type=str)
    parser.add_argument('gif_basename', type=str)
    parser.add_argument('count', type=int)
    args = parser.parse_args()

    config = load_config(args.config_file)

    env_factory: EnvironmentFactory = config.get(KEY_ENVIRONMENT_FACTORY)
    agent: Agent = config.get(KEY_AGENT)

    model_file = Path(args.save_folder) / "model"
    agent.load(model_file)

    for i in range(args.count):
        gif_filename = f"{args.gif_basename}_{i:04d}.gif"
        env = env_factory.create()
        episode = run_episode(env, agent, False)

        frames = []
        for observation in episode:
            frames.append(Image.fromarray(env.render(observation)))

        frames[0].save(gif_filename, format='GIF', append_images=frames[1:],
                       save_all=True, duration=env_factory.get_env_config().d_t * 1000, loop=0)

    # for i in range(num_episodes):
    #     env = env_factory.create()
    #
    #     episode = run_episode(env, agent, True)
    #     avg_reward = sum([observation.reward for observation in episode]) / len(episode)
    #     print(i, avg_reward, agent.epsilon_scheduler.get_epsilon())
    #
    #     agent.train()
    #
    #     if (i + 1) % 100 == 0 or i >= num_episodes - 10:
    #         env = env_factory.create()
    #
    #         episode = run_episode(env, agent, False)
    #
    #         frames = []
    #         for observation in episode:
    #             frames.append(Image.fromarray(env.render(observation)))
    #
    #         frames[0].save(f"out{i + 1:06d}.gif", format='GIF', append_images=frames[1:],
    #                        save_all=True, duration=env_factory.get_env_config().d_t * 1000, loop=0)


    print("Bye!")


if __name__ == '__main__':
    main()
