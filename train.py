import argparse
import os
from datetime import datetime
from pathlib import Path

from PIL import Image

from bblib.agents.Agent import Agent
from bblib.defs import MODEL_FILENAME
from bblib.environments.Environment import EnvironmentFactory
from bblib.episode import run_episode
from utils.config import load_config

SAVES_ROOT = "saves"

KEY_ENVIRONMENT_FACTORY = "environmentFactory"
KEY_AGENT = "agent"
KEY_NUM_EPISODES = "num_episodes"
KEY_SAVE_FOLDER_PREFIX = "save_folder_prefix"


def main():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()

    config = load_config(args.config_file)

    env_factory: EnvironmentFactory = config.get(KEY_ENVIRONMENT_FACTORY)
    agent: Agent = config.get(KEY_AGENT)

    save_folder_prefix = config[KEY_SAVE_FOLDER_PREFIX]
    num_episodes = int(config[KEY_NUM_EPISODES])

    save_folder_suffix = datetime.now().strftime('%y%m%d%H%M%S')
    save_folder = Path(SAVES_ROOT) / (save_folder_prefix+"_"+save_folder_suffix)
    model_file = save_folder / MODEL_FILENAME
    os.makedirs(save_folder, exist_ok=True)

    running_reward = None
    for i in range(num_episodes):
        env = env_factory.create()

        episode = run_episode(env, agent, True)
        avg_reward = sum([observation.reward for observation in episode]) / len(episode)

        running_reward = avg_reward if running_reward is None else 0.99*running_reward + 0.01*avg_reward
        lr = agent.lr_scheduler.get_last_lr()[0]

        print(i, avg_reward, running_reward, agent.epsilon_scheduler.get_epsilon(), lr)

        agent.train()

        if (i + 1) % 100 == 0 or i >= num_episodes - 5:
            env = env_factory.create()

            episode = run_episode(env, agent, False)

            frames = []
            for observation in episode:
                frames.append(Image.fromarray(env.render(observation)))

            frames[0].save(f"out{i + 1:06d}.gif", format='GIF', append_images=frames[1:],
                           save_all=True, duration=env_factory.get_env_config().d_t * 1000, loop=0)

    agent.save(model_file)
    print("Bye!")


if __name__ == '__main__':
    main()
