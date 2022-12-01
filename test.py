import argparse
import pickle
import json
from datetime import datetime
from pathlib import Path

from PIL import Image

from bblib.agents.Agent import Agent
from bblib.defs import SAVES_ROOT, Episode
from bblib.environments.Environment import EnvironmentFactory
from bblib.episode import run_episode, save_episode_to_text, save_episode_to_gif
from utils.config import load_config


KEY_ENVIRONMENT_FACTORY = "environmentFactory"
KEY_AGENT = "agent"

SAVE_FOLDER_PREFIX = "test"


def main():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config_file', type=str)
    parser.add_argument('model_file', type=str)
    parser.add_argument('gif_basename', type=str)
    parser.add_argument('secs', type=float)
    parser.add_argument('count', type=int)
    parser.add_argument('--display_duration', type=float, default=-1.0)
    args = parser.parse_args()

    config = load_config(args.config_file)

    env_factory: EnvironmentFactory = config.get(KEY_ENVIRONMENT_FACTORY)
    agent: Agent = config.get(KEY_AGENT)
    agent.set_env_config(env_factory.get_env_config(), 0)

    model_file = args.model_file
    agent.load(model_file)

    save_folder_suffix = datetime.now().strftime('%y%m%d%H%M%S')
    save_folder = Path(SAVES_ROOT) / (SAVE_FOLDER_PREFIX+"_"+save_folder_suffix)
    save_folder.mkdir(parents=True, exist_ok=True)

    with open(save_folder / 'args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    for i in range(args.count):
        env = env_factory.create()
        episode = run_episode(env, agent, args.secs, False, display_duration=args.display_duration)

        episode_filename = save_folder / f"episode_{i:06d}.pickle"
        with open(episode_filename, 'wb') as fp:
            pickle.dump(episode, fp, protocol=pickle.HIGHEST_PROTOCOL)

        episode_text_filename = save_folder / f"episode_{i:04d}.txt"
        save_episode_to_text(episode, episode_text_filename)

        gif_filename = save_folder / f"anim_{i:04d}.gif"
        save_episode_to_gif(episode, env, gif_filename)

    print("Bye!")


if __name__ == '__main__':
    main()
