import argparse
import pickle
from datetime import datetime
from pathlib import Path

from PIL import Image

from bblib.agents.Agent import Agent
from bblib.defs import SAVES_ROOT, Episode
from bblib.environments.Environment import EnvironmentFactory
from bblib.episode import run_episode
from utils.config import load_config


KEY_ENVIRONMENT_FACTORY = "environmentFactory"
KEY_AGENT = "agent"

SAVE_FOLDER_PREFIX = "test"


def save_episode_to_text(episode: Episode, filename: Path):
    with open(filename, 'wt') as fp:
        for obs in episode:
            fp.write(f"{obs.observed_pos.x} {obs.observed_pos.y} " +
                     f"{obs.estimated_pos.x} {obs.estimated_pos.y} " +
                     f"{obs.estimated_speed.x} {obs.estimated_speed.y} " +
                     f"{obs.angle.x} {obs.angle.y} " +
                     (f"{obs.last_action.x} {obs.last_action.y} " if obs.last_action is not None else "0 0") +
                     f"{obs.reward}\n")


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

    for i in range(args.count):
        env = env_factory.create()
        episode = run_episode(env, agent, args.secs, False, display_duration=args.display_duration)

        episode_filename = save_folder / f"episode_{i:04d}.pickle"
        with open(episode_filename, 'wb') as fp:
            pickle.dump(episode, fp, protocol=pickle.HIGHEST_PROTOCOL)

        episode_text_filename = save_folder / f"episode_{i:04d}.txt"
        save_episode_to_text(episode, episode_text_filename)

        frames = []
        for observation in episode:
            frames.append(Image.fromarray(env.render(observation)))

        gif_filename = save_folder / f"anim_{i:04d}.gif"
        frames[0].save(gif_filename, format='GIF', append_images=frames[1:],
                       save_all=True, duration=env_factory.get_env_config().d_t * 1000, loop=0)

    print("Bye!")


if __name__ == '__main__':
    main()
