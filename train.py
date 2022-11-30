import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image

from bblib.agents.Agent import Agent
from bblib.defs import SAVES_ROOT
from bblib.environments.Environment import EnvironmentFactory
from bblib.episode import run_episode
from utils.config import load_config

MODELS_FOLDER = "models"
GIFS_FOLDER = "gifs"

KEY_ENVIRONMENT_FACTORY = "environmentFactory"
KEY_AGENT = "agent"
KEY_NUM_EPISODES = "num_episodes"
KEY_SAVE_FOLDER_PREFIX = "save_folder_prefix"
KEY_TRAIN_EPISODE_SECS = "train_episode_secs"
KEY_TEST_EPISODE_SECS = "test_episode_secs"


def main():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config_file', type=str)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--display', action="store_true")
    args = parser.parse_args()

    config = load_config(args.config_file)

    num_episodes = int(config[KEY_NUM_EPISODES])
    train_episode_secs = float(config[KEY_TRAIN_EPISODE_SECS])
    test_episode_secs = float(config[KEY_TEST_EPISODE_SECS])

    env_factory: EnvironmentFactory = config.get(KEY_ENVIRONMENT_FACTORY)
    agent: Agent = config.get(KEY_AGENT)
    agent.set_env_config(env_factory.get_env_config(), train_episode_secs)

    if args.model is not None:
        agent.load(args.model)

    save_folder_prefix = config[KEY_SAVE_FOLDER_PREFIX]
    save_folder_suffix = datetime.now().strftime('%y%m%d%H%M%S')
    save_folder = Path(SAVES_ROOT) / (save_folder_prefix+"_"+save_folder_suffix)
    save_folder.mkdir(parents=True, exist_ok=True)

    models_folder = save_folder / MODELS_FOLDER
    models_folder.mkdir(parents=True, exist_ok=True)

    gifs_folder = save_folder / GIFS_FOLDER
    gifs_folder.mkdir(parents=True, exist_ok=True)

    running_reward = None
    running_loss = None
    for i in range(num_episodes):
        env = env_factory.create()

        episode = run_episode(env, agent, train_episode_secs, True, 1e-8 if args.display else -1.0)
        avg_reward = sum([observation.reward for observation in episode]) / len(episode)

        running_reward = avg_reward if running_reward is None else 0.95*running_reward + 0.05*avg_reward
        lr = agent.lr_scheduler.get_last_lr()[0]

        avg_loss = agent.train()
        running_loss = avg_loss if running_loss is None else 0.95*running_loss + 0.05*avg_loss
        running_loss_ = 0.0 if running_loss is None else running_loss
        print(f"it: {i:4d}, avg_reward: {avg_reward:.4f}, running_reward: {running_reward:.4f}, " +
              f"running_loss: {running_loss_:.4f}, epsilon: {agent.epsilon_scheduler.get_epsilon():.4f}, lr: {lr:.6f}")

        if (i + 1) % 100 == 0:
            model_file = models_folder / f"model{i + 1:06d}"
            agent.save(model_file)

        if (i + 1) % 100 == 0 or i >= num_episodes - 5:
            env = env_factory.create()

            episode = run_episode(env, agent, test_episode_secs, False)

            frames = []
            for observation in episode:
                frames.append(Image.fromarray(env.render(observation)))

            gif_file = gifs_folder / f"anim{i + 1:06d}.gif"
            frames[0].save(gif_file, format='GIF', append_images=frames[1:],
                           save_all=True, duration=env_factory.get_env_config().d_t * 1000, loop=0)

    print("Bye!")


if __name__ == '__main__':
    main()
