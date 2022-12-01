import argparse
import pickle
from datetime import datetime
from pathlib import Path

from bblib.agents.Agent import Agent
from bblib.defs import SAVES_ROOT
from bblib.environments.Environment import EnvironmentFactory
from bblib.episode import run_episode, save_episode_to_gif, evaluate_episode
from utils.config import load_config

MODELS_FOLDER = "models"
GIFS_FOLDER = "gifs"
EPISODES_FOLDER = "episodes"
LOG_FILENAME = "log.txt"
NUM_TEST_EPISODES = 10  # at every nth train episodes

KEY_ENVIRONMENT_FACTORY = "environmentFactory"
KEY_AGENT = "agent"
KEY_NUM_EPISODES = "num_episodes"
KEY_SAVE_FOLDER_PREFIX = "save_folder_prefix"
KEY_TRAIN_EPISODE_SECS = "train_episode_secs"
KEY_TEST_EPISODE_SECS = "test_episode_secs"


class Logger:
    def __init__(self, filename: Path):
        self.fp = open(filename, "w+t")

    def log(self, msg: str):
        print(msg, flush=True)
        self.fp.write(msg+"\n")
        self.fp.flush()


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
    save_folder = Path(SAVES_ROOT) / (save_folder_prefix + "_" + save_folder_suffix)
    save_folder.mkdir(parents=True, exist_ok=True)

    models_folder = save_folder / MODELS_FOLDER
    models_folder.mkdir(parents=True, exist_ok=True)

    gifs_folder = save_folder / GIFS_FOLDER
    gifs_folder.mkdir(parents=True, exist_ok=True)

    episodes_folder = save_folder / EPISODES_FOLDER
    episodes_folder.mkdir(parents=True, exist_ok=True)

    logger = Logger(save_folder / LOG_FILENAME)

    running_reward = None
    running_loss = None
    for episode_ix in range(num_episodes):
        env = env_factory.create()

        episode = run_episode(env, agent, train_episode_secs, True, 1e-8 if args.display else -1.0)
        avg_reward = sum([observation.reward for observation in episode]) / len(episode)

        running_reward = avg_reward if running_reward is None else 0.95 * running_reward + 0.05 * avg_reward
        lr = agent.lr_scheduler.get_last_lr()[0]

        avg_loss = agent.train()
        running_loss = avg_loss if running_loss is None else 0.95 * running_loss + 0.05 * avg_loss
        running_loss_ = 0.0 if running_loss is None else running_loss
        logger.log(f"it: {1+episode_ix:4d}, avg_reward: {avg_reward:.4f}, running_reward: {running_reward:.4f}, " +
                   f"running_loss: {running_loss_:.4f}, epsilon: {agent.epsilon_scheduler.get_epsilon():.4f}, "
                   f"lr: {lr:.6f}")

        # episode_filename = episodes_folder / f"train_{episode_ix:06d}.pickle"
        # with open(episode_filename, 'wb') as fp:
        #     pickle.dump(episode, fp, protocol=pickle.HIGHEST_PROTOCOL)

        if (episode_ix + 1) % 100 == 0 or episode_ix == num_episodes - 1:
            model_file = models_folder / f"model{episode_ix + 1:06d}"
            agent.save(model_file)

            all_evals = {}
            for test_ix in range(NUM_TEST_EPISODES):
                env = env_factory.create()
                episode = run_episode(env, agent, test_episode_secs, False)

                episode_filename = episodes_folder / f"test_{episode_ix:06d}_{test_ix:04d}.pickle"
                with open(episode_filename, 'wb') as fp:
                    pickle.dump(episode, fp, protocol=pickle.HIGHEST_PROTOCOL)

                gif_file = gifs_folder / f"test{episode_ix + 1:06d}_{test_ix:04d}.gif"
                save_episode_to_gif(episode, env, gif_file)

                for k, v in evaluate_episode(episode, env).items():
                    if k not in all_evals.keys():
                        all_evals[k] = [v]
                    else:
                        all_evals[k].append(v)

            eval_line = f"ev: {1+episode_ix:4d},"
            for k, v in all_evals.items():
                mean_val = sum(v) / NUM_TEST_EPISODES
                eval_line += f" {k}: {mean_val:.4f},"
            logger.log(eval_line)

    logger.log("Bye!")


if __name__ == '__main__':
    main()
