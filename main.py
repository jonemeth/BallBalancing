import sys

from bblib.agents.Agent import Agent
from bblib.agents.DQN import DQN
from bblib.defs import Episode
from bblib.environments.Environment import Environment, EnvironmentFactory

from utils.config import load_config


KEY_ENVIRONMENT_FACTORY = "environmentFactory"


def run_episode(env: Environment, agent: Agent, steps: int) -> Episode:
    observation = env.observe()
    episode = [observation]

    agent.start()
    for i in range(steps):
        action = agent.step(observation)
        observation = env.update(action)
        episode.append(observation)
    agent.finish()

    return episode


def main():
    config = load_config('test.yaml')
    env_factory: EnvironmentFactory = config.get(KEY_ENVIRONMENT_FACTORY)

    num_episodes = 400
    episode_secs = 15.0
    episode_steps = round(episode_secs / env_factory.get_env_config().d_t)
    start_eps = 0.5
    end_eps = 0.1
    eps_episodes = num_episodes // 2

    agent: Agent = DQN(env_factory.get_env_config(), [3, 3], mem_size=50*episode_steps)

    for i in range(num_episodes):
        env = env_factory.create()

        agent.epsilon = start_eps - (start_eps-end_eps) * min(1.0, i / eps_episodes)
        episode = run_episode(env, agent, episode_steps)
        avg_reward = sum([observation.reward for observation in episode]) / len(episode)
        print(i, avg_reward, agent.epsilon)

        agent.train()

        if (i + 1) % 100 == 0:
            agent.epsilon = -1.0
            env = env_factory.create()

            episode = run_episode(env, agent, episode_steps)

            frames = []
            for observation in episode:
                frames.append(env.render(observation))

            frames[0].save(f"out{i + 1:06d}.gif", format='GIF', append_images=frames[1:],
                           save_all=True, duration=env_factory.get_env_config().d_t * 1000, loop=0)

    print("Bye!")


if __name__ == '__main__':
    main()
