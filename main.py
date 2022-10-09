from bblib.agents.Agent import Agent
from bblib.agents.DQN import DQN
from bblib.defs import EnvironmentConfig, Limits, Position, Rotation, Angle, Episode, \
    RandomVirtualEnvironmentConfig, \
    VirtualEnvironmentNoiseConfig
from bblib.environments.Environment import Environment
from bblib.environments.VirtualEnvironment import VirtualEnvironment
from bblib.utils import random_environment_state, random_virtual_ball, draw_state, random_virtual_environment_config
from utils.config import load_config

default_environment_config = EnvironmentConfig(
    0.05,
    Limits(0.097, 0.080),
    Position(90, 80),
    Rotation(20, 20),
    Rotation(361, 364),
    Angle(0.210438646 * (20.0 / 100.0), 0.327671276 * (20.0 / 100.0))
)


def run_episode(env: Environment, agent: Agent, steps) -> Episode:
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
    print(config)

    env_config = default_environment_config
    random_virtual_env_config = RandomVirtualEnvironmentConfig(3.0, 0.10, 0.33)
    virtual_env_noise_cfg = VirtualEnvironmentNoiseConfig(0.01)

    num_episodes = 400
    episode_secs = 15.0
    episode_steps = round(episode_secs / env_config.d_t)
    start_eps = 0.5
    end_eps = 0.1
    eps_episodes = num_episodes // 2

    agent: Agent = DQN(env_config, [3, 3], mem_size=50*episode_steps)

    for i in range(num_episodes):
        env = VirtualEnvironment(env_config,
                                 random_environment_state(),
                                 random_virtual_environment_config(env_config, random_virtual_env_config),
                                 random_virtual_ball(env_config),
                                 virtual_env_noise_cfg)

        agent.epsilon = start_eps - (start_eps-end_eps) * min(1.0, i / eps_episodes)
        episode = run_episode(env, agent, episode_steps)
        avg_reward = sum([observation.reward for observation in episode]) / len(episode)
        print(i, avg_reward, agent.epsilon)

        agent.train()

        if (i + 1) % 100 == 0:
            agent.epsilon = -1.0
            virtual_config = random_virtual_environment_config(env_config, random_virtual_env_config)
            env = VirtualEnvironment(env_config,
                                     random_environment_state(),
                                     virtual_config,
                                     random_virtual_ball(env_config),
                                     virtual_env_noise_cfg)
            episode = run_episode(env, agent, episode_steps)

            frames = []
            for observation in episode:
                frames.append(draw_state(env_config, observation, virtual_config))

            frames[0].save(f"out{i + 1:06d}.gif", format='GIF', append_images=frames[1:],
                           save_all=True, duration=env_config.d_t * 1000, loop=0)

    print("Bye!")


if __name__ == '__main__':
    main()
