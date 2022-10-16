from bblib.agents.Agent import Agent
from bblib.environments.Environment import EnvironmentFactory
from bblib.episode import run_episode

from utils.config import load_config


KEY_ENVIRONMENT_FACTORY = "environmentFactory"
KEY_AGENT = "agent"


def main():
    config = load_config('exps/default/train.yaml')
    env_factory: EnvironmentFactory = config.get(KEY_ENVIRONMENT_FACTORY)
    agent: Agent = config.get(KEY_AGENT)

    num_episodes = 400

    for i in range(num_episodes):
        env = env_factory.create()

        episode = run_episode(env, agent, True)
        avg_reward = sum([observation.reward for observation in episode]) / len(episode)
        print(i, avg_reward, agent.epsilon_scheduler.get_epsilon())

        agent.train()

        if (i + 1) % 100 == 0:
            agent.epsilon = -1.0
            env = env_factory.create()

            episode = run_episode(env, agent, False)

            frames = []
            for observation in episode:
                frames.append(env.render(observation))

            frames[0].save(f"out{i + 1:06d}.gif", format='GIF', append_images=frames[1:],
                           save_all=True, duration=env_factory.get_env_config().d_t * 1000, loop=0)

    print("Bye!")


if __name__ == '__main__':
    main()
