from bblib.agents.Agent import Agent
from bblib.defs import Episode
from bblib.environments.Environment import Environment


def run_episode(env: Environment, agent: Agent, is_train: bool) -> Episode:
    observation = env.observe()
    episode = [observation]

    agent.start_episode(is_train)
    for i in range(env.get_config().get_episode_steps()):
        action = agent.step(observation)
        observation = env.update(action)
        episode.append(observation)
    agent.finish_episode()

    return episode
