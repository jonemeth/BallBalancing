from bblib.agents.Agent import Agent
from bblib.defs import Episode
from bblib.environments.Environment import Environment


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
