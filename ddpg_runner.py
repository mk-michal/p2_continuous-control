import time

import numpy as np

from ddpg_model import DDPGAgentAdjustedUnity
from utils import get_ddpg_config


def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
        if config.log_interval and not agent.total_steps % config.log_interval:
            agent.logger.info(
                f'steps {agent.total_steps}, {config.log_interval / (time.time() - t0)} steps/s '
                f'Reward over last 10 episodes {np.mean(agent.reward_per_episode[-10:])}'
            )
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()


if __name__ == '__main__':

    brain_name = 'ReacherBrain'
    config = get_ddpg_config(brain_name)
    run_steps(DDPGAgentAdjustedUnity(config, brain_name))
