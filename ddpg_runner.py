import argparse
import time

import deep_rl
from deep_rl import LinearSchedule
from deep_rl.agent import DeterministicActorCriticNet, FCBody, Config
from notebook.jstest import argparser
from unityagents import UnityEnvironment

import torch
import torch.nn.functional as F

from ddpg_model import DDPGAgentAdjustedUnity


def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
        if config.log_interval and not agent.total_steps % config.log_interval:
            agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()


def get_ddpg_config(brain_name):
    config = deep_rl.Config()
    Config.DEVICE = torch.device('cuda:0')
    config.state_dim = 33
    config.action_dim = 4
    config.warm_up = 1000
    config.target_network_mix = 0.001

    config.task_fn = lambda: UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64', worker_id=1, seed=1)
    config.max_steps = int(1e6)
    config.eval_interval = 10000
    config.eval_episodes = 20


    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (100, 250, 100 ), gate=F.relu),
        critic_body=FCBody(config.state_dim + config.action_dim, (100, 250, 100), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: deep_rl.UniformReplay(memory_size=int(1e6), batch_size=100)
    config.discount = 0.99
    config.random_process_fn = lambda: deep_rl.OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.target_network_mix = 1e-3
    run_steps(DDPGAgentAdjustedUnity(config, brain_name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episode', default = 1000, type=int)
    parser.add_argument('--n-steps', default=1000, type=int)
    args = parser.parse_args()

    brain_name = 'ReacherBrain'
    get_ddpg_config(brain_name)
    # agent = DDPGAgentAdjustedUnity(config, 'ReacherBrain')
    # for episode in range(args.n_episode):
