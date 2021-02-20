import deep_rl
import torch
from deep_rl import Config, DeterministicActorCriticNet, FCBody, LinearSchedule
from torch.nn import functional as F
from unityagents import UnityEnvironment


def get_ddpg_config():
    config = deep_rl.Config()
    Config.DEVICE = torch.device('cuda:0')
    config.state_dim = 33
    config.action_dim = 4
    config.warm_up = 1000
    config.target_network_mix = 0.005

    config.task_fn = lambda: UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64', worker_id=1, seed=1)
    config.max_steps = int(1e6)
    config.log_interval = 1000
    config.eval_interval = 0 # dont evaluate at all during training
    config.save_interval = 100000

    config.actor_hidden = (400,300)
    config.critic_hidden = (400, 300)
    config.actor_lr = 1e-4
    config.critic_lr =  1e-3


    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, config.actor_hidden, gate=F.relu),
        critic_body=FCBody(config.state_dim + config.action_dim, config.critic_hidden, gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=config.actor_lr),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=config.critic_lr))

    config.replay_fn = lambda: deep_rl.UniformReplay(memory_size=int(1e6), batch_size=100)
    config.discount = 0.99
    config.random_process_fn = lambda: deep_rl.OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    return config