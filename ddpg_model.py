import datetime
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from deep_rl import DDPGAgent, Config, BaseAgent, to_np


class DDPGAgentAdjustedUnity(BaseAgent):
    '''
    class for adjusting DDPG agent from ShnagtonZheng to our Unity environment
    '''
    def __init__(self, config, brain_name):
        BaseAgent.__init__(self, config)
        self.save_path = os.path.join('data', str(datetime.datetime.now()).replace(' ', '_'))
        self.brain_name = brain_name
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None
        self.reward_per_episode = []
        self.episode_reward = []
        # save config file on class init

        config_to_save = {key: str(value) for key, value in config.__dict__.items()}
        self.logger.info(f'Saving config file as {os.path.join(self.save_path, "config.json")}')
        os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, 'config.json'), 'w') as j:
            json.dump(config_to_save, j, indent=4, sort_keys=True)


    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def eval_episode(self):
        env_info = self.task.reset()[self.brain_name]
        states = env_info.vector_observations
        all_rewards = []
        n_iterations = 0
        while True:
            n_iterations += 1
            action = self.eval_step(states)
            env_info = self.task.step(action)[self.brain_name]
            states = env_info.vector_observations
            done = env_info.local_done
            all_rewards.extend(env_info.rewards)

            if any(done):
                break

        return all_rewards
    def reset(self):
        env_info = self.task.reset()[self.brain_name]


    def step(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            env_info = self.task.reset()[self.brain_name]
            self.state = env_info.vector_observations
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = np.random.randn(1, 4)
            action = np.clip(action, -1,1)
        else:
            action = self.network(self.state)
            action = action.cpu().detach().numpy()
            action += self.random_process.sample()
        action = np.clip(action, -1, 1)

        env_info = self.task.step(action)[self.brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done

        next_state = self.config.state_normalizer(next_state)
        reward = self.config.reward_normalizer(reward)
        self.episode_reward.extend(reward)
        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1 - np.asarray(done, dtype=np.int32),
        ))

        if done[0]:
            self.reward_per_episode.append(np.sum(self.episode_reward))
            self.episode_reward = []
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.warm_up:
            transitions = self.replay.sample()
            states = torch.tensor(transitions.state, device=Config.DEVICE, dtype=torch.float)
            actions = torch.tensor(transitions.action, device=Config.DEVICE, dtype=torch.float)
            rewards = torch.tensor(transitions.reward, dtype=torch.float, device=Config.DEVICE).unsqueeze(-1).float()
            next_states = torch.tensor(transitions.next_state, device=Config.DEVICE, dtype=torch.float)
            mask = torch.tensor(transitions.mask, device=Config.DEVICE, dtype=torch.float).unsqueeze(-1)

            phi_next = self.target_network.feature(next_states).float()
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = config.discount * mask * q_next

            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states).float()
            q = self.network.critic(phi, actions)
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)

    def save(self, filename):
        torch.save(self.network.state_dict(), os.path.join(self.save_path, f'{filename}.model'))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

        plt.plot(self.reward_per_episode)
        plt.savefig(os.path.join(self.save_path, 'results.png'))

    # def load(self, filename):
    #     state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
    #     self.network.load_state_dict(state_dict)