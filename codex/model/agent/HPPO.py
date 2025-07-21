import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agent.BaseOnPolicyRLAgent import BaseOnPolicyRLAgent
import pandas as pd
from torch import nn


class HPPO(BaseOnPolicyRLAgent):
    @staticmethod
    def parse_model_args(parser):
        """
        args:
        - critic_lr
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
            - gamma
            - reward_func
            - n_iter
            - train_every_n_step
            - start_policy_train_at_step
            - initial_epsilon
            - final_epsilon
            - elbow_epsilon
            - explore_rate
            - do_explore_in_train
            - check_episode
            - save_episode
            - save_path
            - actor_lr
            - actor_decay
            - batch_size
        """
        parser = BaseOnPolicyRLAgent.parse_model_args(parser)
        parser.add_argument('--critic_lr', type=float, default=1e-4,
                            help='decay rate for critic')
        parser.add_argument('--critic_decay', type=float, default=1e-4,
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01,
                            help='mitigation factor')
        parser.add_argument('--train_epoch_num', type=int, default=4,
                            help='train epoch num')
        parser.add_argument('--eps_clip', type=float, default=0.8,
                            help='eps_clip')
        return parser

    def __init__(self, *input_args):
        """
        components:
        - critic
        - critic_optimizer
        - actor_target
        - critic_target
        - components from BaseRLAgent:
            - env
            - actor
            - actor_optimizer
            - buffer
            - exploration_scheduler
            - registered_models
        """
        args, env, actor, critic, buffer = input_args
        super().__init__(args, env, actor, buffer)

        # 读取用户流行度偏好、物品类型
        path = f"/home/ucas2/xcj/KuaiSim-V1/code/dataset/{args.dataset}/"
        self.user_pop_ratios = torch.tensor(pd.read_csv(path + 'user_pop_ratio.csv').to_numpy()).to(self.device)
        self.item_types = torch.tensor(pd.read_csv(path + 'item_types.csv').to_numpy()).to(self.device)

        self.critic_lr = args.critic_lr
        self.critic_decay = args.critic_decay
        self.tau = args.target_mitigate_coef
        self.train_epoch_num = args.train_epoch_num
        self.eps_clip = args.eps_clip
        self.aug_weight = args.aug_weight
        self.update_step = 0
        self.MseLoss = nn.MSELoss()

        # models
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)

        # controller
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.h_user_encoder.parameters(), 'lr': args.actor_lr},
            {'params': self.actor.l_user_encoder.parameters(), 'lr': args.actor_lr},
            {'params': self.actor.h_action_layer.parameters(), 'lr': args.actor_lr},
            {'params': self.actor.l_action_layer.parameters(), 'lr': args.actor_lr},
            # {'params': self.actor.h_action_var, 'lr': args.actor_lr / 20},
            # {'params': self.actor.l_action_var, 'lr': args.actor_lr / 20},
            {'params': self.critic.parameters(), 'lr': args.critic_lr},
            # {'params': self.policy.action_var, 'lr': lr_actor}
        ])
        self.do_actor_update = True
        self.do_critic_update = True

        # register models that will be saved
        self.registered_models.append((self.actor, self.optimizer, "_actor"))
        self.registered_models.append((self.critic, self.optimizer, "_critic"))

    def setup_monitors(self):
        """
        This is used in super().action_before_train() in super().train()
        Then the agent will call several rounds of run_episode_step() for collecting initial random data samples
        """
        super().setup_monitors()
        self.training_history.update({'h_actor_loss': [], 'l_actor_loss': [],'critic_loss': [], 'dist_entropy_loss': [],
                                      'H_V': [], 'next_H_V': [], 'L_V': [], 'next_L_V': []})

    def step_train(self):
        """
        @process:
        - get sample
        - calculate V'(s_{t+1}) and V(s_t)
        - critic loss: TD error loss
        - critic optimization
        - actor loss: ratios * advantages maximization
        - actor optimization
        """
        observation, policy_output, user_feedback, done_mask, n_observation = self.buffer.sample()
        h_old_log_prob = policy_output['h_action_log_prob'].view(-1)
        l_old_log_prob = policy_output['l_action_log_prob'].view(-1)
        reward = user_feedback['reward'].view(-1)
        total_len = policy_output['h_action_log_prob'].shape[0]
        unpopular_item_ratio = policy_output['unpopular_item_ratio'].view(-1)
        is_train = True
        # Optimize policy for K epochs
        for _ in range(self.train_epoch_num):
            idxes = np.arange(int((total_len - 1) / self.batch_size))
            np.random.shuffle(idxes)
            self.update_step += idxes.shape[0]
            for i in idxes:
                # Evaluating old actions and values
                start_idx, end_idx = i * self.batch_size, min((i + 1) * self.batch_size, total_len - 1)
                idx = torch.arange(start_idx, end_idx).to(self.device)
                neg_idx = torch.randperm(total_len)[:idx.shape[0]]
                current_observation = {}
                current_observation['user_profile'] = {k: v[idx] for k, v in observation['user_profile'].items()}
                current_observation['user_history'] = {k: v[idx] for k, v in observation['user_history'].items()}
                current_policy_output = {k: v[idx] for k, v in policy_output.items()}
                neg_policy_output = {k: v[neg_idx] for k, v in policy_output.items()}
                current_h_old_log_prob = h_old_log_prob[idx]
                current_l_old_log_prob = l_old_log_prob[idx]
                current_done = done_mask[idx]
                current_reward = reward[idx]
                current_unpopular_item_ratio = unpopular_item_ratio[idx]
                # 获取用户id和推荐物品id
                user_id = current_observation['user_profile']['user_id']#.clone().detach()
                item_id = current_policy_output['effect_action']#.clone().detach()
                # 获取当前用户流行度偏好、推荐物品类型
                # (B, 1)
                user_pop_prefer = current_observation['user_history']['user_pop_prefer'].reshape(-1, 1) #self.user_pop_ratios[user_id - 1].reshape(-1, 1)
                # (B, L)
                item_type = self.item_types[item_id].reshape(current_reward.shape[0], -1)
                # 添加公平性奖励
                # new_reward = org_reward * [(1 - item_pop) * (1 - up) + item_pop * up] * 2
                h_current_reward = current_reward
                # print(f'mushxx{current_reward, torch.mean((1 - item_type) / (0.1 + user_pop_prefer), dim=1) * 0.1}')
                # print(current_unpopular_item_ratio / (0.1 + user_pop_prefer) * 0.1)

                if self.update_step > 00:
                    h_current_reward = current_reward + current_unpopular_item_ratio / (1 + user_pop_prefer) * 0.1 #torch.mean((1 - item_type) / (1 + user_pop_ratio), dim=1) * 0.1
                l_current_reward = current_reward #+ current_unpopular_item_ratio / (1 + user_pop_prefer) * 0.1 #+ torch.mean((1 - item_type) / (1 + user_pop_ratio), dim=1) * 0.1

                next_observation = {}
                next_observation['user_profile'] = {k: v[idx] for k, v in n_observation['user_profile'].items()}
                next_observation['user_history'] = {k: v[idx] for k, v in n_observation['user_history'].items()}

                h_log_prob, h_dist_entropy, l_log_prob, l_dist_entropy, aug_loss = self.actor.evaluate(current_policy_output, neg_policy_output, user_pop_prefer)
                # Finding the ratio (pi_theta / pi_theta__old)
                h_ratios = torch.exp(h_log_prob - current_h_old_log_prob.detach())
                l_ratios = torch.exp(l_log_prob - current_l_old_log_prob.detach())

                # match state_values tensor dimensions with rewards tensor
                current_critic_output = self.apply_critic(current_observation, current_policy_output, self.critic)
                current_h_state_values = current_critic_output['h_v']
                current_l_state_values = current_critic_output['l_v']

                # Compute the target V value
                next_policy_output = self.apply_policy(next_observation, self.actor_target, 0., False, is_train)
                target_critic_output = self.apply_critic(next_observation, next_policy_output, self.critic_target)
                next_h_state_values = target_critic_output['h_v']
                next_l_state_values = target_critic_output['l_v']
                target_h_state_values = self.gamma * torch.squeeze(next_h_state_values) * (
                    ~current_done) + h_current_reward
                target_l_state_values = self.gamma * torch.squeeze(next_l_state_values) * (
                    ~current_done) + l_current_reward

                # Finding Surrogate Loss
                h_advantages = target_h_state_values - current_h_state_values.detach()
                h_surr1 = h_ratios * h_advantages
                h_surr2 = torch.clamp(h_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * h_advantages
                l_advantages = target_l_state_values - current_l_state_values.detach()
                l_surr1 = l_ratios * l_advantages
                l_surr2 = torch.clamp(l_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * l_advantages

                # final loss of clipped objective PPO
                h_actor_loss = -torch.min(h_surr1, h_surr2)
                l_actor_loss = - torch.min(l_surr1, l_surr2)
                actor_loss = - torch.min(h_surr1, h_surr2) - torch.min(l_surr1, l_surr2)
                critic_loss = (self.MseLoss(current_h_state_values.float(), target_h_state_values.float())
                               + self.MseLoss(current_l_state_values.float(), target_l_state_values.float())) * 1
                dist_entropy_loss = (- h_dist_entropy - l_dist_entropy) * 0.0005
                loss = actor_loss + critic_loss + dist_entropy_loss + aug_loss * self.aug_weight

                # take gradient step
                self.optimizer.zero_grad()
                # 异常检测开启
                # torch.autograd.set_detect_anomaly(True)
                # 反向传播时检测是否有异常值，定位code
                with torch.autograd.detect_anomaly():
                    loss.mean().backward()
                # 梯度截断
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10, norm_type=2)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10, norm_type=2)
                self.optimizer.step()
                loss_dict = {'h_actor_loss': h_actor_loss.mean().item(),
                             'l_actor_loss': l_actor_loss.mean().item(),
                             'critic_loss': critic_loss.mean().item(),
                             'dist_entropy_loss': dist_entropy_loss.mean().item(),
                             'H_V': torch.mean(current_h_state_values).item(),
                             'L_V': torch.mean(current_l_state_values).item(),
                             'next_H_V': torch.mean(next_h_state_values).item(),
                             'next_L_V': torch.mean(next_l_state_values).item()}
        #print(f'mush{self.update_step}')       

        # Copy new weights into old policy
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for k in loss_dict:
            if k in self.training_history:
                try:
                    self.training_history[k].append(loss_dict[k].item())
                except:
                    self.training_history[k].append(loss_dict[k])

        return loss_dict

    def apply_policy(self, observation, actor, *policy_args):
        """
        @input:
        - observation:{'user_profile':{
                           'user_id': (B,)
                           'uf_{feature_name}': (B,feature_dim), the user features}
                       'user_history':{
                           'history': (B,max_H)
                           'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}
        - actor: the actor model
        - epsilon: scalar
        - do_explore: boolean
        - is_train: boolean
        @output:
        - policy_output
        """
        epsilon = policy_args[0]
        do_explore = policy_args[1]
        is_train = policy_args[2]
        input_dict = {'observation': observation,
                      # 默认候选集为整个物品集合
                      'candidates': self.env.get_candidate_info(observation),
                      'epsilon': epsilon,
                      'do_explore': do_explore,
                      'is_train': is_train,
                      'batch_wise': False}
        out_dict = actor(input_dict)
        return out_dict

    def apply_critic(self, observation, policy_output, critic):
        feed_dict = {'h_state': policy_output['h_state'], 
                     'l_state': policy_output['l_state'], 
                     'user_pop_prefer': observation['user_history']['user_pop_prefer'].reshape(-1, 1), 
                     'h_action': policy_output['h_action']}
        return critic(feed_dict)

    def save(self):
        super().save()

    def load(self):
        super().load()
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
