import copy
import torch
import torch.nn.functional as F
from model.agent.BaseRLAgent import BaseRLAgent
import pandas as pd


class DQN(BaseRLAgent):
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
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--critic_lr', type=float, default=1e-4,
                            help='decay rate for critic')
        parser.add_argument('--critic_decay', type=float, default=1e-4,
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01,
                            help='mitigation factor')
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

        self.critic_lr = args.critic_lr
        self.critic_decay = args.critic_decay
        self.tau = args.target_mitigate_coef

        # models
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        # controller
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr,
                                                 weight_decay=args.critic_decay)
        self.do_actor_update = True
        self.do_critic_update = True

        # 获取物品质量和流行度
        path = f"/home/liuhao/xcj/KuaiSim-V1/code/dataset/{args.dataset}/"
        self.item_quality = torch.tensor(pd.read_csv(path + 'item_quality.csv').to_numpy()).to(self.device)
        self.item_popularity = torch.tensor(pd.read_csv(path + 'item_popularity.csv').to_numpy()).to(self.device)


        # register models that will be saved
        self.registered_models.append((self.critic, self.critic_optimizer, "_critic"))

    def action_before_train(self):
        """
        Action before training:
        - buffer setup
        - monitor setup
        - run random episodes to build-up the initial buffer
        """
        super().action_before_train()

        # training records
        self.training_history = {'loss': [], 'Q': [], 'next_Q': []}

    def step_train(self):
        """
        @process:
        - buffer.sample(): batch_size --> observation, policy_output, user_response, done_mask, next_observation
            - observation: see self.env.step@output - new_observation
            - policy_output: {
                'state': (B,state_dim),
                'action': (B,K),
                ...}
            - user_feedback: {
                'reward': (B,),
                'immediate_response': (B,K*n_feedback)}
            - done_mask
            - next_observation
        - policy.get_forward(): observation, candidates --> policy_output
        - policy.get_loss(): observation, candidates, policy_output, user_response --> loss
        - optimizer.zero_grad(); loss.backward(); optimizer.step()
        - update training history
        """

        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)
        is_train = True
        # (B, )
        reward = user_feedback['reward'].view(-1)

        # 添加公平性奖励
        item_id = policy_output['action']
        quality = self.item_quality[item_id].reshape(reward.shape[0], -1).reshape(reward.shape[0], -1)
        popularity = self.item_popularity[item_id].reshape(reward.shape[0], -1).reshape(reward.shape[0], -1)
        reward += torch.mean(0.2 * quality * 1 / torch.log10(popularity + 1.1), dim=1)

        # Dqn loss
        # Get current Q estimate
        current_Q = self.apply_critic(observation, policy_output, self.actor)

        # Compute the target Q value
        next_policy_output = self.apply_policy(next_observation, self.actor_target, 0., False, is_train)
        next_Q = self.apply_critic(next_observation, next_policy_output, self.critic_target)
        # (B, )
        target_Q = reward + self.gamma * (~done_mask * next_Q).detach()

        # Compute critic loss
        loss = F.mse_loss(current_Q, target_Q).mean()

        # Regularization loss
        #  critic_reg = current_critic_output['reg']

        if self.do_critic_update and self.critic_lr > 0:
            # Optimize the critic
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
        
        # Update the frozen target models
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        loss_dict = {'loss': loss.item(),
                     'Q': torch.mean(current_Q).item(),
                     'next_Q': torch.mean(next_Q).item()}

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
                      'candidates': self.env.get_candidate_info(observation),
                      'epsilon': epsilon,
                      'do_explore': do_explore,
                      'is_train': is_train,
                      'batch_wise': False}
        out_dict = actor(input_dict)
        out_dict['user_pop_prefer'] = observation['user_history']['user_pop_prefer']
        return out_dict

    def apply_critic(self, observation, policy_output, critic):
        feed_dict = {'state': policy_output['state'],
                     'action': policy_output['action']}
        return self.actor.evaluate(feed_dict)

    def save(self):
        super().save()

    def load(self):
        super().load()
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
