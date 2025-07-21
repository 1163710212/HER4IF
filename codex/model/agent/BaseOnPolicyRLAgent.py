import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm

import utils
from model.reward import *
import json


class BaseOnPolicyRLAgent():
    """
    RL Agent controls the overall learning algorithm:
    - objective functions for the policies and critics
    - design of reward function
    - how many steps to train
    - how to do exploration
    - loading and saving of models

    Main interfaces:
    - train
    """

    @staticmethod
    def parse_model_args(parser):
        """
        args:
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
        # basic settings
        parser.add_argument('--gamma', type=float, default=0.95,
                            help='reward discount')
        parser.add_argument('--reward_func', type=str, default='get_retention_reward',
                            help='reward function name')
        parser.add_argument('--n_iter', type=int, nargs='+', default=[2000],
                            help='number of training iterations')
        parser.add_argument('--train_every_n_step', type=int, default=1,
                            help='number of training iterations')

        # exploration control
        parser.add_argument('--initial_epsilon', type=float, default=0.5,
                            help='probability for using uniform exploration')
        parser.add_argument('--final_epsilon', type=float, default=0.01,
                            help='probability for using uniform exploration')
        parser.add_argument('--elbow_epsilon', type=float, default=1.0,
                            help='probability for using uniform exploration')
        parser.add_argument('--explore_rate', type=float, default=1.0,
                            help='probability of engaging exploration')
        parser.add_argument('--do_explore_in_train', action='store_true',
                            help='probability of engaging exploration')

        # monitoring
        parser.add_argument('--check_episode', type=int, default=100,
                            help='number of iterations to check output and evaluate')
        parser.add_argument('--save_episode', type=int, default=1000,
                            help='number of iterations to save models')
        parser.add_argument('--save_path', type=str, required=True,
                            help='save path for networks')

        # learning
        parser.add_argument('--actor_lr', type=float, default=1e-4,
                            help='learning rate for actor')
        parser.add_argument('--actor_decay', type=float, default=1e-4,
                            help='regularization factor for actor learning')
        parser.add_argument('--batch_size', type=int, default=64,
                            help='training batch size')

        return parser

    def __init__(self, *input_args):
        args, env, actor, buffer = input_args

        self.device = args.device

        # hyperparameters
        self.gamma = args.gamma
        self.reward_func = eval(args.reward_func)
        self.n_iter = args.n_iter
        self.train_every_n_step = args.train_every_n_step

        self.initial_epsilon = args.initial_epsilon
        self.final_epsilon = args.final_epsilon
        self.elbow_epsilon = args.elbow_epsilon
        self.explore_rate = args.explore_rate
        self.do_explore_in_train = args.do_explore_in_train

        self.check_episode = args.check_episode
        self.save_episode = args.save_episode
        self.save_path = args.save_path

        self.actor_lr = args.actor_lr
        self.actor_decay = args.actor_decay
        self.batch_size = args.batch_size

        # components
        self.env = env
        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor.set_init_var()
        self.actor_target.set_init_var()
        self.buffer = buffer

        # controller
        self.exploration_scheduler = utils.LinearScheduler(int(sum(args.n_iter) * args.elbow_epsilon),
                                                           args.final_epsilon,
                                                           initial_p=args.initial_epsilon)
        self.registered_models = []

        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f" ")

    def train(self):
        self.setup_monitors()
        self.env.reset()
        self.buffer.reset(self.env, self.actor)

        if len(self.n_iter) > 2:
            self.load()

        t = time.time()
        start_time = t

        # training
        print("Training:")
        step_offset = sum(self.n_iter[:-1])
        do_buffer_update = True
        observation = deepcopy(self.env.current_observation)
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1] // 1)):
            do_explore = np.random.random() < self.explore_rate if self.explore_rate < 1 else True
            # online inference
            observation = self.run_episode_step(i, self.exploration_scheduler.value(i), observation,
                                                do_buffer_update, do_explore)
            # online training
            if (i + 1) % self.train_every_n_step == 0:
                self.step_train()
                self.buffer.clear()
            # log monitor records
            if i > 0 and i % self.check_episode == 0:
                t_prime = time.time()
                print(f"Episode step {i}, time diff {t_prime - t}, total time diff {t - start_time})")
                episode_report, train_report = self.get_report(smoothness=self.check_episode)
                log_str = f"step: {i} @ online episode: {episode_report} @ training: {train_report}\n"
                with open(self.save_path + ".report", 'a') as outfile:
                    outfile.write(log_str)
                print(log_str)
                t = t_prime

            # save model and training info
            if i % self.save_episode == 0:
                self.save()

        self.action_after_train()

    def setup_monitors(self):
        self.training_history = {}
        self.eval_history = {'avg_reward': [],
                             'reward_variance': [],
                             'avg_total_reward': [0.],
                             'max_total_reward': [0.],
                             'min_total_reward': [0.],
                             'cof': [0],
                             }
        self.eval_history.update({f'{resp}_rate': [] for resp in self.env.response_types})
        self.current_sum_reward = torch.zeros(self.env.episode_batch_size).to(torch.float).to(self.device)
        self.current_step = torch.zeros(self.env.episode_batch_size).to(torch.long).to(self.device) 
        self.current_fair_weight = torch.zeros((self.env.episode_batch_size, 30)).to(torch.float).to(self.device)
        self.current_user_pop_prefer = torch.zeros((self.env.episode_batch_size, 30)).to(torch.float).to(self.device)
        self.user_pop_prefer = {}
        self.user_pop_ratio = {}
        self.fair_weight = {}
        for i in range(40):
            self.user_pop_prefer[i] = []
            self.user_pop_ratio[i] = []
            self.fair_weight[i] = []
        self.record_user_num = 0


    def action_after_train(self):
        self.env.stop()

    def get_report(self, smoothness=10):
        # 环境记录的数据
        episode_report = self.env.get_report(smoothness)
        # 智能体记录的数据
        train_report = {k: np.mean(v[-smoothness:]) for k, v in self.training_history.items()}
        train_report.update({k: np.mean(v[-smoothness:]) for k, v in self.eval_history.items()})
        return episode_report, train_report

    def run_episode_step(self, *episode_args):
        """
        Run one step of user-env interaction
        @input:
        - episode_args: (episode_iter, epsilon, observation, do_buffer_update, do_explore)
        @process:
        - apply_policy: observation, candidate items --> policy_output
        - env.step(): policy_output['action'] --> user_feedback, updated_observation
        - reward_func(): user_feedback --> reward
        - buffer.update(observation, policy_output, user_feedback, updated_observation)
        @output:
        - next_observation
        """
        episode_iter, epsilon, observation, do_buffer_update, do_explore = episode_args
        self.epsilon = epsilon
        is_train = False
        with torch.no_grad():
            # generate action from policy
            policy_output = self.apply_policy(observation, self.actor_target, epsilon, do_explore, is_train)
            user_pop_prefer = policy_output['user_pop_prefer']#.cpu().reshape(-1).numpy()#.sum()
            fair_weight = policy_output['fair_weight']#.cpu().reshape(-1).numpy()#.sum()
            unpopular_item_ratio = policy_output['unpopular_item_ratio']
            cof1 = np.corrcoef(user_pop_prefer.cpu().reshape(-1).numpy(), fair_weight.cpu().reshape(-1).numpy())
            cof2 = np.corrcoef(unpopular_item_ratio.cpu().reshape(-1).numpy(), fair_weight.cpu().reshape(-1).numpy())
            cof3 = np.corrcoef(unpopular_item_ratio.cpu().reshape(-1).numpy(), user_pop_prefer.cpu().reshape(-1).numpy())

            if episode_iter > 0:
                # print(unpopular_item_ratio)
                # print(fair_weight)
                print("-----------------------------------------------------")
                print(torch.mean(user_pop_prefer).cpu().reshape(-1).numpy(), 
                      torch.mean(unpopular_item_ratio).cpu().reshape(-1).numpy(), 
                      torch.mean(fair_weight).cpu().reshape(-1).numpy())
                print(cof1[0, 1], cof2[0, 1], cof3[0, 1])

            # apply action on environment
            # Note: action must be indices on env.candidate_iids
            action_dict = {'action': policy_output['indices']}
            new_observation, user_feedback, update_info, current_step = self.env.step(action_dict)

            # calculate reward
            R = self.get_reward(user_feedback)
            user_feedback['reward'] = R
            self.current_sum_reward = self.current_sum_reward + R
            done_mask = user_feedback['done']

            # # 统计不同时刻用户的流行度偏好
            if episode_iter > 0: #10000:
                current_step = current_step.cpu().numpy().reshape(-1)
                user_pop_prefer = user_pop_prefer.cpu().numpy().reshape(-1).tolist()
                popular_item_ratio = 1 - unpopular_item_ratio
                popular_item_ratio = popular_item_ratio.cpu().numpy().reshape(-1).tolist()
                fair_weight = fair_weight.cpu().numpy().reshape(-1).tolist()
                for i in range(len(current_step)):
                    self.user_pop_prefer[current_step[i]].append(user_pop_prefer[i])
                    self.user_pop_ratio[current_step[i]].append(popular_item_ratio[i])
                    self.fair_weight[current_step[i]].append(fair_weight[i])
                self.record_user_num += len(current_step)
                if self.record_user_num > 12800:
                    with open('./output/user_pop_prefer.txt', 'w') as file:
                        json.dump(self.user_pop_prefer, file, ensure_ascii=False)
                    with open('./output/user_pop_ratio.txt', 'w') as file:
                        json.dump(self.user_pop_ratio, file, ensure_ascii=False)
                    with open('./output/fair_weight.txt', 'w') as file:
                        json.dump(self.fair_weight, file, ensure_ascii=False)
                    self.record_user_num = 0
                    for i in range(40):
                        self.user_pop_prefer[i] = []
                        self.user_pop_ratio[i] = []
                        self.fair_weight[i] = []
            
            # #
            # self.current_user_pop_prefer[:, self.current_step] = user_pop_prefer
            # self.current_fair_weight[:, self.current_step] = fair_weight
            # self.current_step += 1

            # 对已退出的用户的历史数据进行统计
            if torch.sum(done_mask) > 0:
                self.eval_history['avg_total_reward'].append(self.current_sum_reward[done_mask].mean().item())
                self.eval_history['max_total_reward'].append(self.current_sum_reward[done_mask].max().item())
                self.eval_history['min_total_reward'].append(self.current_sum_reward[done_mask].min().item())
                self.current_sum_reward[done_mask] = 0

                # # 
                # temp_fair_weight = self.current_fair_weight[done_mask]
                # temp_user_pop_prefer = self.current_user_pop_prefer[done_mask]
                # temp_step = self.current_step[done_mask]
                # for i in torch.range(0, torch.sum(done_mask) - 1).long():
                #     #print(i, temp_step.shape, temp_user_pop_prefer.shape, torch.sum(done_mask))
                #     t_step = temp_step[i]
                #     t_fair_weight = temp_fair_weight[i, :t_step].cpu().reshape(-1).numpy()
                #     t_user_pop_prefer = temp_user_pop_prefer[i, :t_step].cpu().reshape(-1).numpy()
                # self.eval_history['cof'].append(np.corrcoef(t_fair_weight, t_user_pop_prefer)[0, 1])
                # self.current_step[done_mask] = 0

            # monitor update
            self.eval_history['avg_reward'].append(R.mean().item())
            self.eval_history['reward_variance'].append(torch.var(R).item())

            for i, resp in enumerate(self.env.response_types):
                self.eval_history[f'{resp}_rate'].append(user_feedback['immediate_response'][:, :, i].mean().item())
                # update replay buffer
            if do_buffer_update:
                self.buffer.update(observation, policy_output, user_feedback, update_info['updated_observation'])
        return new_observation

    def apply_policy(self, observation, actor, *policy_args):
        pass

    def get_reward(self, user_feedback):
        user_feedback['immediate_response_weight'] = self.env.response_weights
        R = self.reward_func(user_feedback).detach()
        return R

    def step_train(self):
        """
        @process:
        """
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)

        loss_dict = self.get_loss(observation, policy_output, user_feedback, done_mask, next_observation)

        for k in loss_dict:
            if k in self.training_history:
                try:
                    self.training_history[k].append(loss_dict[k].item())
                except:
                    self.training_history[k].append(loss_dict[k])

    def get_loss(self, observation, policy_output, user_feedback, done_mask, next_observation):
        pass

    def test(self):
        pass

    def save(self):
        for model, opt, prefix in self.registered_models:
            torch.save(model.state_dict(), self.save_path + prefix)
            torch.save(opt.state_dict(), self.save_path + prefix + "_optimizer")

    def load(self):
        for model, opt, prefix in self.registered_models:
            model.load_state_dict(torch.load(self.save_path + prefix, map_location=self.device))
            opt.load_state_dict(torch.load(self.save_path + prefix + "_optimizer", map_location=self.device))
