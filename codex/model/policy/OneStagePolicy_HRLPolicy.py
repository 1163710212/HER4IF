import torch

from model.policy.OneStagePolicy import OneStagePolicy
from model.components import DNN
from model.score_func import *
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions import Multinomial
import torch.nn as nn
import pandas as pd


class OneStagePolicy_HRLPolicy(OneStagePolicy):
    @staticmethod
    def parse_model_args(parser):
        """
        args:
        - from OneStagePolicy:
            - state_encoder_feature_dim
            - state_encoder_attn_n_head
            - state_encoder_hidden_dims
            - state_encoder_dropout_rate
        """
        parser = OneStagePolicy.parse_model_args(parser)
        parser.add_argument('--policy_action_hidden', type=int, nargs='+', default=[128],
                            help='hidden dim of the action net')
        parser.add_argument('--action_std_init', type=float, default=0.6,
                            help='init action std')
        parser.add_argument('--has_continuous_action_space', type=bool, default=True,
                            help='action type')
        return parser

    def __init__(self, args, environment):
        """
        action_space = {'item_id': ('nominal', stats['n_item']),
                        'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
        observation_space = {'user_profile': ('continuous', stats['user_portrait_len'], 'positive']),
                            'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}
        """
        super().__init__(args, environment)
        # action is the set of parameters of linear mapping [item_dim, 1]
        self.h_hyper_action_dim = 2
        self.l_hyper_action_dim = self.enc_dim + 1
        self.h_action_dim = self.h_hyper_action_dim
        self.l_action_dim = self.l_hyper_action_dim
        self.effect_action_dim = self.slate_size

        self.has_continuous_action_space = args.has_continuous_action_space
        self.h_action_layer = nn.Sequential(DNN(self.state_dim, args.policy_action_hidden, self.h_action_dim,
                                                dropout_rate=self.dropout_rate, do_batch_norm=True))#, nn.Softmax(dim=-1))
        self.l_action_layer = DNN(self.state_dim, args.policy_action_hidden, self.l_action_dim,
                                  dropout_rate=self.dropout_rate, do_batch_norm=True)
        self.h_action_std_init = args.action_std_init
        self.l_action_std_init = args.action_std_init

        # 读取用户流行度偏好、物品类型
        path = f"/home/ucas2/xcj/KuaiSim-V1/code/dataset/Kuairand_Pure/"
        self.user_pop_ratios = torch.tensor(pd.read_csv(path + 'user_pop_ratio.csv').to_numpy()).to(self.device)
        item_types = torch.tensor(pd.read_csv(path + 'item_types.csv').to_numpy()).to(self.device)
        self.item_types = item_types.reshape(1, -1).repeat(128, 1)

    def set_init_var(self):
        h_action_var = torch.full((self.h_action_dim,), self.h_action_std_init * self.h_action_std_init,
                                  requires_grad=True).to(self.device)
        self.h_action_var = nn.Parameter(h_action_var)
        l_action_var = torch.full((self.l_action_dim,), self.l_action_std_init * self.l_action_std_init,
                                  requires_grad=True).to(self.device)
        self.l_action_var = nn.Parameter(l_action_var)

    def generate_action(self, state_dict, feed_dict):
        """
        List generation provides three main types of exploration:
        * Greedy top-K: no exploration, set do_effect_action_explore=False in args or do_explore=False in feed_dict
        * Categorical sampling: probabilistic exploration, set do_effect_action_explore=True in args,
                                set do_explore=True and epsilon < 1 in feed_dict
        * Uniform sampling : random exploration, set do_effect_action_explore=True in args,
                             set do_explore=True, epsilon > 0 in feed_dict
        * Gaussian sampling on hyper-action: set do_explore=True, epsilon < 1 in feed_dict
        * Uniform sampling on hyper-action: set do_explore=True, epsilon > 0 in feed_dict

        @input:
        - state_dict: {'state': (B, state_dim), ...}
        - feed_dict: same as OneStagePolicy.get_forward@input - feed_dict
        @output:
        - out_dict: {'action': (B, K),
                     'action_log_prb': (B, K),
                     'action_prob': (B, K),
                     'reg': scalar}
        """
        state = state_dict['state']
        candidates = feed_dict['candidates']
        epsilon = feed_dict['epsilon']
        do_explore = feed_dict['do_explore']
        is_train = feed_dict['is_train']
        batch_wise = feed_dict['batch_wise']
        B = state.shape[0]

        # 上层智能体动作
        h_action_mean = self.h_action_layer(state)
        h_cov_mat = torch.diag(self.h_action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(h_action_mean, h_cov_mat)
        h_hyper_action = dist.sample()
        h_action_log_prob = dist.log_prob(h_hyper_action)

        # 下层智能体动作
        l_action_mean = self.l_action_layer(state)
        l_cov_mat = torch.diag(self.l_action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(l_action_mean, l_cov_mat)
        l_hyper_action = dist.sample()
        l_action_log_prob = dist.log_prob(l_hyper_action)

        # 生成具体推荐物品
        candidate_item_enc, reg = self.user_encoder.get_item_encoding(candidates['item_id'],
                                                                      {k[3:]: v for k, v in candidates.items() if
                                                                       k != 'item_id'},
                                                                      B if batch_wise else 1)
        # (B, L)
        scores = self.get_score(l_hyper_action, candidate_item_enc, self.enc_dim)
        # 上层权重分别作用在两组物品上
        # print(scores.shape, self.item_types[:B].shape, h_hyper_action[:, 0].shape)
        scores = scores * self.item_types[:B] * h_hyper_action[:, 0].reshape(-1, 1) \
                 + scores * (1 - self.item_types[:B]) * h_hyper_action[:, 1].reshape(-1, 1)

        # top-k selection
        _, indices = torch.topk(scores, k=self.slate_size, dim=1)

        reg += self.get_regularization(self.l_action_layer) + self.get_regularization(self.l_action_layer)
        out_dict = {'h_action': h_hyper_action,
                    'l_action': l_hyper_action,
                    'h_action_log_prob': h_action_log_prob,
                    'l_action_log_prob': l_action_log_prob,
                    'indices': indices,
                    'reg': reg}
        return out_dict

    def get_score(self, hyper_action, candidate_item_enc, item_dim):
        """
        Deterministic mapping from hyper-action to effect-action (rec list)
        """
        # (B, L)
        scores = linear_scorer(hyper_action, candidate_item_enc, item_dim)
        return scores

    # 训练阶段使用
    def evaluate(self, feed_dict):
        state = feed_dict['state'].view(-1, self.state_dim)
        h_action = feed_dict['h_action'].view(-1, self.h_action_dim)
        l_action = feed_dict['l_action'].view(-1, self.l_action_dim)

        h_action_mean = self.h_action_layer(state)
        h_cov_mat = torch.diag(self.h_action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(h_action_mean, h_cov_mat)
        h_action_log_prob = dist.log_prob(h_action)
        h_dist_entropy = dist.entropy()

        l_action_mean = self.l_action_layer(state)
        l_cov_mat = torch.diag(self.l_action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(l_action_mean, l_cov_mat)
        l_action_log_prob = dist.log_prob(l_action)
        l_dist_entropy = dist.entropy()

        return h_action_log_prob, h_dist_entropy, l_action_log_prob, l_dist_entropy

    def forward(self, feed_dict: dict, return_prob=True):
        out_dict = self.get_forward(feed_dict)
        return out_dict