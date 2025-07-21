import torch

from model.policy.OneStagePolicy import OneStagePolicy
from model.components import DNN
from model.score_func import *
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions import Multinomial
import torch.nn as nn


class OneStagePolicy_Dqn(OneStagePolicy):
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
        self.action_num = self.item_num
        self.effect_action_dim = self.slate_size
        self.q_layer = DNN(self.state_dim, args.policy_action_hidden, self.action_num,
                           dropout_rate=self.dropout_rate, do_batch_norm=True)

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

        # (B, L)
        scores = self.q_layer(state)
        # top-k selection
        _, indices = torch.topk(scores, k=self.slate_size, dim=1)
        # 是否进行探索
        if do_explore:
            is_rand_choose = (torch.rand(indices.shape) < epsilon).to(self.device)
            indices_rand = torch.randint(0, self.item_num, indices.shape).to(self.device)
            indices = indices * (~is_rand_choose) + indices_rand * is_rand_choose

        reg = self.get_regularization(self.q_layer)
        out_dict = {'action': indices,
                    'indices': indices,
                    'reg': reg}
        return out_dict

    # 训练阶段使用
    def evaluate(self, feed_dict):
        state = feed_dict['state'].view(-1, self.state_dim)
        # (B, L)
        action = feed_dict['action'].view(-1, self.slate_size)
        # (B, item_num) -> (B, L) -> (B, 1)
        q_value = (torch.gather(self.q_layer(state), 1, action)).sum(dim=1)
        return q_value

    def forward(self, feed_dict: dict, return_prob=True):
        out_dict = self.get_forward(feed_dict)
        return out_dict
