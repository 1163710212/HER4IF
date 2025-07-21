import torch

from model.policy.OneStagePolicy import OneStagePolicy
from model.components import DNN
from model.score_func import *
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions import Multinomial
import torch.nn as nn


class OneStagePolicy_OnPolicy(OneStagePolicy):
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
        parser.add_argument('--action_std_init', type=float, default=0.8,
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
        self.hyper_action_dim = self.enc_dim + 1
        self.action_dim = self.hyper_action_dim
        self.effect_action_dim = self.slate_size
        self.has_continuous_action_space = args.has_continuous_action_space
        self.action_layer = DNN(self.state_dim, args.policy_action_hidden, self.action_dim,
                                dropout_rate=self.dropout_rate, do_batch_norm=True)
        self.action_std_init = args.action_std_init

    def set_init_var(self):
        action_var = torch.full((self.action_dim,), self.action_std_init * self.action_std_init).to(self.device)
        #self.action_var = nn.Parameter(action_var)
        self.cov_mat = nn.Parameter(torch.diag(action_var))

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

        action_mean = self.action_layer(state)
        #cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, torch.abs(self.cov_mat) + 1e-4)

        hyper_action = dist.sample()
        action_log_prob = dist.log_prob(hyper_action)

        candidate_item_enc, reg = self.user_encoder.get_item_encoding(candidates['item_id'],
                                                                      {k[3:]: v for k, v in candidates.items() if
                                                                       k != 'item_id'},
                                                                      B if batch_wise else 1)
        # (B, L)
        scores = self.get_score(hyper_action, candidate_item_enc, self.enc_dim)
        # top-k selection
        _, indices = torch.topk(scores, k=self.slate_size, dim=1)
        if batch_wise:
            action = torch.gather(candidates['item_id'], 1, indices).detach()  # (B, slate_size)
        else:
            action = candidates['item_id'][indices].detach()  # (B, slate_size)

        reg += self.get_regularization(self.action_layer)
        out_dict = {'action': hyper_action,
                    'action_log_prob': action_log_prob,
                    'indices': indices,
                    'reg': reg}
        return out_dict
    
    def get_score(self, hyper_action, candidate_item_enc, item_dim):
        '''
        Deterministic mapping from hyper-action to effect-action (rec list)
        '''
        # (B, L)
        scores = linear_scorer(hyper_action, candidate_item_enc, item_dim)
        return scores

    # 训练阶段使用
    def evaluate(self, feed_dict):
        state = feed_dict['state'].view(-1, self.state_dim)
        action = feed_dict['action'].view(-1, self.action_dim)
        if self.has_continuous_action_space:
            action_mean = self.action_layer(state)
            #action_var = self.action_var.expand_as(action_mean)
            #cov_mat = torch.diag_embed(action_var).to(self.device) + 1e-5
            try:
                dist = MultivariateNormal(action_mean, torch.abs(self.cov_mat) + 1e-4)
            except:
                print(nn.functional.relu(self.cov_mat) + 1e-6)
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_value = self.action_layer(state)
            # constraints[:] = 1
            # batch_size = constraints.shape[0]
            # action_value *= self.pop_index[:batch_size] * constraints \
            #                 + (1 - self.pop_index[:batch_size]) * constraints
            dist = Multinomial(self.slate_size, action_prob)
        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_log_prob, dist_entropy

    def forward(self, feed_dict: dict, return_prob=True):
        out_dict = self.get_forward(feed_dict)
        return out_dict
