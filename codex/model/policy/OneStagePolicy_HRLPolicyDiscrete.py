import torch

from model.policy.OneStagePolicy import OneStagePolicy
from model.components import DNN
from model.score_func import *
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions import Multinomial
import torch.nn as nn
import pandas as pd
from model.policy.BackboneUserEncoder import BackboneUserEncoder


class OneStagePolicy_HRLPolicyDiscrete(OneStagePolicy):
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
        parser.add_argument('--action_std_init', type=float, default=0.1,
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
        self.l_hyper_action_dim = self.item_num
        self.h_action_dim = self.h_hyper_action_dim
        self.l_action_dim = self.l_hyper_action_dim
        self.effect_action_dim = self.slate_size

        self.has_continuous_action_space = args.has_continuous_action_space
        self.h_action_layer = nn.Sequential(DNN(self.state_dim + 0, args.policy_action_hidden, self.h_action_dim,
                                                dropout_rate=self.dropout_rate, do_batch_norm=True), nn.Softmax(dim=-1))
        self.l_action_layer = nn.Sequential(DNN(self.state_dim + 2, args.policy_action_hidden, self.l_action_dim,
                                                dropout_rate=self.dropout_rate, do_batch_norm=True), nn.Softmax(dim=-1))
        self.h_action_std_init = args.action_std_init
        self.l_action_std_init = args.action_std_init

        # 读取用户流行度偏好、物品类型
        path = f"/home/ucas2/xcj/KuaiSim-V1/code/dataset/{args.dataset}/"
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

    def generate_action(self, h_state_dict, l_state_dict, feed_dict):
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
        h_state = h_state_dict['state']
        l_state = l_state_dict['state']
        user_pop_prefer = feed_dict['observation']['user_history']['user_pop_prefer']
        candidates = feed_dict['candidates']
        epsilon = feed_dict['epsilon']
        do_explore = feed_dict['do_explore']
        is_train = feed_dict['is_train']
        batch_wise = feed_dict['batch_wise']
        B = l_state.shape[0]

        # 上层智能体动作
        h_action_mean = self.h_action_layer(h_state)
        h_cov_mat = torch.diag(self.h_action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(h_action_mean, h_cov_mat)
        h_action = dist.sample()
        h_action_log_prob = dist.log_prob(h_action)

        # 下层智能体动作
        l_state = torch.cat((l_state, h_action), dim=-1)
        #l_state = torch.cat((l_state, h_action), dim=-1)
        l_action_mean = self.l_action_layer(l_state)
        # 上层权重分别作用在两组物品上
        t_h_action = nn.functional.relu(h_action)
        l_action_mean = l_action_mean * self.item_types[:B] * t_h_action[:, 0].reshape(-1, 1) \
                 + l_action_mean * (1 - self.item_types[:B]) * t_h_action[:, 1].reshape(-1, 1)
        l_action_mean = nn.functional.relu(l_action_mean)
        l_action_mean = (l_action_mean / torch.sum(l_action_mean, dim=-1).reshape(-1, 1))
        dist = Categorical(l_action_mean)
        l_action = torch.multinomial(l_action_mean, num_samples=self.effect_action_dim, replacement=False)
        l_action_log_prob = torch.mean(dist.log_prob(l_action.transpose(1, 0)).transpose(1, 0), dim=1).squeeze()

        reg = self.get_regularization(self.h_action_layer) + self.get_regularization(self.l_action_layer)
        #unpopular_item_ratio = (l_action_mean * (1 - self.item_types[:B])).sum(dim=1).squeeze()
        # 获取实际长尾物品的曝光比例
        item_types = 1 - self.item_types[0].reshape(-1)
        unpopular_item_ratio = item_types[l_action.reshape(-1)].reshape(l_action.shape[0], -1).float().mean(dim=1)
        fair_weight = t_h_action[:, 1] #/ (t_h_action[:, 0] + 1e-5)
        out_dict = {'h_action': h_action,
                    'l_action': l_action,
                    'h_action_log_prob': h_action_log_prob,
                    'l_action_log_prob': l_action_log_prob,
                    'indices': l_action,
                    'user_pop_prefer': user_pop_prefer,
                    'unpopular_item_ratio': unpopular_item_ratio,
                    'fair_weight': fair_weight,
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
    def evaluate(self, feed_dict, neg_feed_dict,user_pop_prefer):
        h_state = feed_dict['h_state'].view(-1, self.state_dim)
        neg_h_state = neg_feed_dict['h_state'].view(-1, self.state_dim)
        l_state = feed_dict['l_state'].view(-1, self.state_dim)
        neg_l_state = neg_feed_dict['l_state'].view(-1, self.state_dim)

        B = h_state.shape[0]

        h_action = feed_dict['h_action'].view(-1, self.h_action_dim)
        l_action = feed_dict['l_action'].view(-1, self.effect_action_dim)

        h_action_mean = self.h_action_layer(h_state)
        
        h_cov_mat = torch.diag(self.h_action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(h_action_mean, h_cov_mat)
        h_action_log_prob = dist.log_prob(h_action)
        h_dist_entropy = dist.entropy()
        
        t_h_action = nn.functional.relu(h_action)
        l_state = torch.cat((l_state, t_h_action.detach()), dim=-1)
        #l_state = torch.cat((l_state,  t_h_action.detach()), dim=-1)
        l_action_mean = self.l_action_layer(l_state)
        # 上层权重分别作用在两组物品上
        l_action_mean = l_action_mean * self.item_types[:B] * t_h_action[:, 0].reshape(-1, 1) \
                 + l_action_mean * (1 - self.item_types[:B]) * t_h_action[:, 1].reshape(-1, 1)
        l_action_mean = nn.functional.relu(l_action_mean)
        dist = Categorical(l_action_mean)
        l_action_log_prob = torch.mean(dist.log_prob(l_action.transpose(1, 0)).transpose(1, 0), dim=1).squeeze()
        l_dist_entropy = dist.entropy()
        
        
        # 状态增强
        aug_time = 10
        # 上层智能体损失
        h_action_aug_mean = None
        neg_h_action_mean = self.h_action_layer(neg_h_state).detach()
        for _ in range(aug_time):
            state_arg = h_state + torch.randn_like(h_state) * 0.001
            action_mean = self.h_action_layer(state_arg).detach()
            if h_action_aug_mean == None:
                h_action_aug_mean = action_mean
            else:
                h_action_aug_mean += action_mean
        h_action_aug_mean /= aug_time
        #print("mush shape")
        #print(neg_h_action_mean.shape, h_action_aug_mean.shape, l_action_mean.shape)
        aug_loss = -F.logsigmoid((neg_h_action_mean - h_action_mean) ** 2 - (h_action_mean - h_action_aug_mean) ** 2).mean()

        # 下层智能体损失
        l_action_aug_mean = None
        neg_l_state = torch.cat((neg_l_state, neg_h_action_mean.detach()), dim=-1)
        neg_l_action_mean = nn.functional.relu(self.l_action_layer(neg_l_state).detach())
        l_action_mean = nn.functional.relu(self.l_action_layer(l_state))
        for _ in range(aug_time):
            state_arg = l_state + torch.randn_like(l_state) * 0.001
            action_mean = self.l_action_layer(state_arg).detach()
            if l_action_aug_mean == None:
                l_action_aug_mean = action_mean
            else:
                l_action_aug_mean += action_mean
        l_action_aug_mean = nn.functional.relu(l_action_aug_mean)
        l_action_aug_mean /= aug_time
        aug_loss += -F.logsigmoid(F.kl_div(l_action_mean.log(), neg_l_action_mean) - F.kl_div(l_action_mean.log(), l_action_aug_mean)).mean()


        return h_action_log_prob, h_dist_entropy, l_action_log_prob, l_dist_entropy, aug_loss

    def forward(self, feed_dict: dict, return_prob=True):
        out_dict = self.get_forward(feed_dict)
        return out_dict
    
    def get_forward(self, feed_dict: dict):
        '''
        @input:
        - feed_dict: {
            'observation':{
                'user_profile':{
                    'user_id': (B,)
                    'uf_{feature_name}': (B,feature_dim), the user features}
                'user_history':{
                    'history': (B,max_H)
                    'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}
            'candidates':{
                'item_id': (B,L) or (1,L), the target item
                'item_{feature_name}': (B,L,feature_dim) or (1,L,feature_dim), the target item features}
            'epsilon': scalar, 
            'do_explore': boolean,
            'candidates': {
                'item_id': (B,L) or (1,L), the target item
                'item_{feature_name}': (B,L,feature_dim) or (1,L,feature_dim), the target item features},
            'action_dim': slate size K,
            'action': (B,K),
            'response': {
                'reward': (B,),
                'immediate_response': (B,K*n_feedback)},
            'is_train': boolean,
            'batch_wise': boolean
        }
        @output:
        - out_dict: {
            'state': (B,state_dim), 
            'prob': (B,K),
            'all_prob': (B,L),
            'action': (B,K),
            'reg': scalar}
        '''
        observation = feed_dict['observation']
        # observation --> user state
        h_state_dict, l_state_dict = self.get_user_state(observation)
        # user state + candidates --> dict(state, prob, action, reg)
        out_dict = self.generate_action(h_state_dict, l_state_dict, feed_dict)
        out_dict['h_state'] = h_state_dict['state']
        out_dict['l_state'] = l_state_dict['state']
        out_dict['reg'] = h_state_dict['reg'] + l_state_dict['reg'] + out_dict['reg']
        return out_dict

    def get_user_state(self, observation):
        feed_dict = {}
        feed_dict.update(observation['user_profile'])
        feed_dict.update(observation['user_history'])
        return self.h_user_encoder.get_forward(feed_dict), self.l_user_encoder.get_forward(feed_dict)
    
    def _define_params(self, args, reader_stats):
        args.state_dropout_rate = 0
        # self.h_user_encoder = BackboneUserEncoder(args, reader_stats)
        # self.l_user_encoder = BackboneUserEncoder(args, reader_stats)
        self.user_encoder = BackboneUserEncoder(args, reader_stats)
        self.h_user_encoder = BackboneUserEncoder(args, reader_stats, has_up=True)
        self.l_user_encoder = BackboneUserEncoder(args, reader_stats, has_up=True)
        self.enc_dim = self.h_user_encoder.enc_dim
        self.state_dim = self.h_user_encoder.state_dim
        self.action_dim = self.slate_size
        self.item_num = reader_stats['n_item']
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction = 'none')