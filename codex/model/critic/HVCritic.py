import torch.nn.functional as F
import torch.nn as nn
import torch

from model.components import DNN
from utils import get_regularization


class HVCritic(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        """
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        """
        parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[128],
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--critic_dropout_rate', type=float, default=0.2,
                            help='dropout rate in deep layers')
        return parser

    def __init__(self, args, environment, policy):
        super().__init__()
        self.state_dim = policy.state_dim
        self.h_action_dim = policy.h_action_dim
        self.l_action_dim = policy.l_action_dim
        #         self.state_encoder = policy.state_encoder
        self.h_net = DNN(self.state_dim, args.critic_hidden_dims, 1,
                         dropout_rate=args.critic_dropout_rate, do_batch_norm=True)
        self.l_net = DNN(self.state_dim+2, args.critic_hidden_dims, 1,
                         dropout_rate=args.critic_dropout_rate, do_batch_norm=True)

    def forward(self, feed_dict):
        """
        @input:
        - feed_dict: {'state_emb': (B, state_dim), 'action_emb': (B, action_dim)}
        """
        h_state_emb = feed_dict['h_state']
        l_state_emb = feed_dict['l_state']
        user_pop_prefer = feed_dict['user_pop_prefer'].reshape(-1, 1)
        h_action =feed_dict ['h_action'].detach()
        #H_V = self.h_net(h_state_emb).view(-1)
        t_h_action = nn.functional.relu(h_action)
        #L_V = self.l_net(torch.cat([l_state_emb, t_h_action], dim=1)).view(-1)
        H_V = self.h_net(torch.cat([h_state_emb], dim=1)).view(-1)
        L_V = self.l_net(torch.cat([l_state_emb, t_h_action], dim=1)).view(-1)
        h_reg = get_regularization(self.h_net)
        l_reg = get_regularization(self.l_net)
        return {'h_v': H_V, 'h_reg': h_reg, 'l_v': L_V, 'l_reg': l_reg}
