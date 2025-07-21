import torch.nn.functional as F
import torch.nn as nn
import torch

from model.components import DNN
from utils import get_regularization

class TwinnedQCritic(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        '''
        parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[128], 
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--critic_dropout_rate', type=float, default=0.1, 
                            help='dropout rate in deep layers')
        return parser
    
    def __init__(self, args, environment, policy):
        super().__init__()
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim
        self.net1 = DNN(self.state_dim, args.critic_hidden_dims, self.action_dim, 
                       dropout_rate = args.critic_dropout_rate, do_batch_norm = True)
        self.net2 = DNN(self.state_dim, args.critic_hidden_dims, self.action_dim, 
                       dropout_rate = args.critic_dropout_rate, do_batch_norm = True)
        
    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'state': (B, state_dim), 'action': (B, action_dim)}
        '''
        state_emb = feed_dict['state'].view(-1, self.state_dim)
        Q1 = self.net1(state_emb)
        Q2 = self.net2(state_emb)
        Q = torch.min(Q1, Q2)
        reg = get_regularization(self.net1) + get_regularization(self.net2)
        return {'q': Q, 'q1': Q1, 'q2': Q2, 'reg': reg}