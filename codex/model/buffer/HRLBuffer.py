import torch
import torch.nn.functional as F
import random
import numpy as np

import utils
from model.buffer.BaseBuffer import BaseBuffer


class HRLBuffer(BaseBuffer):

    @staticmethod
    def parse_model_args(parser):
        """
        args:
        - from BaseBuffer:
            - buffer_size
        """
        parser = BaseBuffer.parse_model_args(parser)
        return parser

    def reset(self, *reset_args):
        """
        @output:
        - buffer: {'observation': {'user_profile': {'user_id': (L,),
                                                    'uf_{feature_name}': (L, feature_dim)},
                                   'user_history': {'history': (L, max_H),
                                                    'history_if_{feature_name}': (L, max_H * feature_dim),
                                                    'history_{response}': (L, max_H),
                                                    'history_length': (L,)}}
                   'policy_output': {'state': (L, state_dim),
                                     'action': (L, action_dim) = (L, slate_size),
                                     'hyper_action': (L, hyper_action_size)},
                   'next_observation': same format as @output-buffer['observation'],
                   'done_mask': (L,),
                   'response': {'reward': (L,),
                                'immediate_response':, (L, action_dim * response_dim)}}
        """
        env = reset_args[0]
        actor = reset_args[1]

        super().reset(env, actor)
        self.buffer['user_response']['immediate_response'] = torch.zeros(self.buffer_size,
                                                                         env.response_dim * actor.effect_action_dim) \
            .to(torch.float).to(self.device)
        self.buffer['policy_output']['h_state'] = torch.zeros(self.buffer_size, actor.state_dim) \
            .to(torch.float).to(self.device)
        self.buffer['policy_output']['l_state'] = torch.zeros(self.buffer_size, actor.state_dim) \
            .to(torch.float).to(self.device)
        self.buffer['policy_output']['h_action'] = torch.zeros(self.buffer_size, actor.h_hyper_action_dim) \
            .to(torch.float).to(self.device)
        self.buffer['policy_output']['l_action'] = torch.zeros(self.buffer_size, actor.effect_action_dim) \
            .to(torch.long).to(self.device)
        self.buffer['policy_output']['h_action_log_prob'] = torch.zeros(self.buffer_size, 1) \
            .to(torch.float).to(self.device)
        self.buffer['policy_output']['l_action_log_prob'] = torch.zeros(self.buffer_size, 1) \
            .to(torch.float).to(self.device)
        self.buffer['policy_output']['effect_action'] = torch.zeros(self.buffer_size, actor.effect_action_dim)\
                                                         .to(torch.long).to(self.device)
        self.buffer['policy_output']['unpopular_item_ratio'] = torch.zeros(self.buffer_size, 1).to(self.device)

        return self.buffer

    def clear(self):
        self.buffer_head = 0
        self.current_buffer_size = 0
        self.n_stream_record = 0

    def sample(self):
        """
        Sample all the data
        """
        # get indices
        indices = np.arange(self.current_buffer_size)
        # observation
        profile = {k: v[indices] for k, v in self.buffer["observation"]["user_profile"].items()}
        history = {k: v[indices] for k, v in self.buffer["observation"]["user_history"].items()}
        observation = {"user_profile": profile, "user_history": history}
        # next observation
        profile = {k: v[indices] for k, v in self.buffer["next_observation"]["user_profile"].items()}
        history = {k: v[indices] for k, v in self.buffer["next_observation"]["user_history"].items()}
        next_observation = {"user_profile": profile, "user_history": history}
        # policy output
        policy_output = {"h_state": self.buffer["policy_output"]["h_state"][indices],
                         "l_state": self.buffer["policy_output"]["l_state"][indices],
                         "h_action": self.buffer["policy_output"]["h_action"][indices],
                         "l_action": self.buffer["policy_output"]["l_action"][indices],
                         'effect_action': self.buffer["policy_output"]["effect_action"][indices],
                         "h_action_log_prob": self.buffer["policy_output"]["h_action_log_prob"][indices],
                         "l_action_log_prob": self.buffer["policy_output"]["l_action_log_prob"][ indices], 
                         "unpopular_item_ratio": self.buffer["policy_output"]["unpopular_item_ratio"][indices]}  # main change to BaseBuffer
        # user response
        user_response = {"reward": self.buffer["user_response"]["reward"][indices],
                         "immediate_response": self.buffer["user_response"]["immediate_response"][indices]}
        # done mask
        done_mask = self.buffer["done_mask"][indices]
        return observation, policy_output, user_response, done_mask, next_observation

    def update(self, observation, policy_output, user_feedback, next_observation):
        """
        @input:
        - observation: {'user_profile': {'user_id': (B,),
                                         'uf_{feature_name}': (B, feature_dim)},
                        'user_history': {'history': (B, max_H),
                                         'history_if_{feature_name}': (B, max_H * feature_dim),
                                         'history_{response}': (B, max_H),
                                         'history_length': (B,)}}
        - policy_output: {'user_state': (B, state_dim),
                          'h_action': (B, action_dim),
                          'l_action': (B, action_dim),
                          'h_log_prob': (B, 1),
                          'l_log_prob': (B, 1)}
        - user_feedback: {'done': (B,),
                          'immdiate_response':, (B, action_dim * feedback_dim),
                          'reward': (B,)}
        - next_observation: same format as update_buffer@input-observation
        """
        # get buffer indices to update
        B = len(user_feedback['reward'])
        if self.buffer_head + B >= self.buffer_size:
            tail = self.buffer_size - self.buffer_head
            indices = [self.buffer_head + i for i in range(tail)] + \
                      [i for i in range(B - tail)]
        else:
            indices = [self.buffer_head + i for i in range(B)]
        indices = torch.tensor(indices).to(torch.long).to(self.device)

        # update buffer - observation
        for k, v in observation['user_profile'].items():
            self.buffer['observation']['user_profile'][k][indices] = v
        for k, v in observation['user_history'].items():
            self.buffer['observation']['user_history'][k][indices] = v #.squeeze()
        # update buffer - next observation
        for k, v in next_observation['user_profile'].items():
            self.buffer['next_observation']['user_profile'][k][indices] = v
        for k, v in next_observation['user_history'].items():
            self.buffer['next_observation']['user_history'][k][indices] = v
        # update buffer - policy output
        # self.buffer['policy_output']['state'][indices] = policy_output['state']
        self.buffer['policy_output']['h_state'][indices] = policy_output['h_state']
        self.buffer['policy_output']['l_state'][indices] = policy_output['l_state']
        self.buffer['policy_output']['h_action'][indices] = policy_output['h_action']
        self.buffer['policy_output']['l_action'][indices] = policy_output['l_action']
        self.buffer['policy_output']['effect_action'][indices] = policy_output['indices']
        self.buffer['policy_output']['h_action_log_prob'][indices] = policy_output['h_action_log_prob'].view(B, -1)
        self.buffer['policy_output']['l_action_log_prob'][indices] = policy_output['l_action_log_prob'].view(B, -1)
        self.buffer['policy_output']['unpopular_item_ratio'][indices] = policy_output['unpopular_item_ratio'].view(B, -1)

        # update buffer - user response
        self.buffer['user_response']['immediate_response'][indices] = user_feedback['immediate_response'].view(B, -1)
        self.buffer['user_response']['reward'][indices] = user_feedback['reward']
        # update buffer - done
        self.buffer['done_mask'][indices] = user_feedback['done']

        # buffer pointer
        self.buffer_head = (self.buffer_head + B) % self.buffer_size
        self.n_stream_record += B
        self.current_buffer_size = min(self.n_stream_record, self.buffer_size)
