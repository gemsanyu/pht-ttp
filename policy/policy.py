from abc import abstractmethod
import math
from typing import List

import torch


CPU_DEVICE = torch.device("cpu")
# ES object
# generate parameters, map parameters, replace paramater of a model
# update parameters


class Policy(object):
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        # self.n_params = 2a+2(4a^2+2a) = 2a+6a^2+4a = 6a+6a^2
        self.n_params = 6*(num_neurons+num_neurons**2)

    def create_param_dict(self, param_vec):
        param_vec = param_vec.ravel()
        params_idx = 0
        v0 = param_vec[:self.num_neurons].view(1,1,self.num_neurons)
        params_idx += self.num_neurons
        v1 = param_vec[params_idx:params_idx+self.num_neurons].view(1,1,self.num_neurons)
        params_idx += self.num_neurons
        fe0_weight = param_vec[params_idx:params_idx+2*(self.num_neurons**2)].view(self.num_neurons, 2*self.num_neurons)
        params_idx += 2*(self.num_neurons**2)
        fe1_weight = param_vec[params_idx:params_idx+2*(self.num_neurons**2)].view(self.num_neurons, 2*self.num_neurons)
        params_idx += 2*(self.num_neurons**2)
        fe0_bias = param_vec[params_idx:params_idx+self.num_neurons].view(self.num_neurons)
        params_idx += self.num_neurons
        fe1_bias = param_vec[params_idx:params_idx+self.num_neurons].view(self.num_neurons)
        params_idx += self.num_neurons
        qe0_weight = param_vec[params_idx:params_idx+(self.num_neurons**2)].view(self.num_neurons, self.num_neurons)
        params_idx += (self.num_neurons**2)
        qe1_weight = param_vec[params_idx:params_idx+(self.num_neurons**2)].view(self.num_neurons, self.num_neurons)
        params_idx += (self.num_neurons**2)
        qe0_bias = param_vec[params_idx:params_idx+self.num_neurons].view(self.num_neurons)
        params_idx += self.num_neurons
        qe1_bias = param_vec[params_idx:params_idx+self.num_neurons].view(self.num_neurons)
        params_idx += self.num_neurons
                
        param_dict = {
                     "v0":v0,
                     "v1":v1,
                     "fe0_weight":fe0_weight,
                     "fe1_weight":fe1_weight,
                     "fe0_bias":fe0_bias,
                     "fe1_bias":fe1_bias,
                     "qe0_weight":qe0_weight,
                     "qe1_weight":qe1_weight,
                     "qe0_bias":qe0_bias,
                     "qe1_bias":qe1_bias,
                     }      
        return param_dict

    @abstractmethod
    def generate_random_parameters(self, n_sample):
        pass

    @abstractmethod
    def generate_on_mean(self):
        pass

    @abstractmethod
    def logprob(self, sample_list):
        pass

    @abstractmethod
    def set_new_device(self, new_device, actor_device=None):
        pass


def get_multi_importance_weight(policy_list, sample_list):
    num_sample, _ = sample_list.shape
    num_policy = len(policy_list)
    logprobs = torch.zeros((num_policy, num_sample),
                           dtype=torch.float32)
    for i in range(len(policy_list)):
        logprobs[i, :] = policy_list[i].logprob(sample_list)

    weights = torch.softmax(logprobs, dim=0)
    # get the first weights, because the first weights is the current policy
    weights = weights[0, :]
    weights *= num_policy
    return weights.unsqueeze(1)
