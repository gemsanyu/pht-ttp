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
        self.n_params = 0
        self.n_params += self.num_neurons
        self.n_params += 2*(self.num_neurons**2)
        self.n_params += self.num_neurons**2

    def create_param_dict(self, param_vec):
        param_vec = param_vec.ravel()
        params_idx = 0
        v1 = param_vec[params_idx:params_idx+self.num_neurons].view(1,1,self.num_neurons)
        params_idx += self.num_neurons
        fe1_weight = param_vec[params_idx:params_idx+2*(self.num_neurons**2)].view(self.num_neurons, 2*self.num_neurons)
        params_idx += 2*(self.num_neurons**2)
        qe1_weight = param_vec[params_idx:params_idx+(self.num_neurons**2)].view(self.num_neurons, self.num_neurons)
        params_idx += (self.num_neurons**2)
        param_dict = {
                     "v1":v1,
                     "fe1_weight":fe1_weight,
                     "qe1_weight":qe1_weight,
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

