import math

import torch
import matplotlib.pyplot as plt

from agent.agent import Agent
from policy.policy import Policy, get_multi_importance_weight
from policy.utils import get_hv_contributions, nondominated_sort, simcos, get_hypervolume, combine_with_nondom
from policy.utils import get_score_hv_contributions, get_score_nsga2

CPU_DEVICE = torch.device("cpu")
# ES object
# generate parameters, map parameters, replace paramater of a model
# update parameters


class SNES(Policy):
    def __init__(self,
                 num_neurons,
                 num_dynamic_features):
        super(SNES, self).__init__(num_neurons, num_dynamic_features)

        self.norm_dist = torch.distributions.Normal(0,1)
        self.mu = torch.randn(size=(1, self.n_params), dtype=torch.float32)
        self.sigma = torch.full(size=(1, self.n_params), fill_value=math.exp(-3), dtype=torch.float32)

        # hyperparams
        self.negative_hv = -1e-4
        self.lr_mu = 1
        self.lr_sigma = 0.6 * (3 + math.log(self.n_params)) / 3 / math.sqrt(self.n_params) #pybrain
        # self.lr_sigma = (3+math.log(self.n_params))/(5*math.sqrt(self.n_params))
        self.batch_size = 4 + int(math.floor(3 * math.log(self.n_params)))    

    def copy_to_mu(self, agent: Agent):
        for name, param in agent.named_parameters():
            if name == "project_embeddings.weight":
                pe_weight = param.data.ravel()
            if name == "project_current_state.weight":
                pcs_weight = param.data.ravel()
            if name == "project_node_state.weight":
                pns_weight = param.data.ravel()
            if name == "project_out.weight":
                po_weight = param.data.ravel()
        self.mu = torch.cat([pe_weight,pcs_weight,pns_weight,po_weight])
        self.mu = self.mu.unsqueeze(0)

    # def get_max_variance(self):
    #     return torch.exp(self.ld*2/self.n_params)

    '''
    y ~ N(0,I)
    k ~ N(0,1)
    theta = mu + s*sigma
    return theta mapped with param names of the policy
    '''
    def generate_random_parameters(self, n_sample: int = 2, use_antithetic=True):
        if n_sample > 1:
            if use_antithetic:
                s_list = self.norm_dist.sample((int(n_sample/2), self.n_params))
                s_list = torch.cat((s_list,-s_list), dim=0)
            else:
                s_list = self.norm_dist.sample((n_sample, self.n_params))
        else:
            s_list = self.norm_dist.sample((1, self.n_params))

        random_params = self.mu + self.sigma*s_list

        param_dict_list = []
        for param_vec in random_params:
            param_dict_list += [self.create_param_dict(param_vec)]
        return param_dict_list, random_params

    '''
    theta = mu
    return theta mapped with param names of the policy
    '''

    def generate_on_mean(self):
        param_dict = self.create_param_dict(self.mu)
        return param_dict

    # update given the values
    def update(self, s_list, f_list, weight=None, reference_point=None, nondom_archive=None):
        score = get_score_hv_contributions(f_list, self.negative_hv, nondom_archive, reference_point)
        # score = get_score_nsga2(f_list, nondom_archive, reference_point)
        if weight is None:
            weight = 1
    
        ngrad_mu_j = torch.sum(score*s_list, dim=0)
        ngrad_sigma_j = torch.sum(score*(s_list*s_list-1), dim=0)

        self.mu = self.mu + self.lr_mu*self.sigma*ngrad_mu_j
        self.sigma = self.sigma*torch.exp(self.lr_sigma*ngrad_sigma_j/2)

    # @property
    # def _getMaxVariance(self):
    #     return math.exp(self.ld * 2 / self.n_params)

    def logprob(self, sample_list):
        s_list = sample_list
        logprob = -s_list*s_list/2 - torch.log(self.sigma*math.sqrt(2*math.pi))
        logprob = torch.sum(logprob, dim=1)
        return logprob

    # update with experience replay
    def update_with_er(self, er, reference_point=None, nondom_archive=None):
        num_samples = er.num_sample*er.num_saved_policy
        s_list = er.s_list[:num_samples, :]
        f_list = er.f_list[:num_samples, :]
        weight = get_multi_importance_weight(er.policy_list[:er.num_saved_policy], s_list)
        self.update(s_list, f_list, weight, reference_point=reference_point, nondom_archive=nondom_archive)
    
    def write_progress_to_tb(self, writer, step):
        # note the parameters
        writer.add_scalar("Mu Norm", torch.norm(self.mu).cpu().item(), step)
        writer.add_scalar("Sigma Norm", torch.norm(self.sigma).cpu().item(), step)
        writer.add_scalar("Max Sigma", torch.max(abs(self.sigma)), step)    
        # writer.add_scalar("Lambda", self.ld.item(), step)
        writer.flush()

'''
Lets separate experience replay for each policy, so that we can save whatever needed
without preprocessing in every update step to improve runtime.
'''
class ExperienceReplay(object):
  def __init__(self, dim, num_obj=2, max_saved_policy=5, num_sample=10):
    super().__init__()
    self.dim = dim
    self.max_saved_policy = max_saved_policy
    self.num_sample = num_sample
    self.num_obj = num_obj
  
    self.num_saved_policy = 0
    self.policy_list = []

    self.s_list = torch.zeros((self.max_saved_policy*num_sample, dim), dtype=torch.float32)
    self.f_list = torch.zeros((self.max_saved_policy*num_sample, num_obj), dtype=torch.float32)

  def clear(self):
    self.num_saved_policy = 0
  
  def add(self, policy:SNES, sample_list, f_list):
    self.num_saved_policy = min(self.num_saved_policy+1, self.max_saved_policy)

    # push to front, while rolling to remove overflowing samples
    self.policy_list = [policy] + self.policy_list[0:self.max_saved_policy-1]

    # sample = mu + sigma*s
    s_list = (sample_list - policy.mu)/policy.sigma
    self.s_list = self.s_list.roll(self.num_sample, dims=0)
    self.s_list[:self.num_sample,:] = s_list

    self.f_list = self.f_list.roll(self.num_sample, dims=0)
    self.f_list[:self.num_sample] = f_list
