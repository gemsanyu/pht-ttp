import math

import torch
import matplotlib.pyplot as plt

from policy.policy import Policy, get_multi_importance_weight
from policy.utils import get_hypervolume, get_utility, nondominated_sort, get_nondominated_rank
from policy.utils import get_bc_var

CPU_DEVICE = torch.device("cpu")
# ES object
# generate parameters, map parameters, replace paramater of a model
# update parameters


class R1_NES_DOM(Policy):
    def __init__(self,
                 dummy_model: torch.nn.Module,
                 device=CPU_DEVICE,
                 actor_device=CPU_DEVICE):
        super(R1_NES_DOM, self).__init__(dummy_model, device, actor_device)

        self.norm_dist = torch.distributions.Normal(0, 1)
        self.mu = torch.randn(size=(1, self.n_params),
                              dtype=torch.float32, device=self.device)*0.01
        self.ld = torch.rand(size=(1,), dtype=torch.float32, device=self.device)
            # torch.rand(
            # size=(1,), dtype=torch.float32, device=self.device)*2
        # reparametrize self.v = e^c *self.z
        # c is the length of v
        # self.z must be ||z|| = 1
        self.c = torch.rand(size=(1,), dtype=torch.float32,
                            device=self.device)
        self.z = torch.randn(size=(self.n_params,),
                             dtype=torch.float32, device=self.device)
        self.z = self.z/torch.norm(self.z)
        self.v = torch.exp(self.c)*self.z

        # hyperparams
        self.negative_hv = -0.1
        self.lr_mu = 1
        self.lr = (3+math.log(self.n_params))/(5*math.sqrt(self.n_params))

        # note when ngrad_c_j is negative, why NaN???
        self.ngrad_c_j_sign = 0

    def copy_to_mu(self, agent):
        for (name, p) in agent.named_parameters():
            shape, param_range = self.map_dict[name]
            a,b = param_range
            self.mu[0, a:b] = p.ravel()

    '''
    y ~ N(0,I)
    k ~ N(0,1)
    theta = mu + s*sigma
    return theta mapped with param names of the policy
    '''
    def generate_random_parameters(self, n_sample: int = 2, use_antithetic=True):
        if n_sample > 1:
            if use_antithetic:
                y_list = self.norm_dist.sample(
                    (int(n_sample/2), self.n_params))
                y_list = torch.cat((y_list, -y_list), dim=0).to(self.device)
                k_list = self.norm_dist.sample((int(n_sample/2), 1))
                k_list = torch.cat((k_list, -k_list), dim=0).to(self.device)
            else:
                y_list = self.norm_dist.sample((n_sample, self.n_params))
                k_list = self.norm_dist.sample((n_sample, 1))
        else:
            y_list = self.norm_dist.sample((1, self.n_params)).to(self.device)
            k_list = self.norm_dist.sample((1, 1)).to(self.device)

        g = torch.exp(self.ld) * (y_list + k_list*self.v)
        random_params = self.mu + g

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
    def update(self, w_list, x_list, f_list, weight=None):
        num_sample, M = f_list.shape
        if weight is None:
            weight = 1
        # generate
        d = self.n_params
        # prepare utility score
        utility = get_utility(num_sample, device=self.device)
        rank = get_nondominated_rank(f_list)
        score = utility[rank].unsqueeze(1)

        # prepare natural gradients
        # x_list = self.mu + torch.exp(self.ld)*w_list
        r = torch.norm(self.v)
        r = r.clamp(min=1e-6)
        u = self.v/r
        wtw = torch.sum(w_list*w_list, dim=1, keepdim=True)
        wtu = torch.sum(w_list*u, dim=1, keepdim=True)

        ngrad_mu_l = x_list
        ngrad_ld_l = 1/(2*(d-1)) * ((wtw-d) - ((wtu)**2-1))
        ngrad_v_l = ((r**2-d+2)*(wtu)**2-(r**2+1)*wtw) * \
            u/(2*r*(d-1)) + (wtu*w_list)/r
        nvtz = torch.sum(ngrad_v_l*u, dim=1, keepdim=True)
        ngrad_c_l = nvtz/r
        ngrad_z_l = (ngrad_v_l - nvtz*u)/r

        # start updating
        # print(weight.shape, score.shape, ngrad_mu_l.shape)
        ngrad_mu_j = torch.mean(weight*score*ngrad_mu_l, dim=0)
        ngrad_ld_j = torch.mean(weight*score*ngrad_ld_l, dim=0)
        self.mu = self.mu + self.lr_mu*ngrad_mu_j
        self.ld = self.ld + self.lr*ngrad_ld_j

        ngrad_c_j = torch.mean(weight*score*ngrad_c_l, dim=0)
        # conditional update on c,z,v to prevent unstable (flipping and large) v update
        if ngrad_c_j < 0:
            self.ngrad_c_j_sign = -1
            self.c = self.c + self.lr*ngrad_c_j
            ngrad_z_j = torch.mean(weight*score*ngrad_z_l, dim=0, keepdim=True)
            z_ = self.z + self.lr*ngrad_z_j
            self.z = z_/torch.norm(z_)
            self.v = torch.exp(self.c)*self.z
        else:
            self.ngrad_c_j_sign = 1
            ngrad_v_j = torch.mean(weight*score*ngrad_v_l, dim=0, keepdim=True)
            self.v = self.v + self.lr*ngrad_v_j
            self.c = torch.log(torch.norm(self.v)).unsqueeze(0)
            self.z = self.v/torch.norm(self.v)

    def logprob(self, sample_list):
        x_list = sample_list
        r = torch.norm(self.v)
        xtx = torch.sum(x_list*x_list, dim=1)
        xtv = torch.sum(x_list*self.v, dim=1)

        cc = self.n_params*math.log(2*math.pi)
        temp1 = -self.ld*self.n_params - \
            torch.log(1+r**2)/2 - torch.exp(-2*self.ld)*xtx/2
        temp2 = ((torch.exp(-2*self.ld))/(2*(1+r**2)))*xtv**2
        logprob = cc + temp1 + temp2
        return logprob

    # update with experience replay
    def update_with_er(self, er):
        num_samples = er.num_sample*er.num_saved_policy
        w_list = er.w_list[:num_samples, :]
        x_list = er.x_list[:num_samples, :]
        f_list = er.f_list[:num_samples, :]
        weight = get_multi_importance_weight(
            er.policy_list[:er.num_saved_policy], x_list)

        # adding BC to f_list as the third objective
        node_order_list = er.node_order_list[:num_samples, :]
        item_selection_list = er.item_selection_list[:num_samples, :]
        bc = get_bc_var(node_order_list, item_selection_list)
        f_list = torch.cat((f_list, bc), dim=1)
        self.update(w_list, x_list, f_list, weight)

    def set_new_device(self, new_device, actor_device=None):
        self.device = new_device
        self.mu = self.mu.to(self.device)
        self.ld = self.ld.to(self.device)
        self.v = self.v.to(self.device)
        self.c = self.c.to(self.device)
        self.z = self.z.to(self.device)

        if actor_device is not None:
            self.actor_device = actor_device

    # def write_progress_to_tb(self, writer, problem, f_list, normalized_f_list, epoch):
    def write_progress_to_tb(self, writer, sample_solutions, f_list, epoch):
        plt.figure()
        plt.scatter(f_list[:, 0], f_list[:, 1], c="blue")
        plt.scatter(sample_solutions[:, 0], sample_solutions[:, 1], c="red")
            # problem.sample_solutions[:, 0], problem.sample_solutions[:, 1], c="red")
        writer.add_figure("Solutions", plt.gcf(), epoch)

        # let's also note the hypervolume, we need it, boom
        # is_nondom = nondominated_sort(normalized_f_list)
        # nondom_solutions = normalized_f_list[is_nondom]
        # hv = get_hypervolume(nondom_solutions)
        # writer.add_scalar("Validation HV", hv, epoch)

        # note the parameters
        writer.add_scalar("Mu Norm", torch.norm(self.mu).cpu().item(), epoch)
        writer.add_scalar("V Norm", torch.norm(self.v).cpu().item(), epoch)
        writer.add_scalar("Lambda", self.ld.cpu().item(), epoch)
        writer.add_scalar("C", self.c.cpu().item(), epoch)
        writer.add_scalar("Z", torch.norm(self.z).cpu().item(), epoch)
        writer.flush()

'''
Lets separate experience replay for each policy, so that we can save whatever needed
without preprocessing in every update step to improve runtime.
'''
class ExperienceReplay(object):
    def __init__(self, dim, num_obj=2, max_saved_policy=5, num_sample=10, device=CPU_DEVICE):
        super().__init__()
        self.dim = dim
        self.max_saved_policy = max_saved_policy
        self.num_sample = num_sample
        self.device = device
        self.num_obj = num_obj

        self.num_saved_policy = 0
        self.policy_list = []

        self.w_list = torch.zeros(
            (self.max_saved_policy*num_sample, dim), dtype=torch.float32, device=self.device)
        self.x_list = torch.zeros(
            (self.max_saved_policy*num_sample, dim), dtype=torch.float32, device=self.device)
        self.f_list = torch.zeros(
            (self.max_saved_policy*num_sample, num_obj), dtype=torch.float32, device=self.device)
        self.node_order_list = None
        self.item_selection_list = None

    def clear(self):
        self.num_saved_policy = 0
        self.node_order_list = None
        self.item_selection_list = None


    def add(self, policy, sample_list, f_list, node_order_list, item_selection_list):
        self.num_saved_policy = min(
            self.num_saved_policy+1, self.max_saved_policy)

        # push to front, while rolling to remove overflowing samples
        self.policy_list = [policy] + \
            self.policy_list[0:self.max_saved_policy-1]

        x_list = sample_list - policy.mu
        self.x_list = self.x_list.roll(self.num_sample, dims=0)
        self.x_list[:self.num_sample, :] = x_list

        w_list = x_list/torch.exp(policy.ld)
        self.w_list = self.w_list.roll(self.num_sample, dims=0)
        self.w_list[:self.num_sample, :] = w_list

        self.f_list = self.f_list.roll(self.num_sample, dims=0)
        self.f_list[:self.num_sample] = f_list

        if self.node_order_list is None:
            num_sample, num_nodes = node_order_list.shape 
            self.node_order_list = torch.zeros((self.max_saved_policy*self.num_sample, num_nodes), dtype=torch.long)
        self.node_order_list = self.node_order_list.roll(self.num_sample, dims=0)
        self.node_order_list[:self.num_sample, :] = node_order_list

        if self.item_selection_list is None:
            num_sample, num_items = item_selection_list.shape 
            self.item_selection_list = torch.zeros((self.max_saved_policy*self.num_sample, num_items), dtype=torch.bool)
        self.item_selection_list = self.item_selection_list.roll(self.num_sample, dims=0)
        self.item_selection_list[:self.num_sample, :] = item_selection_list