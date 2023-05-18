import os
import pathlib

import numpy as np
import torch
from pymoo.indicators.igd import IGD

from policy.non_dominated_sorting import fast_non_dominated_sort
from policy.hv import Hypervolume
from policy.normalization import normalize
from policy.utils import combine_with_nondom

"""
omega is the patience length for
early stopping
"""
class Validator:
    def __init__(self,
                 omega:int,
                 num_validation_samples:int,
                 num_objectives:int=2):
        self.omega = omega
        self.num_validation_samples = num_validation_samples
        self.num_objectives = num_objectives
        self.nadir_points = None
        self.utopia_points = None
        self.delta_nadir = None
        self.delta_utopia = None
        self.mean_delta_nadir = None
        self.mean_delta_utopia = None
        self.nd_solutions_list = None
        self.running_igd = None
        self.mean_running_igd = None
        self.initial_utopia_points = None
        self.initial_nadir_points = None
        self.hv_list = None
        self.best_mean = None
        self.init_box_area = None
        self.epoch = 0
    
    def compute_init_box_area(self):
        print(self.initial_utopia_points)
        print(self.initial_nadir_points)
        w_list = self.initial_nadir_points-self.initial_utopia_points
        A = w_list[:,0]*w_list[:,1]
        self.init_box_area = A

    def set_initial_utopia_points(self, utopia_points):
        self.initial_utopia_points = utopia_points

    def set_initial_nadir_points(self, nadir_points):
        self.initial_nadir_points = nadir_points

    @property
    def is_improving(self):
        if self.mean_running_igd is None or len(self.mean_running_igd) <=1:
            return True
        return self.mean_running_igd[-1] >= self.mean_running_igd[-2]
        # max_running_igd = float(np.max(self.mean_running_igd))
        # max_delta_nadir = float(np.max(self.mean_delta_nadir))
        # max_delta_utopia = float(np.max(self.mean_delta_utopia))
        # is_convergence_done = (max_delta_nadir<=0.05) and (max_delta_utopia<=0.05)
        # is_diversification_done = max_running_igd<=0.05
        # return not (is_convergence_done and is_diversification_done)
    
    def get_last_mean_running_igd(self):
        if self.mean_running_igd is None:
            return None
        return self.mean_running_igd[-1]

    def get_last_delta_refpoints(self):
        if self.mean_delta_nadir is None:
            return None, None
        return self.mean_delta_nadir[-1], self.mean_delta_utopia[-1]

    def insert_new_nd_solutions(self, nd_solutions_list):
        if self.nd_solutions_list is None:
            self.nd_solutions_list = [nd_solutions_list]
        else:
            last_nd_solutions_list = self.nd_solutions_list[-1]
            new_nd_solutions_list = []
            for i in range(len(last_nd_solutions_list)):
                all_nd_solutions = np.concatenate([last_nd_solutions_list[i], nd_solutions_list[i]])
                nondom_idx = fast_non_dominated_sort(all_nd_solutions)[0]
                new_nd_solutions = all_nd_solutions[nondom_idx]
                new_nd_solutions_list += [new_nd_solutions]
            self.nd_solutions_list += [new_nd_solutions_list]
            # exit()
            # self.nd_solutions_list += [nd_solutions_list]
        if len(self.nd_solutions_list) > self.omega:
            self.nd_solutions_list = self.nd_solutions_list[-self.omega:]
        self.update_running_igd(nd_solutions_list)
        
    def insert_new_ref_points(self, new_nadir_points, new_utopia_points):
        new_nadir_points = new_nadir_points.copy()
        new_utopia_points = new_utopia_points.copy()
        if self.nadir_points is None:
            self.nadir_points = new_nadir_points[np.newaxis, :, :]
            self.utopia_points = new_utopia_points[np.newaxis, :, :]
        else:
            last_nadir_points = self.nadir_points[-1]
            last_utopia_points = self.utopia_points[-1]
            new_nadir_points = np.maximum(last_nadir_points, new_nadir_points)
            new_utopia_points = np.minimum(new_utopia_points, last_utopia_points)
            self.nadir_points = np.concatenate([self.nadir_points, new_nadir_points[np.newaxis, :, :]], axis=0)
            self.utopia_points = np.concatenate([self.utopia_points, new_utopia_points[np.newaxis, :, :]], axis=0)
        if len(self.nadir_points) > self.omega:
            self.nadir_points = self.nadir_points[-self.omega:]
            self.utopia_points = self.utopia_points[-self.omega:]
        self.update_delta_ref_points()

    def update_running_igd(self, new_nd_solutions_list):
        if len(self.nd_solutions_list) <= 10 or self.nd_solutions_list is None:
            return
        num_problems = len(self.nd_solutions_list[0])
        running_igd_list = []
        for i in range(num_problems):
            current_nd_solutions = self.nd_solutions_list[-1][i]
            new_nd_solutions = new_nd_solutions_list[i]
            all_nd_solutions = np.concatenate([current_nd_solutions, new_nd_solutions])
            nadir_points = np.max(all_nd_solutions, axis=0, keepdims=True)
            utopia_points = np.min(all_nd_solutions, axis=0, keepdims=True)
            # denom = nadir_points - utopia_points
            # last_nd_solutions = self.nd_solutions_list[-2][i]
            # nadir_points = self.nadir_points[-1,i,np.newaxis,:]
            # utopia_points = self.utopia_points[-1,i,np.newaxis,:]
            denom = nadir_points-utopia_points
            denom[denom==0] = 1
            current_nd_solutions = (current_nd_solutions-utopia_points)/denom
            new_nd_solutions = (new_nd_solutions-utopia_points)/denom
            # combined_nd = np.concatenate([current_nd_solutions, last_nd_solutions], axis=0)
            # nondom_idx = fast_non_dominated_sort(combined_nd)[0]
            # combined_nd = combined_nd[nondom_idx]
            igd_getter = IGD(pf=current_nd_solutions)
            running_igd = igd_getter._do(new_nd_solutions)
            running_igd_list += [running_igd]
        running_igd_list = np.asanyarray([running_igd_list])
        if self.running_igd is None:
            self.running_igd = running_igd_list
        else:
            self.running_igd = np.concatenate([self.running_igd, running_igd_list], axis=0)
        new_mean_running_igd = np.mean(self.running_igd[-1], keepdims=True)
        if self.mean_running_igd is None:
            self.mean_running_igd = new_mean_running_igd
        else:
            self.mean_running_igd = np.concatenate([self.mean_running_igd, new_mean_running_igd],axis=0)
        if len(self.running_igd) > self.omega:
            self.running_igd = self.running_igd[-self.omega:]
            self.mean_running_igd = self.mean_running_igd[-self.omega:]
        
    def update_delta_ref_points(self):
        if len(self.nadir_points) == 1 or self.nadir_points is None:
            return
        len_history = len(self.nadir_points)
        numerator_nadir = np.abs(self.nadir_points[len_history-2] - self.nadir_points[len_history-1])
        numerator_utopia = np.abs(self.utopia_points[len_history-2] - self.utopia_points[len_history-1])
        denominator = self.nadir_points[len_history-1] - self.utopia_points[len_history-1]
        denominator[denominator==0] = 1
        new_delta_nadir = np.max(numerator_nadir/denominator, axis=-1)[np.newaxis, :]
        new_delta_utopia = np.max(numerator_utopia/denominator, axis=-1)[np.newaxis, :]
        if self.delta_nadir is None:
            self.delta_nadir = new_delta_nadir
            self.delta_utopia = new_delta_utopia
            self.mean_delta_nadir = np.mean(new_delta_nadir)[np.newaxis]
            self.mean_delta_utopia = np.mean(new_delta_utopia)[np.newaxis]
        else:
            self.delta_nadir = np.concatenate([self.delta_nadir, new_delta_nadir], axis=0)
            self.delta_utopia = np.concatenate([self.delta_utopia, new_delta_utopia], axis=0)        
            self.mean_delta_nadir =  np.concatenate([self.mean_delta_nadir, np.mean(new_delta_nadir)[np.newaxis]])
            self.mean_delta_utopia = np.concatenate([self.mean_delta_utopia, np.mean(new_delta_utopia)[np.newaxis]])
        if len(self.delta_nadir) > self.omega:
            self.delta_nadir = self.delta_nadir[-self.omega:]
            self.delta_utopia = self.delta_utopia[-self.omega:]
            self.mean_delta_nadir = self.mean_delta_nadir[-self.omega:]
            self.mean_delta_utopia = self.mean_delta_utopia[-self.omega:]

    def compute_new_hv(self, nd_solutions_list):
        # if self.initial_nadir_points is None:
        #     nadir_points = []
        #     utopia_points = []
        #     for i, nd_solutions in enumerate(nd_solutions_list):
        #         utopia_point = np.min(nd_solutions, axis=0, keepdims=True)
        #         utopia_points += [utopia_point]
        #         nadir_point = np.max(nd_solutions, axis=0, keepdims=True)
        #         nadir_points += [nadir_point]
        #     utopia_points = np.concatenate(utopia_points)
        #     nadir_points = np.concatenate(nadir_points)
        #     self.set_initial_utopia_points(utopia_points)
        #     self.set_initial_nadir_points(nadir_points)
        #     self.compute_init_box_area()

        hv_list = []
        for i, nd_solutions in enumerate(nd_solutions_list):
            utopia_point = np.min(nd_solutions, axis=0)
            nadir_point = np.minimum(utopia_point, self.initial_utopia_points[i])
            nadir_point = np.max(nd_solutions, axis=0)
            nadir_point = np.maximum(nadir_point, self.initial_nadir_points[i])
            w_list = nadir_point-utopia_point
            new_box_area = w_list[0]*w_list[1]
            box_ratio = new_box_area/self.init_box_area[i]
            _N = normalize(nd_solutions, utopia_point, nadir_point)
            _hv = Hypervolume(np.array([1.1,1.1])).calc(_N)
            # print(_hv)
            # print(_N)
            hv_list += [_hv[np.newaxis]*box_ratio]
        # print("----------------")
        _hv_list = np.concatenate(hv_list)
        if self.hv_list is None:
            self.best_mean = _hv_list.mean()
            self.hv_list = _hv_list[np.newaxis, :]
        else:
            self.hv_list = np.concatenate([self.hv_list,_hv_list[np.newaxis,:]])
        if len(self.hv_list > self.omega):
            self.hv_list = self.hv_list[-self.omega:,:]

# def reflect_diagonal(X):
#     R = np.asanyarray([[0,-1],[-1,0]],dtype=np.float32)
#     temp = np.matmul(X,R)
#     return temp

def load_validator(args) -> Validator:
    title = args.title
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validator_path = checkpoint_dir/(title+"_validator.pt")
    validator = Validator(args.omega, args.num_validation_samples)
    if os.path.isfile(validator_path.absolute()):
        validator = torch.load(validator_path.absolute())
    return validator

def save_validator(validator, title):
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validator_path = checkpoint_dir/(title+"_validator.pt")
    validator_path_ = checkpoint_dir/(title+"_validator.pt_") #backup
    torch.save(validator, validator_path.absolute())
    torch.save(validator, validator_path_.absolute())