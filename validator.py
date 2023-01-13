import os
import pathlib
import platform

import torch

from ttp.ttp_dataset import read_prob, prob_list_to_env

def load_validation_env_list(num_instance_per_config=3):
    # pickle error if run on linux, cuz the data is generated
    # on windows
    if platform.system() == 'Linux':
        pathlib.WindowsPath = pathlib.PosixPath
    num_nodes_list = [20,30]
    nipc_list = [1,3,5]
    num_ic = 3
    config_list = [(nn, nipc) for nn in num_nodes_list for nipc in nipc_list]
    data_root = "data_full" 
    validation_dir = pathlib.Path(".")/data_root/"validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    solution_dir = validation_dir/"solutions"
    env_list = []
    env_nadir_points, env_utopia_points = [], []
    for config in config_list:
        nn, nipc = config[0],config[1]
        prob_list = []
        nadir_points, utopia_points = [], []
        for ic in range(num_ic):
            for idx in range(num_instance_per_config):
                dataset_name = "nn_"+str(nn)+"_nipc_"+str(nipc)+"_ic_"+str(ic)+"_"+str(idx)
                dataset_path = validation_dir/(dataset_name+".pt")
                solution_path = solution_dir/(dataset_name+".txt")  
                prob = read_prob(dataset_path=dataset_path)
                solutions = []
                with open(solution_path.absolute(), "r") as data_file:
                    lines = data_file.readlines()
                    for i, line in enumerate(lines):
                        strings = line.split()
                        sol = [float(strings[0]), float(strings[1])]
                        solutions += [sol]
                sample_solutions = torch.tensor(solutions)
                sample_solutions[:,1] = -sample_solutions[:,1]
                nadir_point, _ = torch.max(sample_solutions, dim=0)
                nadir_points += [nadir_point.unsqueeze(0)]
                utopia_point, _ = torch.min(sample_solutions, dim=0)
                utopia_points += [utopia_point.unsqueeze(0)]
                prob_list += [prob]
        nadir_points = torch.cat(nadir_points).unsqueeze(0)
        utopia_points = torch.cat(utopia_points).unsqueeze(0)
        env_nadir_points += [nadir_points]
        env_utopia_points += [utopia_points]
        env = prob_list_to_env(prob_list)
        env_list += [env]
    env_nadir_points = torch.cat(env_nadir_points)
    env_utopia_points = torch.cat(env_utopia_points)
    return env_list, env_nadir_points, env_utopia_points 


class Validator:
    def __init__(self, 
                validation_env_list=None,
                utopia_points=None,
                nadir_points=None):
        if validation_env_list is None:
            validation_env_list, nadir_points, utopia_points = load_validation_env_list(num_instance_per_config=3)
        self.validation_env_list = validation_env_list
        self.nadir_points = nadir_points
        self.utopia_points = utopia_points
        batch_size = validation_env_list[0].batch_size
        num_env = len(validation_env_list)
        self.hv_history = None
        self.best_hv = 0
        self.last_hv = 0
        self.epoch = 0
        self.is_improving = True
        self.hv_multiplier = torch.ones((num_env, batch_size, 1), dtype=torch.float32)
        # when nadir_points/utopia_points are updated, then
        # the rectangle must be enlarged, right? 
        # so we multiply the new hv     

    def update_ref_points(self, new_nadir_points, new_utopia_points):
        # update hv multiplier
        old_nadir_points, old_utopia_points = self.nadir_points, self.utopia_points
        old_area = (old_nadir_points[:,:,0]-old_utopia_points[:,:,0])*(old_nadir_points[:,:,1]-old_utopia_points[:,:,1])
        new_area = (new_nadir_points[:,:,0]-new_utopia_points[:,:,0])*(new_nadir_points[:,:,1]-new_utopia_points[:,:,1])
        ratio = (new_area/old_area).unsqueeze(2)
        # replace only with bigger areas
        is_bigger = (ratio > 1).expand_as(old_nadir_points)
        self.nadir_points[is_bigger] = new_nadir_points[is_bigger]
        self.utopia_points[is_bigger] = new_utopia_points[is_bigger]
        
        #update hv multiplier
        ratio[ratio<=1] = 1
        self.hv_multiplier *= ratio

    def insert_hv_history(self, new_hv_history):
        new_hv_history = new_hv_history * self.hv_multiplier
        new_hv_mean = new_hv_history.mean()
        self.last_hv = new_hv_mean
        if new_hv_mean > self.best_hv:
            self.best_hv = new_hv_mean 
            self.is_improving = True
        else:
            self.is_improving = False
        if self.hv_history is None:
            self.hv_history = new_hv_history
        else:
            self.hv_history = torch.cat([self.hv_history, new_hv_history], dim=-1)
        

def load_validator(title) -> Validator:
    checkpoint_root = "checkpoints"
    checkpoint_dir = pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validator_path = checkpoint_dir/(title+"_validator.pt")
    validator = Validator()
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