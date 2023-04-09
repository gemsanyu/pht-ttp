from typing import Dict
import torch as T
import torch.nn as nn

CPU_DEVICE = T.device("cpu")

class PHN(T.nn.Module):
# class PHN(T.jit.ScriptModule):
    def __init__(
            self,
            ray_hidden_size: int=128,
            num_neurons: int=128,
            device=CPU_DEVICE,
        ) -> None:
        '''
        ### Embedder class.
        -----
        It uses MLP method.

        Parameter:
            input_size: size for input in int
            hidden_layer_sizes: size for layers in hidden layer
        '''
        super(PHN, self).__init__()
        self.ray_init_layer = nn.Linear(2, ray_hidden_size)
        self.gae_init_layer = nn.Linear(num_neurons,ray_hidden_size)
        self.ray_layer = nn.Sequential(
                                        nn.Linear(2*ray_hidden_size, ray_hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(ray_hidden_size, ray_hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(ray_hidden_size, ray_hidden_size))
        self.v1_layer = nn.Linear(ray_hidden_size, num_neurons)
        # self.fe1_layer = nn.Linear(ray_hidden_size, 2*num_neurons*num_neurons)
        # self.qe1_layer = nn.Linear(ray_hidden_size, num_neurons*num_neurons)
        self.ray_hidden_size = ray_hidden_size
        self.num_neurons = num_neurons
        self.device = device
        self.to(device)

    # @T.jit.script_method
    def forward(self, ray: T.Tensor, graph_embeddings: T.Tensor) -> Dict[str, T.Tensor]:
        '''
        ### Calculate embedding.
        -----
        
        Parameter:
            input: weight preferences/ray
        
        Return: appropriate weights
        '''
        batch_size, _ = graph_embeddings.shape
        gae_features = self.gae_init_layer(graph_embeddings)
        ray_features = self.ray_init_layer(ray).expand_as(gae_features)
        features = T.concatenate([ray_features, gae_features], dim=1)
        # ray_features = self.ray_layer(ray)
        # v1 = self.v1_layer(ray_features).view(1,1,self.num_neurons)
        fembed = self.ray_layer(features)
        v1 = self.v1_layer(fembed).view(batch_size,1,self.num_neurons)
        # fe1_weight = self.fe1_layer(ray_features).view(self.num_neurons, 2*self.num_neurons)
        # qe1_weight = self.qe1_layer(ray_features).view(self.num_neurons, self.num_neurons)
        param_dict = {
                     "v1":v1,
                    #  "fe1_weight":fe1_weight,
                    #  "qe1_weight":qe1_weight,
                     }
        return param_dict
