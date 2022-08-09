from typing import Dict
import torch as T
import torch.nn as nn

CPU_DEVICE = T.device("cpu")

class PHN(T.jit.ScriptModule):
    def __init__(
            self,
            ray_hidden_size: int=128,
            num_neurons: int=64,
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
        self.ray_layer = nn.Sequential(
                                        nn.Linear(2, ray_hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(ray_hidden_size, ray_hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(ray_hidden_size, ray_hidden_size))
        self.v0_layer = nn.Linear(ray_hidden_size, num_neurons)
        self.v1_layer = nn.Linear(ray_hidden_size, num_neurons)
        self.fe0_layer = nn.Linear(ray_hidden_size, 2*num_neurons*num_neurons)
        self.fe1_layer = nn.Linear(ray_hidden_size, 2*num_neurons*num_neurons)
        self.fe0_bias = nn.Linear(ray_hidden_size, num_neurons)
        self.fe1_bias = nn.Linear(ray_hidden_size, num_neurons)
        self.qe0_layer = nn.Linear(ray_hidden_size, num_neurons*num_neurons)
        self.qe1_layer = nn.Linear(ray_hidden_size, num_neurons*num_neurons)
        self.qe0_bias = nn.Linear(ray_hidden_size, num_neurons)
        self.qe1_bias = nn.Linear(ray_hidden_size, num_neurons)
        self.ray_hidden_size = ray_hidden_size
        self.num_neurons = num_neurons
        self.to(device)

    @T.jit.script_method
    def forward(self, ray: T.Tensor) -> Dict[str, T.Tensor]:
        '''
        ### Calculate embedding.
        -----

        Parameter:
            input: weight preferences/ray

        Return: appropriate weights
        '''
        ray_features = self.ray_layer(ray)
        v0 = self.v0_layer(ray_features).view(1,1,self.num_neurons)
        v1 = self.v1_layer(ray_features).view(1,1,self.num_neurons)
        fe0_weight = self.fe0_layer(ray_features).view(self.num_neurons, 2*self.num_neurons)
        fe1_weight = self.fe1_layer(ray_features).view(self.num_neurons, 2*self.num_neurons)
        fe0_bias = self.fe0_bias(ray_features).view(self.num_neurons)
        fe1_bias = self.fe1_bias(ray_features).view(self.num_neurons)
        qe0_weight = self.qe0_layer(ray_features).view(self.num_neurons, self.num_neurons)
        qe1_weight = self.qe1_layer(ray_features).view(self.num_neurons, self.num_neurons)
        qe0_bias = self.qe0_bias(ray_features).view(self.num_neurons)
        qe1_bias = self.qe1_bias(ray_features).view(self.num_neurons)
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
