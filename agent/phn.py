from typing import Dict
import torch as T
import torch.nn as nn

CPU_DEVICE = T.device("cpu")

# class PHN(T.jit.ScriptModule):
class PHN(T.nn.Module):
    def __init__(
            self,
            agent_template,
            ray_hidden_size: int=128,
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
        # self.pe_layer = nn.Linear(ray_hidden_size, num_neurons*3*num_neurons)
        layer_dict = {}
        shape_dict = {}
        for k,v in agent_template.named_parameters():
            if "normalizer" in k:
                continue
            module_name = k.replace(".","_")
            shape_dict[module_name] = v.shape
            total_shape = v.shape.numel()
            layer_dict[module_name] = nn.Linear(ray_hidden_size, total_shape)
        self.shape_dict = shape_dict
        self.layer_dict = T.nn.ModuleDict(layer_dict)
        self.to(device)

    # @T.jit.script_method
    def forward(self, ray: T.Tensor) -> Dict[str, T.Tensor]:
        '''
        ### Calculate embedding.
        -----

        Parameter:
            input: weight preferences/ray

        Return: appropriate weights
        '''
        ray_features = self.ray_layer(ray)
        param_dict = {}
        for k,v in self.layer_dict.items():
            param_dict[k] = v(ray_features).view(self.shape_dict[k])
        return param_dict
