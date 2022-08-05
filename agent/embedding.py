import torch as T
import torch.nn as nn

CPU_DEVICE = T.device("cpu")

class Embedder(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int=64,
            device=CPU_DEVICE,
            use_relu=False
        ) -> None:
        '''
        ### Embedder class.
        -----
        It uses MLP method.

        Parameter:
            input_size: size for input in int
            hidden_layer_sizes: size for layers in hidden layer
        '''
        super(Embedder, self).__init__()
        self.layer = nn.Linear(input_size, hidden_size, device=device)
        if use_relu:
            self.layer = nn.Sequential(
                            nn.Linear(input_size, hidden_size, device=device),
                            nn.ReLU())
        self.to(device)

    def forward(self, input: T.Tensor) -> T.Tensor:
        '''
        ### Calculate embedding.
        -----

        Parameter:
            input: features with torch.tensor type
                    (customers static features)
        Return: an embedded features
        '''
        return self.layer(input)
