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
        self.layer = nn.Conv1d(input_size, hidden_size, kernel_size=1, device=device)
        # self.layer = nn.Linear(input_size, hidden_size, device=device)
        # if use_relu:
        #     self.layer = nn.Sequential(
        #                     nn.Linear(input_size, hidden_size, device=device),
        #                     nn.ReLU())
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
        if input.dim() == 3:
            out = self.layer(input.permute(0,2,1)).permute(0,2,1)
        else:
            out = self.layer(input.unsqueeze(2)).squeeze(2)
        return out
