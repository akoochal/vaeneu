import torch
import torch.nn as nn
from collections.abc import Callable
from helper_models.tcn import TemporalConvNet

class Encoder(nn.Module):
    def __init__(self, history_window_size:int, horizon:int, hidden_size:int,noise_size:int,\
                    num_layers:int, kernel_size:int,device:str):
        """
        Args:
            history_window_size (int): Length of the history window
            horizon (int): Length of forecast horizon
            hidden_size (int): Channel size in  in Temporal Block layers
            noise_size (int): The noise size which would be the output size of encoder
            num_layers (int): Number of Temporal Block inside TCN
            kernel_size (int): Kernel size  in Temporal Block
            device (str): 
        """
        super().__init__()
        self.history_window_size = history_window_size
        self.horizon =horizon

        channel_sizes = [hidden_size] * num_layers

        channel_sizes.insert(0,1)
        self.input_to_latent = TemporalConvNet(num_channels=channel_sizes,
                                                kernel_size=kernel_size)


        self.model_mu = nn.Linear(in_features=hidden_size, out_features=noise_size)
        self.model_logvar = nn.Linear(in_features=hidden_size, out_features=noise_size)
        self.to(device)

    def forward(self, history_window:torch.Tensor, forecast:torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_window (torch.Tensor): (batch_size, history_window_size, 1)
            forecast (torch.Tensor): (batch_size, horizon)

        Returns:
            torch.Tensor,torch.Tensor: (batch_size, noise_size),(batch_size, noise_size)
        """
        
        encoder_input = torch.cat((history_window, forecast.view(-1,self.horizon, 1)), dim=1)
        # encoder_input: (batch_size, history_window_size + horizon, 1)

        encoder_input = encoder_input.transpose(1, 2)
        # encoder_input: (batch_size,1, history_window_size + horizon)

        encoder_latent = self.input_to_latent(encoder_input)
        # encoder_latent: (batch_size,hidden_size)

        mu = self.model_mu(encoder_latent)
        logvar = self.model_logvar(encoder_latent)
        # mu: (batch_size, noise_size)
        # logvar: (batch_size, noise_size)
        return mu,logvar
