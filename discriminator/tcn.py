import torch
import torch.nn as nn
from collections.abc import Callable
from helper_models.tcn import TemporalConvNet

class Discriminator(nn.Module):
    def __init__(self, history_window_size:int, horizon:int, hidden_size:int,\
                    num_layers:int, kernel_size:int,device:str):
        """
        Args:
            history_window_size (int): Length of the history window
            horizon (int): Length of forecast horizon
            hidden_size (int): Channel size in  in Temporal Block layers
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


        self.model = nn.Linear(in_features=hidden_size, out_features=1)
        self.to(device)

    def forward(self, history_window:torch.Tensor, forecast:torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_window (torch.Tensor): (batch_size, history_window_size, 1)
            forecast (torch.Tensor): (batch_size, horizon)

        Returns:
            torch.Tensor: (batch_size, 1)
        """
        
        discriminator_input = torch.cat((history_window, forecast.view(-1,self.horizon, 1)), dim=1)
        # discriminator_input: (batch_size, history_window_size + horizon, 1)

        discriminator_input = discriminator_input.transpose(1, 2)
        # discriminator_input: (batch_size,1, history_window_size + horizon)

        discriminator_latent = self.input_to_latent(discriminator_input)
        # discriminator_latent: (batch_size,hidden_size)

        output = self.model(discriminator_latent)
        # output: (batch_size, 1)
        return output
