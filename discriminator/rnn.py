import torch
import torch.nn as nn
from collections.abc import Callable

class Discriminator(nn.Module):
    """ RNN based Discriminator """
    def __init__(self, history_window_size:int, horizon:int, hidden_size:int, num_layers:int,\
                cell_type:str, device:str):
        """
        Args:
            history_window_size (int): Length of the history window
            horizon (int): Length of forecast horizon
            hidden_size (int): Number of the RNN cells in discriminator
            num_layers (int): Number of layers in RNN
            cell_type (str): The type of RNN cells (GRU/LSTM)
            device (str): 

        Raises:
            NotImplementedError: Throws when the specified RNN type is not implemented
        """

        super().__init__()
        self.history_window_size = history_window_size
        self.horizon = horizon


        if cell_type == "GRU":
            self.input_to_latent = nn.GRU(input_size=1,
                                            hidden_size=hidden_size,
                                            num_layers=num_layers)
        elif cell_type == "LSTM":
            self.input_to_latent = nn.LSTM(input_size=1,
                                            hidden_size=hidden_size,
                                            num_layers=num_layers)
            
        else:
            raise NotImplementedError

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
        discriminator_input = torch.cat((history_window, forecast.reshape(-1,self.horizon,1)), dim=1)
        # discriminator_input: (batch_size, history_window_size + horizon, 1)

        discriminator_input = discriminator_input.transpose(0, 1)
        # discriminator_input: (history_window_size + horizon, batch_size, 1)
        
        discriminator_latent, _ = self.input_to_latent(discriminator_input)
        # discriminator_latent: (history_window_size + horizon, batch_size, hidden_size)
    
        discriminator_latent = discriminator_latent[-1]
        # discriminator_latent: (batch_size, hidden_size)

        output = self.model(discriminator_latent)
        # output: (batch_size, 1)
        return output