import torch
import torch.nn as nn

class Encoder(nn.Module):
    """ RNN based Encoder """
    def __init__(self, history_window_size:int, horizon:int, hidden_size:int, num_layers:int, noise_size:int,\
                cell_type:str, device:str):
        """
        Args:
            history_window_size (int): Length of the history window
            horizon (int): Length of forecast horizon
            hidden_size (int): Number of the RNN cells in encoder
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
        
        self.model_mu = nn.Linear(in_features=hidden_size, out_features=noise_size)
        self.model_logvar = nn.Linear(in_features=hidden_size, out_features=noise_size)
        self.to(device)

    def forward(self, history_window:torch.Tensor, forecast:torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_window (torch.Tensor): (batch_size, history_window_size, 1)
            forecast (torch.Tensor): (batch_size, horizon)

        Returns:
            torch.Tensor: (batch_size, 1)
        """
        encoder_input = torch.cat((history_window, forecast.view(-1,self.horizon,1)), dim=1)
        # encoder_input: (batch_size, history_window_size + horizon, 1)

        encoder_input = encoder_input.transpose(0, 1)
        # encoder_input: (history_window_size + horizon, batch_size, 1)
        
        encoder_latent, _ = self.input_to_latent(encoder_input)
        # encoder_latent: (history_window_size + horizon, batch_size, hidden_size)
    
        encoder_latent = encoder_latent[-1]
        # encoder_latent: (batch_size, hidden_size)

        mu = self.model_mu(encoder_latent)
        logvar = self.model_logvar(encoder_latent)

        return mu,logvar