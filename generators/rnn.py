import torch
import torch.nn as nn
from collections.abc import Callable

class Generator(nn.Module):
    """ RNN based Generator """
    def __init__(self, noise_size:int, history_window_size:int, horizon:int, hidden_size:int, num_layers:int, cell_type:str, device:str):
        """
        Args:
            noise_size (int): Dimensionality of the noise space
            history_window_size (int): Length of the history window
            horizon (int): Length of forecast horizon
            hidden_size (int): Number of the RNN cells in generator
            num_layers (int): Number of layers in RNN
            cell_type (str): The type of RNN cells (GRU/LSTM)
            device (str): 

        Raises:
            NotImplementedError: Throws when the specified RNN type is not implemented
        """

        super().__init__()
        self.history_window_size = history_window_size
        self.horizon = horizon
        self.device = device
        self.noise_size = noise_size

        if cell_type == "GRU":
            self.history_window_to_latent = nn.GRU(input_size= 1,
                                                    hidden_size=hidden_size,
                                                    num_layers=num_layers)
        elif cell_type == "LSTM":
            self.history_window_to_latent = nn.LSTM(input_size= 1,
                                                    hidden_size=hidden_size,
                                                    num_layers=num_layers)
                                                    
        else:
            raise NotImplementedError

        self.model = nn.Sequential(
            nn.Linear(in_features=hidden_size + noise_size, out_features=hidden_size + noise_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size + noise_size , out_features=1)
        )

        self.to(device)

    def forward(self, history_window: torch.Tensor,noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_window (torch.Tensor): (batch_size, history_window_size, 1)
            noise (torch.Tensor): (batch_size,noise_size)

        Returns:
            torch.Tensor: one step ahead forecast of shape (batch_size, 1)
        """

        history_window = history_window.transpose(0, 1)
        # history_window (history_window_size, batch_size, 1)

        _latent, _ = self.history_window_to_latent(history_window)
        # latent (history_window_size, batch size, hidden size)

        latent = _latent[-1]
        # latent (batch size, hidden size)

        generator_input = torch.cat((latent, noise), dim=1)
        # generator_input (batch_size, hidden size + noise_size)

        output = self.model(generator_input).unsqueeze(2)
        # output (batch_size, 1,1)

        return output
    
    def forecast(self,history_window: torch.Tensor,n_samples: int,max_batch_size: int)-> torch.Tensor:
        """ n_samples of forecasts for history_window

        Args:
            history_window (torch.Tensor): (batch_size, history_window_size, 1)
            n_samples (int): (batch_size, horizon, 1)

        Returns:
            torch.Tensor: (n_samples, batch_size, horizon, 1)
        """
        org_shape = history_window.shape
        history_window = history_window.unsqueeze(0).repeat((n_samples,1,1,1)).reshape((-1,org_shape[1],org_shape[2]))

        result = []
        n_iter = -(history_window.shape[0] // -max_batch_size) #ceiling division

        for i in range(n_iter):
            input = history_window[i*max_batch_size:(i+1)*max_batch_size]

            noise = torch.randn(size=(input.shape[0],self.noise_size), device=self.device)
            # noise: (max_batch_size, noise_size)
            
            input = input.transpose(0, 1)
            # input: (history_window_size, max_batch_size, 1)

            _latent, _hidden = self.history_window_to_latent(input)
            # _latent: (history_window_size, max_batch_size, hidden size)
            # _hidden: ( num_layers, max_batch_size, hidden size)

            latent = _latent[-1]
            # latent: (max_batch_size, hidden size)

            temp_res = [self.model( torch.cat((latent, noise), dim=1)).unsqueeze(2)]

            for _ in range(self.horizon-1):
                input = temp_res[-1]
                # input (max_batch_size, 1 , 1)

                input = input.transpose(0, 1)
                # input (1,max_batch_size, 1)
            
                _latent, _hidden = self.history_window_to_latent(input,_hidden)
                # _latent: (1, max_batch_size, hidden size)
                # _hidden: ( num_layers, max_batch_size, hidden size)
                
                latent = _latent[-1]
                # latent: (max_batch_size, hidden size)
                
                noise = torch.randn(size=(latent.shape[0],self.noise_size), device=self.device)
                # noise: (max_batch_size, noise_size)

                temp_res.append(self.model( torch.cat((latent, noise), dim=1)).unsqueeze(2))
            
            temp_res = torch.cat(temp_res,dim=1)
            # temp_res: (max_batch_size,horizon,1)
            result.append(temp_res)

        if n_iter == 1:
            result = result[0].reshape((n_samples,org_shape[0],self.horizon,org_shape[2]))
        else:
            result = torch.cat(result,dim=0).reshape((n_samples,org_shape[0],self.horizon,org_shape[2]))
            # result (n_samples , batch_size, horizon , 1)
        return result