import torch
import torch.nn as nn
from collections.abc import Callable
from helper_models.tcn import TemporalConvNet


class Generator(nn.Module):
    def __init__(self, noise_size:int,history_window_size:int, horizon:int, hidden_size:int,\
                    num_layers:int, kernel_size:int,device:str):
        """
        Args:
            noise_size (int): The dimensionality of noise size
            history_window_size (int): Length of the history window
            horizon (int): Length of forecast horizon
            hidden_size (int): Channel size in  in Temporal Block layers
            num_layers (int): Number of Temporal Block inside TCN
            kernel_size (int): Kernel size  in Temporal Block
            device (str): 
        """
        super().__init__()
        self.history_window_size = history_window_size
        self.horizon = horizon
        self.noise_size = noise_size
        self.device = device

        channel_sizes = [hidden_size] * num_layers
        channel_sizes.insert(0,1)

        self.history_window_to_latent = TemporalConvNet(num_channels=channel_sizes,
                                        kernel_size=kernel_size)



        self.model = nn.Sequential(
            nn.Linear(in_features=hidden_size + noise_size, out_features=hidden_size + noise_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size + noise_size, out_features=1)
        )

        self.to(self.device)

    def forward(self, history_window:torch.Tensor,noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_window (torch.Tensor): (batch_size, history_window_size, 1)
            noise (torch.Tensor): (batch_size,noise_size)

        Returns:
            torch.Tensor: (batch_size, 1)
        """

        history_window = history_window.transpose(1, 2)
        # history_window: (batch_size,1,history_window_size)
        latent = self.history_window_to_latent(history_window)
        # latent: (batch_size, hidden_size)
        generator_input = torch.cat((latent, noise), dim=1)
        # generator_input: (batch_size, hidden_size+noise_size)

        output = self.model(generator_input).unsqueeze(2)
        # output: (batch_size, 1,1)
        return output

    def forecast(self,history_window: torch.Tensor,n_samples: int,max_batch_size: int=100000)-> torch.Tensor:
        """ n_samples of forecasts for history_window

        Args:
            history_window (torch.Tensor): (batch_size, history_window_size, 1)
            n_samples (int): (batch_size, horizon, 1)

        Returns:
            torch.Tensor: (n_samples, batch_size, horizon, 1)
        """
        org_shape = history_window.shape
        history_window = history_window.unsqueeze(0).repeat((n_samples,1,1,1)).reshape((-1,org_shape[1],org_shape[2]))
        # history_window: (n_samples * batch_size, history_window_size, 1)


        result = []
        n_iter = -(history_window.shape[0] // -max_batch_size) #ceiling division

        for i in range(n_iter):
            input = history_window[i*max_batch_size:(i+1)*max_batch_size]
            noise = torch.randn(size=(input.shape[0],self.noise_size), device=self.device)
            # noise: (max_batch_size, noise_size)
            temp_res = [self.forward(input, noise)]

            for _ in range(self.horizon-1):
                prv_forecast = temp_res[-1]
                # input (max_batch_size, 1 , 1)

                input = torch.cat((input[:,1:,:],prv_forecast),dim=1)
                # input: (max_batch_size,history_window_size,1)
                noise = torch.randn(size=(input.shape[0],self.noise_size), device=self.device)
                # noise: (max_batch_size, noise_size)
                temp_res.append(self.forward(input, noise))
            
            temp_res = torch.cat(temp_res,dim=1)
            # temp_res: (max_batch_size,horizon,1)
            result.append(temp_res)

        if n_iter == 1:
            result = result[0].reshape((n_samples,org_shape[0],self.horizon,org_shape[2]))
        else:
            result = torch.cat(result,dim=0).reshape((n_samples,org_shape[0],self.horizon,org_shape[2]))
            # result (n_samples , batch_size, horizon , 1)


        return result