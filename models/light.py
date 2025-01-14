import torch
import torch.nn as nn
import generators
from utils import config
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Light(nn.Module):
    def __init__(self, config: config.Config):
        super(Light, self).__init__()
        self.config = config
        self.decoder = self.get_decoder()
        
        self.optimizer,self.sched_opt= self.get_optimizers()
        self.crps_part1 = nn.L1Loss()
        self.crps_part2 = nn.L1Loss()

    def forward(self, history_window):
        
        history_window = history_window.repeat(self.config.model.repeat_factor,1,1)
        z = torch.randn(size=(history_window.shape[0],self.config.model.dec.noise_size), device=self.decoder.device)
        forecast = self.decoder(history_window,z)

        return forecast
    
    def train(self, history_window, ground_truth):
        self.optimizer.zero_grad()
        forecast = self.forward(history_window)
        forecast = forecast.reshape((self.config.model.repeat_factor,-1,forecast.shape[1],forecast.shape[2]))
        ground_truth = ground_truth.unsqueeze(0).repeat(forecast.size(0), 1, 1, 1)
        crps_app_part1 = self.crps_part1(ground_truth,forecast)
        
        
        forecast2 = forecast.clone()
        forecast2 = forecast2[torch.randperm(forecast2.shape[0]),:,:,:]
        crps_app_part2 = self.crps_part2(forecast2,forecast)
        loss = crps_app_part1 - 0.5*crps_app_part2
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def report_progress(self,val_crps):
        self.sched_opt.step(val_crps)

    def get_forecast(self, history_window, n_samples,max_batch_size):
        self.decoder.eval()

        with torch.no_grad():
            forecast = self.decoder.forecast(history_window,n_samples,max_batch_size).detach().cpu().numpy()
                
        self.decoder.train()
        return forecast

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.config.model.device)
        self.decoder.load_state_dict(checkpoint['d_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step'], checkpoint['best_crps']

    def save_model(self, step, models_path, best_crps):
        torch.save({
            'step': step,
            'd_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_crps': best_crps,
        }, os.path.join(models_path))
    
    def get_decoder(self):
        model_config = self.config.model
        
        if model_config.dec.type in ["LSTM","GRU"]:
            dec = generators.rnn.Generator(noise_size = model_config.dec.noise_size,
                                        history_window_size = model_config.history_window_size,
                                        horizon = model_config.horizon,
                                        hidden_size = model_config.dec.hidden_size,
                                        num_layers = model_config.dec.num_layers,
                                        cell_type = model_config.dec.type,
                                        device = model_config.device)

        elif model_config.dec.type == "TCN":
            dec = generators.tcn.Generator(noise_size = model_config.dec.noise_size,
                                        history_window_size = model_config.history_window_size,
                                        horizon = model_config.horizon,
                                        hidden_size = model_config.dec.hidden_size,
                                        num_layers = model_config.dec.num_layers,
                                        kernel_size = model_config.dec.kernel_size,
                                        device = model_config.device)

        return dec
    
    def get_optimizers(self):
        model_config = self.config.model

        if model_config.optimizer == "sgd":
            opt = torch.optim.SGD(self.parameters(),lr=model_config.lr)

        elif model_config.optimizer == "rms":
            opt = torch.optim.RMSprop(self.parameters(),lr=model_config.lr)

        elif model_config.optimizer == "adam":
            opt = torch.optim.Adam(self.parameters(),lr=model_config.lr)

        sched_opt = ReduceLROnPlateau(opt, factor=0.5, patience=50, min_lr=0.00001,verbose=True)

        return opt,sched_opt
