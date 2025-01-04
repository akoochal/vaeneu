import torch
import torch.nn as nn
import generators,encoders
from utils import config
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

class VAEneu(nn.Module):
    def __init__(self, config: config.Config):
        super(VAEneu, self).__init__()
        self.config = config

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        
        self.optimizer,self.sched_opt= self.get_optimizers()
        self.crps_part1 = nn.L1Loss()
        self.crps_part2 = nn.L1Loss()

    def reparameterization(self, mu, logvar):
        mu = mu.repeat(self.config.model.repeat_factor,1)
        logvar = logvar.repeat(self.config.model.repeat_factor,1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, history_window, ground_truth):

        mu,logvar = self.encoder(history_window,ground_truth)
        z = self.reparameterization(mu,logvar)
        history_window = history_window.repeat(self.config.model.repeat_factor,1,1)
        forecast = self.decoder(history_window,z)

        return forecast,mu,logvar
    
    def train(self, history_window, ground_truth):
        self.optimizer.zero_grad()

        forecast,mu,logvar = self.forward(history_window, ground_truth)
        forecast = forecast.reshape((self.config.model.repeat_factor,-1,forecast.shape[1],forecast.shape[2]))
        ground_truth = ground_truth.unsqueeze(0).repeat(forecast.size(0), 1, 1, 1)
        crps_app_part1 = self.crps_part1(ground_truth,forecast)
        
        
        forecast2 = forecast.clone()
        forecast2 = forecast2[torch.randperm(forecast2.shape[0]),:,:,:]
        crps_app_part2 = self.crps_part2(forecast2,forecast)
        recon_loss = crps_app_part1 - 0.5*crps_app_part2
        kld_loss = self.kld(mu,logvar)
        loss = kld_loss + recon_loss
        loss.backward()
        self.optimizer.step()

        return recon_loss.detach().cpu().numpy(),kld_loss.detach().cpu().numpy()

    def kld(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

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
        self.encoder.load_state_dict(checkpoint['e_state_dict'])
        self.decoder.load_state_dict(checkpoint['d_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step'], checkpoint['best_crps']

    def save_model(self, step, models_path, best_crps):
        torch.save({
            'step': step,
            'e_state_dict': self.encoder.state_dict(),
            'd_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_crps': best_crps,
        }, os.path.join(models_path))

    def get_encoder(self):
        model_config = self.config.model
        if model_config.enc.type in ["LSTM","GRU"]:
            enc = encoders.rnn.Encoder(history_window_size = model_config.history_window_size,
                                                horizon = 1,
                                                hidden_size = model_config.enc.hidden_size,
                                                num_layers = model_config.enc.num_layers,
                                                cell_type = model_config.enc.type,
                                                noise_size=model_config.dec.noise_size,
                                                device = model_config.device)

        elif model_config.enc.type == "TCN":
            enc = encoders.tcn.Encoder(history_window_size = model_config.history_window_size,
                                                horizon = 1,
                                                hidden_size = model_config.enc.hidden_size,
                                                noise_size=model_config.dec.noise_size,
                                                num_layers = model_config.enc.num_layers,
                                                kernel_size = model_config.enc.kernel_size,
                                                device = model_config.device)
            
        return enc
    
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
