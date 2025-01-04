
import torch
import numpy as np
import generators,discriminator
from utils import losses,config
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ForGAN:
    def __init__(self, config: config.Config):
        self.config = config
        self.discriminator = self.get_discriminator()
        self.generator = self.get_generator()
        self.d_optimizer, self.g_optimizer,self.sched_d_opt,self.sched_g_opt = self.get_optimizers()
        self.d_apply_loss, self.g_apply_loss = self.get_losses()
        if self.config.model.loss == "wgp":
            self.lambda_gp = self.config.model.lambda_gp

    def train_discriminator(self, history_window, ground_truth):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        noise = torch.randn(size=(history_window.shape[0],self.config.model.gen.noise_size), device=self.config.model.device)
        forecast = self.generator(history_window,noise)
        d_logit_real = self.discriminator(history_window, ground_truth)
        d_logit_fake = self.discriminator(history_window, forecast)
        d_loss = self.d_apply_loss(d_logit_real, d_logit_fake)

        if self.config.model.loss == "wgp":
            gradient_penalty = losses.cal_grad_penalty(ground_truth, history_window, forecast, self.discriminator, self.config.model.device)
            d_loss += self.lambda_gp * gradient_penalty

        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.detach().cpu().numpy()

    def train_generator(self, history_window):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

        noise = torch.randn(size=(history_window.shape[0],self.config.model.gen.noise_size), device=self.config.model.device)
        forecast = self.generator(history_window,noise)

        d_logit_fake = self.discriminator(history_window, forecast)
        g_loss = self.g_apply_loss(d_logit_fake)

        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.detach().cpu().numpy()
    
    def report_progress(self,val_crps):
        self.sched_d_opt.step(val_crps)
        self.sched_g_opt.step(val_crps)

    def get_forecast(self, history_window, n_samples,max_batch_size):
        self.generator.eval()

        with torch.no_grad():
            forecast = self.generator.forecast(history_window,n_samples,max_batch_size).detach().cpu().numpy()
                
        self.generator.train()
        return forecast

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.config.model.device)
        self.generator.load_state_dict(checkpoint['g_state_dict'])
        self.discriminator.load_state_dict(checkpoint['d_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        return checkpoint['step'], checkpoint['best_crps']

    def save_model(self, step, models_path, best_crps):
        torch.save({
            'step': step,
            'g_state_dict': self.generator.state_dict(),
            'd_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'best_crps': best_crps,
        }, os.path.join(models_path))

    def get_discriminator(self):
        model_config = self.config.model

        if model_config.dsc.type in ["LSTM","GRU"]:
            d = discriminator.rnn.Discriminator(history_window_size = model_config.history_window_size,
                                                horizon = 1,
                                                hidden_size = model_config.dsc.hidden_size,
                                                num_layers = model_config.dsc.num_layers,
                                                cell_type = model_config.dsc.type,
                                                device = model_config.device)

        elif model_config.dsc.type == "TCN":
            d = discriminator.tcn.Discriminator(history_window_size = model_config.history_window_size,
                                                horizon = 1 ,
                                                hidden_size = model_config.dsc.hidden_size,
                                                num_layers = model_config.dsc.num_layers,
                                                kernel_size = model_config.dsc.kernel_size,
                                                device = model_config.device)
            
        return d
    
    def get_generator(self):
        model_config = self.config.model
        
        if model_config.gen.type in ["LSTM","GRU"]:
            g = generators.rnn.Generator(noise_size = model_config.gen.noise_size,
                                        history_window_size = model_config.history_window_size,
                                        horizon = model_config.horizon,
                                        hidden_size = model_config.gen.hidden_size,
                                        num_layers = model_config.gen.num_layers,
                                        cell_type = model_config.gen.type,
                                        device = model_config.device)

        elif model_config.gen.type == "TCN":
            g = generators.tcn.Generator(noise_size = model_config.gen.noise_size,
                                        history_window_size = model_config.history_window_size,
                                        horizon = model_config.horizon,
                                        hidden_size = model_config.gen.hidden_size,
                                        num_layers = model_config.gen.num_layers,
                                        kernel_size = model_config.gen.kernel_size,
                                        device = model_config.device)

        return g
    
    def get_optimizers(self):
        model_config = self.config.model

        if model_config.optimizer == "sgd":
            d_opt = torch.optim.SGD(self.discriminator.parameters(),lr=model_config.lr)
            g_opt = torch.optim.SGD(self.generator.parameters(),lr=model_config.lr)
        elif model_config.optimizer == "rms":
            d_opt = torch.optim.RMSprop(self.discriminator.parameters(),lr=model_config.lr)
            g_opt = torch.optim.RMSprop(self.generator.parameters(),lr=model_config.lr)
        elif model_config.optimizer == "adam":
            d_opt = torch.optim.Adam(self.discriminator.parameters(),lr=model_config.lr,betas=(0.5,0.999))
            g_opt = torch.optim.Adam(self.generator.parameters(),lr=model_config.lr,betas=(0.5,0.999))

        sched_d_opt = ReduceLROnPlateau(d_opt, factor=0.5, patience=50, min_lr=0.00001,verbose=True)
        sched_g_opt = ReduceLROnPlateau(g_opt, factor=0.5, patience=50, min_lr=0.00001,verbose=True)

        return d_opt,g_opt,sched_d_opt,sched_g_opt
    
    def get_losses(self):
        model_config = self.config.model
        if model_config.loss == "vanila":
            d_loss = losses.d_vanilla
            g_loss = losses.g_vanilla
        elif model_config.loss == "ns":
            d_loss = losses.d_vanilla
            g_loss = losses.g_non_saturated
        elif model_config.loss == "wgp":
            d_loss = losses.d_wasserstein
            g_loss = losses.g_wasserstein
        elif model_config.loss == "ls":
            d_loss = losses.d_ls
            g_loss = losses.g_ls

        return d_loss,g_loss