import numpy as np
import os
from utils.config import Config
from utils.dataset import Dataset
from utils.metrics import calc_crps_sample
import logging
from models.vaeneu import VAEneu

def train(vaeneu: VAEneu, config: Config, dataset: Dataset,logger: logging.Logger):

        kld_losses, recon_losses, crps = [], [], []
        best_crps = np.inf
        early_stopping_counter = 0
        test_history_window, test_ground_truth = dataset.get_test_set()

        logger.info("Starting training for {} steps.".format(config.train.n_steps))

        for step in range(0, config.train.n_steps):
            
            history_window, ground_truth = dataset.get_random_batch(config.train.batch_size)
            recon_loss,kld_loss = vaeneu.train(history_window, ground_truth)
            recon_losses.append(recon_loss)
            kld_losses.append(kld_loss)

            if step % config.train.val_interval == 0 and step >= config.train.skip_val:
                val_forecast = vaeneu.get_forecast(test_history_window,config.train.n_samples,int(1e4))
                crps.append(calc_crps_sample(test_ground_truth, dataset.inverse_transform(val_forecast)).mean())
                vaeneu.report_progress(crps[-1])

                if crps[-1] <= best_crps:
                    best_crps = crps[-1]
                    vaeneu.save_model(step, os.path.join(config.train.model_path, "best.torch"), best_crps)
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += config.train.val_interval

                if early_stopping_counter > config.train.early_stopping_tolerance:
                    break

                if step >= config.train.skip_val:
                    logger.info("Step : {:8} , recon_Loss : {:4f} , kld_Loss : {:4f}, CRPS : {:4e}, Best_CRPS : {:4e}"
                                .format(step, recon_losses[-1], kld_losses[-1], crps[-1],best_crps))
                    
        vaeneu.save_model(step, os.path.join(config.train.model_path, "last.torch"), best_crps)
        np.save(os.path.join(config.train.log_path, "recon_losses.npy"), np.array(recon_losses))
        np.save(os.path.join(config.train.log_path, "kld_losses.npy"), np.array(kld_losses))
        np.save(os.path.join(config.train.log_path, "crps.npy"), np.array(crps))
        return best_crps
