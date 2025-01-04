import numpy as np
import os
from utils.config import Config
from utils.dataset import Dataset
from utils.metrics import calc_crps_sample
import logging
from models.forgan import ForGAN

def train(forgan: ForGAN, config: Config, dataset: Dataset,logger: logging.Logger):

        d_losses, g_losses, crps = [], [], []
        best_crps = np.inf
        early_stopping_counter = 0
        test_history_window, test_ground_truth = dataset.get_test_set()

        logger.info("Starting training for {} steps.".format(config.train.n_steps))
        
        for step in range(0, config.train.n_steps):
            
            d_loss = []
            for _ in range(config.train.d_iter):
                history_window, ground_truth = dataset.get_random_batch(config.train.batch_size)
                d_loss.append(forgan.train_discriminator(history_window, ground_truth))

            d_losses.append(np.mean(d_loss)/2)
            g_losses.append(forgan.train_generator(history_window))

            if step % config.train.val_interval == 0 and step >= config.train.skip_val:
                val_forecast = forgan.get_forecast(test_history_window,config.train.n_samples,int(1e8))
                crps.append(calc_crps_sample(test_ground_truth, dataset.inverse_transform(val_forecast)).mean())
                forgan.report_progress(crps[-1])

                if crps[-1] <= best_crps:
                    best_crps = crps[-1]
                    forgan.save_model(step, os.path.join(config.train.model_path, "best.torch"), best_crps)
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += config.train.val_interval

                if early_stopping_counter > config.train.early_stopping_tolerance:
                    break

                if step >= config.train.skip_val:
                    logger.info("Step : {:8} , d_Loss : {:4f} , g_Loss : {:4f}, CRPS : {:4e}, Best_CRPS : {:4e}"
                                .format(step, d_losses[-1], g_losses[-1], crps[-1],best_crps))

        forgan.save_model(step, os.path.join(config.train.model_path, "last.torch"), best_crps)
        np.save(os.path.join(config.train.log_path, "d_losses.npy"), np.array(d_losses))
        np.save(os.path.join(config.train.log_path, "g_losses.npy"), np.array(g_losses))
        np.save(os.path.join(config.train.log_path, "crps.npy"), np.array(crps))
        return best_crps