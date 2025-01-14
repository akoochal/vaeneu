import numpy as np
import os
from utils.config import Config
from utils.dataset import Dataset
from utils.metrics import calc_crps_sample
import logging
from models.light import Light

def train(light: Light, config: Config, dataset: Dataset,logger: logging.Logger):

        losses, crps = [], []
        best_crps = np.inf
        early_stopping_counter = 0
        test_history_window, test_ground_truth = dataset.get_test_set()

        logger.info("Starting training for {} steps.".format(config.train.n_steps))

        for step in range(0, config.train.n_steps):
            
            history_window, ground_truth = dataset.get_random_batch(config.train.batch_size)
            loss = light.train(history_window, ground_truth)
            losses.append(loss)

            if step % config.train.val_interval == 0 and step >= config.train.skip_val:
                val_forecast = light.get_forecast(test_history_window,config.train.n_samples,int(1e4))
                crps.append(calc_crps_sample(test_ground_truth, dataset.inverse_transform(val_forecast)).mean())
                light.report_progress(crps[-1])

                if crps[-1] <= best_crps:
                    best_crps = crps[-1]
                    light.save_model(step, os.path.join(config.train.model_path, "best.torch"), best_crps)
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += config.train.val_interval

                if early_stopping_counter > config.train.early_stopping_tolerance:
                    break

                if step >= config.train.skip_val:
                    logger.info("Step : {:8} , Loss : {:4f} , CRPS : {:4e}, Best_CRPS : {:4e}"
                                .format(step, losses[-1], crps[-1],best_crps))
                    
        light.save_model(step, os.path.join(config.train.model_path, "last.torch"), best_crps)
        np.save(os.path.join(config.train.log_path, "losses.npy"), np.array(losses))
        np.save(os.path.join(config.train.log_path, "crps.npy"), np.array(crps))
        return best_crps
