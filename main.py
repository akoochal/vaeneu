import numpy as np
import random
import torch
from utils.config import Config
from utils.data_loader import convert_tsf_to_dataframe
from utils.dataset import Dataset
import os
import importlib
import datetime
import logging
import coloredlogs
import argparse

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)


forecasters = {"forgan":"models.forgan",
            "vaeneu":"models.vaeneu"}

models = {"forgan":"ForGAN",
        "vaeneu":"VAEneu"}

trainers = {"forgan":"trainers.forgan",
            "vaeneu":"trainers.vaeneu"}
'''
 Choice from the following datasets:
 [
        "gold_price_dataset",
        "hepc_dataset",
        "internet_traffic_a1h_dataset",
        "internet_traffic_a5m_dataset",
        "internet_traffic_b1h_dataset",
        "internet_traffic_b5m_dataset",
        "mackey_glass_dataset",
        "saugeenday_dataset",
        "solar_4_seconds_dataset",
        "sunspot_dataset_without_missing_values",
        "us_births_dataset",
        "wind_4_seconds_dataset"
	]
'''

result_path = "./results"

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train a forecasting model.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--device', type=str, required=True)
    args = parser.parse_args()

    dataset = args.dataset
    config_path = args.config_path
    device = args.device

    print("Dataset : ",dataset)
    print("Config Path : ",config_path)
    print("Device : ",device)
    
    
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = Config(config_path,device)
    
    Forecaster = getattr(importlib.import_module(forecasters[config.model_type]),models[config.model_type])
    Trainer = importlib.import_module(trainers[config.model_type])
        
    df, frequency, forecast_horizon, contain_missing_values, contain_equal_length,history_window_size = \
        convert_tsf_to_dataframe(os.path.join("./datasets",dataset+".tsf"))

    unique_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S") 
    config.train.model_path = os.path.join(result_path,dataset,config.model_type+"_"+unique_id)
    config.train.log_path = os.path.join(config.train.model_path,"log")
    os.makedirs(config.train.model_path, exist_ok=True)
    os.makedirs(config.train.log_path, exist_ok=True)
    
    # Set up logging with colored output and specific format
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging.INFO)
    if config.train.screen_log:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
        coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s %(levelname)s %(message)s')
    if config.train.file_log:
        handler = logging.FileHandler(os.path.join(config.train.log_path,"log.txt"))
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
        
    
    config.model.history_window_size = history_window_size
    config.model.horizon = forecast_horizon
    # Number of samples to draw from the forecast distribution during validation
    config.train.n_samples = 1000

    config.auto_config()
    config.save_config(os.path.join(config.train.model_path,"config.yaml"))

    dataset = Dataset(ts_df=df,
                    horizon=config.model.horizon,
                    history_window_size=config.model.history_window_size,
                    device=device)

    forecaster = Forecaster(config)
    logger.info("Config :\n{}\n**********************".format(config))
    if config.model_type == "forgan":
        logger.info("Generator :\n{}\n**********************".format(forecaster.generator))
        logger.info("Discriminator :\n{}\n**********************".format(forecaster.discriminator))
    elif config.model_type == "vaeneu":
        logger.info("Encoder :\n{}\n**********************".format(forecaster.encoder))
        logger.info("Decoder :\n{}\n**********************".format(forecaster.decoder))
    Trainer.train(forecaster, config=config, dataset=dataset,logger=logger)
