#!/bin/bash

#  Choice from the following datasets:
#  [
#         "gold_price_dataset",
#         "hepc_dataset",
#         "internet_traffic_a1h_dataset",
#         "internet_traffic_a5m_dataset",
#         "internet_traffic_b1h_dataset",
#         "internet_traffic_b5m_dataset",
#         "mackey_glass_dataset",
#         "saugeenday_dataset",
#         "solar_4_seconds_dataset",
#         "sunspot_dataset_without_missing_values",
#         "us_births_dataset",
#         "wind_4_seconds_dataset"
# 	]

python3 ./main.py --config_path ./configs/vaeneu_rnn.yaml --dataset gold_price_dataset
