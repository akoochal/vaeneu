# VAEneu: A New Avenue for VAE Application on Probabilistic Forecasting

This repository contains the implementation of [**VAEneu**](https://link.springer.com/article/10.1007/s10489-024-06203-5), an innovative method for multistep-ahead univariate probabilistic time series forecasting using Conditional Variational Autoencoders (CVAE). VAEneu optimizes the predictive distribution likelihood with the Continuous Ranked Probability Score (CRPS), resulting in sharp and well-calibrated predictive distributions.

The repository also contains a new version of code for [ForGAN](https://ieeexplore.ieee.org/abstract/document/8717640) which is used in this publication.

Update (1/14/2025) : The light variant of VAEneu model has been added. **Light** is decoder only model trained with CRPS loss. It is faster to train and the performance is similar to VAEneu model.

---

## Table of Contents

- [Enviorment](#enviorment)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Enviorment

Tha code has been test in following container image : nvcr.io/nvidia/pytorch:21.07-py3

Technically , it should work for any newer version of this container image as well.

---

## Training

There are six models presnted in this repository. You can find the config file for these models in config folder. The hyperparameters of the networks are infered from the size of history window automatically as described in the paper.

The folder dataset contains dataset files with .tsf format. These files contain the time series data and also meta data such as target horizon.

To train each of the models, modify and run run.sh file.

```bash
chmod +x run.sh
./run.sh
```

---

## Results

Running the code should result in CRPS similar to following table:

|  | VAEneu-RNN | VAEneu-TCN | ForGAN-RNN | ForGAN-TCN | Light-RNN | Light-TCN |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Gold price | 1.9e+1 | 2.1e+1 | 1.9e+1 | 2.0e+1 | 1.8e+1 | 2.0e+1 |
| HEPC | 3.1e-1 | 2.9e-1 | 3.0e-1 | 3.0e-1 | 3.1e-1 | 2.8e-1 |
| Internet traffic A1H | 5.3e+10 | 5.9e+10 | 8.9e+10 | 7.7e+10 | 5.5e+10 | 6.6e+10 |
| Internet traffic A5M | 1.1e+8 | 1.1e+8 | 9.6e+7 | 1.0e+8 | 1.0e+8 | 9.4e+7 |
| Internet traffic B1H | 5.8e+14 | 5.3e+14 | 1.1e+15 | 5.9e+14 | 5.6e+14 | 5.2e+14 |
| Internet traffic B5M | 4.2e+13 | 3.5e+13 | 4.0e+13 | 3.2e+13 | 4.0e+13 | 3.5e+13 |
| Mackey Glass | 8.6e-4 |4.5e-4 | 4.9e-3 | 3.5e-3 | 8.1e-4 |4.6e-4 |
| Saugeen river | 8.4e+0 | 8.4e+0 | 8.3e+0 | 8.4e+0 | 8.3e+0 | 8.5e+0 |
| Solar 4 seconds | 2.8e+3 | 1.8e+3 | 3.1e+3 | 1.4e+3 | 2.7e+3 | 1.7e+3 |
| Sunspot | 3.4e+0 | 2.7e+0 | 3.6e+0 | 2.7e+0 | 3.5e+0 | 2.8e+0 |
| US births | 2.9e+2 | 2.9e+2 | 2.8e+2 | 3.1e+2 | 2.8e+2 | 2.7e+2 |
| Wind 4 seconds | 6.9e+3 | 7.4e+3 | 6.9e+3 | 6.8e+3 | 7.0e+3 | 7.5e+3 |


---

## License

This project is licensed under the "[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)".

---
