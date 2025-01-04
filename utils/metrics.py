import numpy as np
from scipy.stats import norm, kde
from scipy.integrate import quad

def calc_crps_sample(actual, forecast):
    rng = np.random.default_rng()
    shuffled_forecast = rng.permutation(forecast)
    return np.absolute(forecast - actual).mean(axis=0) - 0.5 * np.absolute(forecast - shuffled_forecast).mean(axis=0)

def calc_quantile_crps(actual, forecast,q_interval):
    quantiles = np.arange(q_interval,1,q_interval)

    actual_rep = np.repeat(np.expand_dims(actual,axis=0),repeats=len(quantiles),axis=0)
    errors = (actual_rep-forecast)
    q_times_e = np.zeros_like(errors)
    qp_times_e = np.zeros_like(errors)
    
    for i,q in enumerate(quantiles):
        q_times_e[i] = q * errors[i]
        qp_times_e[i] = (q-1) * errors[i]
    quantile_losses = np.maximum(q_times_e,qp_times_e)
    return (2*quantile_losses*q_interval).sum(axis=0)
