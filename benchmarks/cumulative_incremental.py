# Imports

import lifetimes
from lifetimes.datasets import load_transaction_data
from lifetimes.plotting import plot_cumulative_transactions
from lifetimes.plotting import plot_incremental_transactions

import pandas as pd

import matplotlib.pyplot as plt

####################################################################
# Loading the Data
####################################################################

# This function uses the `example_transactions.csv` file currently.
transaction_data = load_transaction_data()

beginning = pd.to_datetime(transaction_data['date'].min())
calibration_period_end = '2014-09-01'
observation_period_end = '2014-12-31'

summary_cal_holdout = lifetimes.utils.calibration_and_holdout_data(
    transactions           = transaction_data, 
    customer_id_col        = 'id', 
    datetime_col           = 'date',
    calibration_period_end = '2014-09-01',
    observation_period_end = '2014-12-31'
)

####################################################################
# Fitting the Model
####################################################################

bgf = lifetimes.BetaGeoFitter(
    penalizer_coef = 0.0
)

bgf.fit(
    summary_cal_holdout['frequency_cal'], 
    summary_cal_holdout['recency_cal'], 
    summary_cal_holdout['T_cal']
)

print(bgf.summary)

####################################################################
# Plotting
####################################################################

t = 300
t_cal = (pd.to_datetime(calibration_period_end) - beginning).days

plot_path = 'benchmarks/'

plot_cumulative_transactions(
    model           = bgf,
    transactions    = transaction_data,
    datetime_col    = 'date',
    customer_id_col = 'id',
    t               = t,
    t_cal           = t_cal
)
plt.savefig(plot_path + 'cumulative.svg')
plt.close()

plot_incremental_transactions(
    model           = bgf,
    transactions    = transaction_data,
    datetime_col    = 'date',
    customer_id_col = 'id',
    t               = t,
    t_cal           = t_cal
)
plt.savefig(plot_path + 'incremental.svg')
plt.close()