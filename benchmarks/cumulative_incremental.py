####################################################################
# Imports
####################################################################

import sys

import lifetimes
from lifetimes.datasets import load_transaction_data
from lifetimes.plotting import (
    plot_cumulative_transactions,
    plot_incremental_transactions,
    plot_period_transactions,
    plot_calibration_purchases_vs_holdout_purchases
)

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

####################################################################
# Terminal Variables
####################################################################

filename = sys.argv[1] if len(sys.argv) == 2 else None
# filename = 'CDNOW_master.csv' # 'CDNOW_master.csv' or None

####################################################################
# Loading the Data
####################################################################

transaction_data = load_transaction_data(filename = filename)

if filename == 'CDNOW_master.csv':
    transaction_data = transaction_data[['customer_id', 'date', 'number_of_cds', 'dollar_value']]
    transaction_data['date'] = transaction_data['date'].astype(str)
    transaction_data['date'] = transaction_data['date'].apply(lambda x : x[0:4] + '-' + x[4:6] + '-' + x[6:8])
    transaction_data['date'] = pd.to_datetime(transaction_data['date'])

    calibration_period_end = '1997-09-01' 
    observation_period_end = '1997-12-31'

else:
    calibration_period_end = '2014-07-01'
    observation_period_end = '2014-12-31'

beginning = pd.to_datetime(transaction_data['date'].min())

summary_cal_holdout = lifetimes.utils.calibration_and_holdout_data(
    transactions           = transaction_data, 
    customer_id_col        = 'customer_id' if filename == 'CDNOW_master.csv' else 'id', 
    datetime_col           = 'date',
    calibration_period_end = calibration_period_end,
    observation_period_end = observation_period_end
)

print('Transaction Data Shape:', transaction_data.shape)
print('Cal-Holdout Shape:', '\t', summary_cal_holdout.shape)
print('')

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

plot_path = 'benchmarks/images/'
img_type = '.svg'

plot_cumulative_transactions(
    model           = bgf,
    transactions    = transaction_data,
    datetime_col    = 'date',
    customer_id_col = 'customer_id' if filename == 'CDNOW_master.csv' else 'id',
    t               = t,
    t_cal           = t_cal
)
plt.savefig(plot_path + 'cumulative' + img_type)
plt.close()

plot_incremental_transactions(
    model           = bgf,
    transactions    = transaction_data,
    datetime_col    = 'date',
    customer_id_col = 'customer_id' if filename == 'CDNOW_master.csv' else 'id',
    t               = t,
    t_cal           = t_cal
)
plt.savefig(plot_path + 'incremental' + img_type)
plt.close()

plot_period_transactions(
    model = bgf
)
plt.savefig(plot_path + 'period' + img_type)
plt.close()

plot_calibration_purchases_vs_holdout_purchases(
    model                      = bgf, 
    calibration_holdout_matrix = summary_cal_holdout
)
plt.savefig(plot_path + 'calibration_vs_holdout_purchases' + img_type)
plt.close()

####################################################################
# Extra Plots
####################################################################

# plotting a helper graph to see if everything is ok

transaction_grouped = transaction_data.groupby('date', as_index = False).size()

plt.plot(transaction_grouped)
plt.ylabel('# of transactions')
plt.title('Transactions by Date from the Original Data')
plt.xticks(rotation = 90)
plt.savefig(plot_path + 'grouped_transactions' + img_type)
plt.close()

####################################################################
# Tiny Report on the prob_alive
####################################################################

summary_cal_holdout['prob_alive'] = summary_cal_holdout.apply(
    lambda x : float(bgf.conditional_probability_alive(
        frequency = x.loc['frequency_cal'],
        recency   = x.loc['recency_cal'],
        T         = x.loc['T_cal']
    )), axis = 1
)

total_num_customers = int(summary_cal_holdout['prob_alive'].sum())

print(f'O total de clientes existentes para esse período é de: {summary_cal_holdout.shape[0]}')
print(f'O Total Efetivo de Clientes é de aproximadamente: {total_num_customers}')

plt.hist(summary_cal_holdout['prob_alive'].values, bins = 100)
plt.title('Histogram of the Probability of a Customer Being Alive')
plt.xlabel('probability of being alive')
plt.savefig(plot_path + 'prob_alive' + img_type)
plt.close()