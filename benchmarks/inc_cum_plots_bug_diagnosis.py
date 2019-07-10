####################################################################
# Imports
####################################################################

import sys

import lifetimes
from lifetimes.datasets import load_transaction_data
from lifetimes.plotting import (
    plot_cumulative_transactions,
    plot_incremental_transactions
)
from lifetimes.utils import expected_cumulative_transactions

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

print('')

####################################################################
# Documentation
####################################################################

# The purpose of this file is to diagnose the problems with the
# Incremental and Cumulative Plots.

# Mainly, they seem to predict the behavior of the data all too well.
# After much investigation, it was found out that they were using
# the full transactions data instead of only the calibration data.

####################################################################
# Terminal and Variables
####################################################################

# Terminal Options:

# 1. filename       : default or CDNOW_master
# 2. penalizer_coef : number for the penalizer of the model's fit

filename = sys.argv[1]
penalizer_coef = float(sys.argv[2])

# How to execute this (e.g.): 
# `python benchmarks/inc_cum_plots_bug_diagnosis.py CDNOW_master.csv 0.0`

####################################################################
# Loading the Data
####################################################################

transaction_data = load_transaction_data(
    filename = filename if filename != 'default' else None
)

if filename == 'CDNOW_master.csv':
    transaction_data = transaction_data[['customer_id', 'date', 'number_of_cds', 'dollar_value']]
    transaction_data['date'] = transaction_data['date'].astype(str)
    transaction_data['date'] = transaction_data['date'].apply(lambda x : x[0:4] + '-' + x[4:6] + '-' + x[6:8])
    transaction_data['date'] = pd.to_datetime(transaction_data['date'])

    calibration_period_end = '1997-09-01' 
    observation_period_end = '1997-12-31'
elif filename == 'default': # artificial default dataset
    calibration_period_end = '2014-08-01'
    observation_period_end = '2014-12-31'

print(transaction_data.head())
print('')

transaction_data['date'] = pd.to_datetime(transaction_data['date'])

beginning = pd.to_datetime(transaction_data['date'].min())

print('Initial Date: {}\nLast Date:{}\nNumber of days: {}'.format(
    transaction_data['date'].min(), 
    transaction_data['date'].max(),
    transaction_data['date'].max() - transaction_data['date'].min()
))
print('')

####################################################################
# Forcing
####################################################################

# Forcing purchases on the holdout period to be next to zero.
# This way there won't be a way for the graph to know what's on the holdout data.

# This part should usually not be used. 
# If you wish, you can enable the `force` parameter.

force = False
if force == True:
    saved_transaction_data = transaction_data
    all_dates = saved_transaction_data['date']
    transaction_data = transaction_data[transaction_data['date'] <= calibration_period_end]

    for day in all_dates:
        day = pd.to_datetime(day)
        if day > pd.to_datetime(calibration_period_end):
            id_col = 'customer_id' if filename == 'CDNOW_master.csv' else 'id'
            # Adding just 1 purchase for 1 customer to keep the date in.
            transaction_data = transaction_data.append(
                {'date' : day, id_col : 9999},
                ignore_index = True
            )

####################################################################
# RFM Model
####################################################################

summary_cal_holdout = lifetimes.utils.calibration_and_holdout_data(
    transactions           = transaction_data, 
    customer_id_col        = 'customer_id' if filename == 'CDNOW_master.csv' else 'id', 
    datetime_col           = 'date',
    calibration_period_end = calibration_period_end,
    observation_period_end = observation_period_end
)

print('Transaction Data Shape: {}'.format(transaction_data.shape))
print('Cal-Holdout Shape: \t {}'.format(summary_cal_holdout.shape))
print('')

####################################################################
# Fitting the Model
####################################################################

bgf = lifetimes.BetaGeoFitter(
    penalizer_coef = penalizer_coef
)

bgf.fit(
    summary_cal_holdout['frequency_cal'], 
    summary_cal_holdout['recency_cal'], 
    summary_cal_holdout['T_cal']
)

print(bgf.summary.round(4)) # ParetoNBDFitter doesn't have a summary yet.
print('')

####################################################################
# Plotting
####################################################################

t = 400 # number of days on the plot
t_cal = (pd.to_datetime(calibration_period_end) - beginning).days

plot_path = 'benchmarks/images/'
img_type = '.svg'

####################################################################
# Older (incorrect) Version of the Plots
####################################################################

# Previous plots. 
# Only used to prove that they were wrong. If you're on a version that
# is already correct, don't bother to uncomment this.

# plot_cumulative_transactions(
#     model           = bgf,
#     transactions    = transaction_data,
#     datetime_col    = 'date',
#     customer_id_col = 'customer_id' if filename == 'CDNOW_master.csv' else 'id',
#     t               = t,
#     t_cal           = t_cal
# )
# plt.savefig(plot_path + 'cumulative_old' + img_type)
# plt.close()

# plot_incremental_transactions(
#     model           = bgf,
#     transactions    = transaction_data,
#     datetime_col    = 'date',
#     customer_id_col = 'customer_id' if filename == 'CDNOW_master.csv' else 'id',
#     t               = t,
#     t_cal           = t_cal
# )
# plt.savefig(plot_path + 'incremental_old' + img_type)
# plt.close()

####################################################################
# Local Version of the Cumulative Transactions
####################################################################

def plot_cumulative_transactions(
    model,
    transactions,
    calibration_period_end,
    datetime_col,
    customer_id_col,
    t,
    t_cal,
    datetime_format=None,
    freq="D",
    set_index_date=False,
    title="Tracking Cumulative Transactions",
    xlabel="day",
    ylabel="Cumulative Transactions",
    legend=['actual', 'model on all data', 'model on calibration data'],
    ax=None,
    **kwargs
):

    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.subplot(111)

    # Using only the purchases on the calibration period:
    holdout_transactions = transactions[transactions['date'] > calibration_period_end]
    cal_transactions = transactions[transactions['date'] <= calibration_period_end]

    df_cum_transactions = expected_cumulative_transactions(
        model,
        transactions,
        datetime_col,
        customer_id_col,
        t,
        datetime_format=datetime_format,
        freq=freq,
        set_index_date=set_index_date,
    )

    df_cum_transactions_cal = expected_cumulative_transactions(
        model,
        cal_transactions,
        datetime_col,
        customer_id_col,
        t,
        datetime_format=datetime_format,
        freq=freq,
        set_index_date=set_index_date,
    )

    df_cum_transactions.plot(ax=ax, title=title, color=['royalblue', 'orange'], **kwargs)
    df_cum_transactions_cal['predicted'].plot(ax=ax, title=title, color=['red'], **kwargs)

    if set_index_date:
        x_vline = df_cum_transactions.index[int(t_cal)]
        xlabel = "date"
    else:
        x_vline = t_cal
    ax.axvline(x=x_vline, color="r", linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(legend, loc = 'upper left')

    return ax

# New plot call:

plot_cumulative_transactions(
    model                  = bgf,
    transactions           = transaction_data,
    calibration_period_end = calibration_period_end,
    datetime_col           = 'date',
    customer_id_col        = 'customer_id' if filename == 'CDNOW_master.csv' else 'id',
    t                      = t,
    t_cal                  = t_cal
)
plt.savefig(plot_path + 'cumulative_new' + img_type)
plt.close()

####################################################################
# Local Version of the Incremental Transactions
####################################################################

def plot_incremental_transactions(
    model,
    transactions,
    calibration_period_end,
    datetime_col,
    customer_id_col,
    t,
    t_cal,
    datetime_format=None,
    freq="D",
    set_index_date=False,
    title="Tracking Daily Transactions",
    xlabel="day",
    ylabel="Transactions",
    legend=['actual', 'model on all data', 'model on calibration data'],
    ax=None,
    **kwargs
):

    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.subplot(111)

    # Using only the purchases on the calibration period:
    holdout_transactions = transactions[transactions['date'] > calibration_period_end]
    cal_transactions = transactions[transactions['date'] <= calibration_period_end]

    df_cum_transactions = expected_cumulative_transactions(
        model,
        transactions,
        datetime_col,
        customer_id_col,
        t,
        datetime_format=datetime_format,
        freq=freq,
        set_index_date=set_index_date,
    )

    df_cum_transactions_cal = expected_cumulative_transactions(
        model,
        cal_transactions,
        datetime_col,
        customer_id_col,
        t,
        datetime_format=datetime_format,
        freq=freq,
        set_index_date=set_index_date,
    )

    # get incremental from cumulative transactions
    df_inc_transactions = df_cum_transactions.apply(lambda x: x - x.shift(1))
    df_inc_transactions_cal = df_cum_transactions_cal.apply(lambda x: x - x.shift(1))

    df_inc_transactions.plot(ax=ax, title=title, color=['royalblue', 'orange'], **kwargs)
    df_inc_transactions_cal['predicted'].plot(ax=ax, title=title, color=['red'], **kwargs)

    if set_index_date:
        x_vline = df_inc_transactions.index[int(t_cal)]
        xlabel = "date"
    else:
        x_vline = t_cal
    ax.axvline(x=x_vline, color="r", linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(legend, loc = 'upper left')

    return ax

# New plot call:

plot_incremental_transactions(
    model                  = bgf,
    transactions           = transaction_data,
    calibration_period_end = calibration_period_end,
    datetime_col           = 'date',
    customer_id_col        = 'customer_id' if filename == 'CDNOW_master.csv' else 'id',
    t                      = t,
    t_cal                  = t_cal
)
plt.savefig(plot_path + 'incremental_new' + img_type)
plt.close()