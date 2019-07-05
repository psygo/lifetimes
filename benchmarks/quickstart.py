####################################################################
# Documentation
####################################################################

# This is supposed to be the code represented in the 
# Quickstart docs page.

# The `#%%` signs are the VS Code way of creating a Jupyter Notebook.
# Usually, they should be ignored as comments, if executed as normal
# Python files.

####################################################################
# Initial Imports
####################################################################

#%%

import lifetimes

####################################################################
# Quickstart
####################################################################

#%%

####################################################################
# Loading the Data
####################################################################

from lifetimes.datasets import load_cdnow_summary

data = load_cdnow_summary(index_col = [0])

print(data.head())

#%%

####################################################################
# Initial Fit
####################################################################

from lifetimes import BetaGeoFitter

# similar API to scikit-learn and lifelines.
bgf = BetaGeoFitter(penalizer_coef = 0.0)
bgf.fit(
    data['frequency'], 
    data['recency'], 
    data['T']
)

print(bgf.summary)

#%%

####################################################################
# Visualizing our Frequency/Recency Matrix
####################################################################

from lifetimes.plotting import plot_frequency_recency_matrix

plot_frequency_recency_matrix(bgf)

#%%

####################################################################
# Probability Alive Matrix
####################################################################

from lifetimes.plotting import plot_probability_alive_matrix

plot_probability_alive_matrix(bgf)

#%%

####################################################################
# Ranking customers from best to worst
####################################################################

t = 1
data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    t, 
    data['frequency'], 
    data['recency'], 
    data['T']
)

data.sort_values(by='predicted_purchases').tail(5)

#%%

####################################################################
# Assessing model fit
####################################################################

from lifetimes.plotting import plot_period_transactions

plot_period_transactions(bgf)

#%%

####################################################################
# Example using transactional datasets
####################################################################

from lifetimes.datasets import load_transaction_data
from lifetimes.utils import summary_data_from_transaction_data

transaction_data = load_transaction_data()

print(transaction_data.head())

summary = summary_data_from_transaction_data(
    transaction_data, 
    'id', 
    'date', 
    observation_period_end = '2014-12-31'
)

print(summary.head())

# #%%

# ####################################################################
# # Extra: Cumulative Plot
# ####################################################################

# from lifetimes.plotting import plot_cumulative_transactions

# plot_cumulative_transactions(
#     model = bgf,
#     transactions = transaction_data,
#     datetime_col = 'date',
#     customer_id_col = 'id',
#     t = 200,
#     t_cal = 100
# )

# #%%

# ####################################################################
# # Extra: Incremental Plot
# ####################################################################

# from lifetimes.plotting import plot_incremental_transactions

# plot_incremental_transactions(
#     model = bgf,
#     transactions = transaction_data,
#     datetime_col = 'date',
#     customer_id_col = 'id',
#     t = 200,
#     t_cal = 100
# )

#%%

####################################################################
# More model fitting
####################################################################

from lifetimes.utils import calibration_and_holdout_data

summary_cal_holdout = calibration_and_holdout_data(
    transaction_data, 
    'id', 
    'date',
    calibration_period_end = '2014-09-01',
    observation_period_end = '2014-12-31'
)

print(summary_cal_holdout.head())

from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

bgf.fit(
    summary_cal_holdout['frequency_cal'], 
    summary_cal_holdout['recency_cal'], 
    summary_cal_holdout['T_cal']
)

plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout)

#%%

####################################################################
# Customer Predictions
####################################################################

t = 10 #predict purchases in 10 periods
individual = summary.iloc[20]
# The below function is an alias to `bfg.conditional_expected_number_of_purchases_up_to_time`
bgf.predict(
    t, 
    individual['frequency'], 
    individual['recency'], 
    individual['T']
)

#%%

####################################################################
# Customer Probability Histories
####################################################################

from lifetimes.plotting import plot_history_alive

id = 35
days_since_birth = 200
sp_trans = transaction_data.loc[transaction_data['id'] == id]

plot_history_alive(bgf, days_since_birth, sp_trans, 'date')

#%%

####################################################################
# Estimating customer lifetime value using the Gamma-Gamma model
####################################################################

from lifetimes.datasets import load_cdnow_summary_data_with_monetary_value

summary_with_money_value = load_cdnow_summary_data_with_monetary_value()
summary_with_money_value.head()
returning_customers_summary = summary_with_money_value[summary_with_money_value['frequency'] > 0]

print(returning_customers_summary.head())

#%%

####################################################################
# The Gamma-Gamma model and the independence assumption
####################################################################

returning_customers_summary[['monetary_value', 'frequency']].corr()

from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef = 0)

ggf.fit(
    returning_customers_summary['frequency'],
    returning_customers_summary['monetary_value']
)

print(ggf)

print(
    ggf.conditional_expected_average_profit(
        summary_with_money_value['frequency'],
        summary_with_money_value['monetary_value']
    ).head(10)
)

#%%

print("Expected conditional average profit: %s, Average profit: %s" % (
    ggf.conditional_expected_average_profit(
        summary_with_money_value['frequency'],
        summary_with_money_value['monetary_value']
    ).mean(),
    summary_with_money_value[summary_with_money_value['frequency']>0]['monetary_value'].mean()
))

#%%

bgf.fit(summary_with_money_value['frequency'], summary_with_money_value['recency'], summary_with_money_value['T'])

print(ggf.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    summary_with_money_value['frequency'],
    summary_with_money_value['recency'],
    summary_with_money_value['T'],
    summary_with_money_value['monetary_value'],
    time=12, # months
    discount_rate=0.01 # monthly discount rate ~ 12.7% annually
).head(10))