"""
NGC program, implemented in Python for comparison purposes
by Grayson Ohnstad
"""

import sys
import time
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt


def granger_run(n, max_lag):
    '''
    Inputs:
    n is an integer number of time data being used and max_lag is
    the integer (1, 2, 3, or 4) which is used as the maximum lag
    being used by the function grangercausalitytests().
    Returns:
    the compute time and p-value for that particular run
    '''

    # Simulate the independent variable X (e.g., a random walk)
    X = pd.Series(np.cumsum(np.random.randn(n)))

    # Gaussian noise to be used in dependent variable
    e = np.random.normal(0, 1)

    # Simulate an appropriate dependent variable for the given max_lag
    Y = e
    for lag in range(1, max_lag + 1):
        Y += 0.1 * lag * X.shift(lag)

    # Create a DataFrame with the simulated time series data
    data = pd.DataFrame({'X': X, 'Y': Y})

    # Omits NaN values
    data = data.dropna().reset_index(drop=True)

    # Only timing the Granger Causality itself
    start_time = time.time()
    result = grangercausalitytests(data, max_lag, verbose=False)
    end_time = time.time()

    p_value = result[max_lag][0]['ssr_ftest'][1]

    compute_time = end_time - start_time

    return compute_time, p_value


def plotter(main_data, n_max, lag_max):
    '''
    Input:
    pandas DataFrame to be plotted
    Generates:
    plots for analysis
    '''

    # Create subplots for the plots
    _, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot compute time vs number of time data
    axes[0, 0].scatter(main_data['Number of time data'],
                       main_data['Compute time'])
    axes[0, 0].set_xlabel('Number of Time Data')
    axes[0, 0].set_ylabel('Compute Time')
    axes[0, 0].set_title('Compute Time vs Number of Time Data')

    # Adding lines of best fit and r-squared values
    fit_coef1 = np.polyfit(main_data['Number of time data'],
                           main_data['Compute time'], 1)
    best_fit_line1 = np.polyval(fit_coef1, main_data['Number of time data'])
    r_squared1 = np.corrcoef(main_data['Number of time data'],
                             main_data['Compute time'])[0, 1] ** 2
    axes[0, 0].plot(main_data['Number of time data'], best_fit_line1,
                    color='red', label='Best Fit')
    axes[0, 0].text(0.95, 0.05, f'R-squared: {r_squared1:.4f}',
                    transform=axes[0, 0].transAxes, ha='right')

    # Plot compute time vs Max lag
    axes[0, 1].scatter(main_data['Max lag'], main_data['Compute time'])
    axes[0, 1].set_xlabel('Max lag')
    axes[0, 1].set_ylabel('Compute Time')
    axes[0, 1].set_title('Compute Time vs Max lag')

    # Adding lines of best fit and r-squared values
    fit_coef2 = np.polyfit(main_data['Max lag'],
                           main_data['Compute time'], 1)
    best_fit_line2 = np.polyval(fit_coef2, main_data['Max lag'])
    r_squared2 = np.corrcoef(main_data['Max lag'],
                             main_data['Compute time'])[0, 1] ** 2
    axes[0, 1].plot(main_data['Max lag'], best_fit_line2,
                    color='red', label='Best Fit')
    axes[0, 1].text(0.95, 0.05, f'R-squared: {r_squared2:.4f}',
                    transform=axes[0, 1].transAxes, ha='right')

    # Plot p-value vs number of time data
    axes[1, 0].scatter(main_data['Number of time data'], main_data['p-value'])
    axes[1, 0].set_xlabel('Number of Time Data')
    axes[1, 0].set_ylabel('p-value')
    axes[1, 0].set_title('p-value vs Number of Time Data')

    # Adding lines of best fit and r-squared values
    fit_coef3 = np.polyfit(main_data['Number of time data'],
                           main_data['p-value'], 1)
    best_fit_line3 = np.polyval(fit_coef3, main_data['Number of time data'])
    r_squared3 = np.corrcoef(main_data['Number of time data'],
                             main_data['p-value'])[0, 1] ** 2
    axes[1, 0].plot(main_data['Number of time data'], best_fit_line3,
                    color='red', label='Best Fit')
    axes[1, 0].text(0.95, 0.05, f'R-squared: {r_squared3:.4f}',
                    transform=axes[1, 0].transAxes, ha='right')

    # Plot p-value vs Max lag
    axes[1, 1].scatter(main_data['Max lag'], main_data['p-value'])
    axes[1, 1].set_xlabel('Max lag')
    axes[1, 1].set_ylabel('p-value')
    axes[1, 1].set_title('p-value vs Max lag')

    # Adding lines of best fit and r-squared values
    fit_coef4 = np.polyfit(main_data['Max lag'],
                           main_data['p-value'], 1)
    best_fit_line4 = np.polyval(fit_coef4, main_data['Max lag'])
    r_squared4 = np.corrcoef(main_data['Max lag'],
                             main_data['p-value'])[0, 1] ** 2
    axes[1, 1].plot(main_data['Max lag'], best_fit_line4,
                    color='red', label='Best Fit')
    axes[1, 1].text(0.95, 0.05, f'R-squared: {r_squared4:.4f}',
                    transform=axes[1, 1].transAxes, ha='right')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save figure to file
    plt.savefig(f'figure for {n_max} time data and max lag of {lag_max}.png')

    # Show the plots
    plt.show()


def main(n_max, lag_max):
    '''
    Inputs:
    n_max is the highest number of time data to be used in simulation
    lag_max is the largest maximum lag to be used in simulation
    Outpus:
    pandas DataFrame for use in visualization
    '''

    # Setting the random seed, for reproducibility
    np.random.seed(123)

    n_values = list(range(50, n_max + 1, 10))
    table_n = []

    max_lag_values = list(range(1, lag_max + 1))
    table_max_lag = []

    compute_times = []
    p_values = []

    # Builds data for DataFrame using granger_run()
    for n in n_values:
        for max_lag in max_lag_values:

            # Where the magic happens
            current_run = granger_run(n, max_lag)

            # List additions made out of current run
            table_n.append(n)
            table_max_lag.append(max_lag)
            compute_times.append(current_run[0])
            p_values.append(current_run[1])

    # Set up DataFrame for compute time and p-value results
    main_data = pd.DataFrame({'Number of time data': table_n,
                              'Max lag': table_max_lag,
                              'Compute time': compute_times,
                              'p-value': p_values})

    # Export to csv, to compare with R
    main_data.to_csv(f'python generated data with {n_max} time data and max lag of {lag_max}.csv',
                     index=False)

    plotter(main_data, n_max, lag_max)


if __name__ == "__main__":
    # Importing arguments from terminal
    args = sys.argv[1:]

    n_max = 5000
    lag_max = 4

    if len(args) > 0:
        n_max = int(args[0])
        lag_max = int(args[1])

    print(f'Running granger causality with {n_max} time data and max lag of {lag_max}')
    main(n_max, lag_max)
