import pandas as pd
from get_values import get_values
from transform_functions import minmax_norm
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import loess

#EEGRaw = pd.read_csv("Raw Data/DataID 174258.TXT", sep=",", header=None)

# Function to calculate LOESS for each column

    
def loess_filter_2(column,span):
    x = [i for i in range(0,len(column))]
    xout, yout, wout = loess.loess_1d(x, column, frac=span)
    return yout


def process_data(EEGRaw):
    EEGdata = get_values(EEGRaw)

    EEGdata = EEGdata.iloc[15:]

    EEGdata.iloc[:, 1:10] = EEGdata.iloc[:, 1:10].apply(minmax_norm)

    # Define the columns to apply LOESS
    vars = EEGdata.columns[2:10]  # Equivalent to EEGdata[3:10] in R

    # Define the ID variable
    id = np.arange(1, len(EEGdata) + 1)

    def loess_filter(column, span):
        result = lowess(endog=column, exog=id, frac=span)
        return result[:, 1]  # Return the smoothed values

    # Apply LOESS to each column
    span = 0.05
    smoothed_data = {col: loess_filter(EEGdata[col], span) for col in vars}

    # Convert the smoothed data back to a DataFrame
    EEGdata_trend = pd.DataFrame(smoothed_data, columns=vars)

    # If you need to keep the same column names as the original
    EEGdata_trend.columns = EEGdata.columns[2:10]

    return EEGdata_trend
