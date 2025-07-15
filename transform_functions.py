import numpy as np

# Skewness Correction Functions:
# 1. Square Root Transformation
def sqrt_normal(x):
    return np.sqrt(x)

# 2. Natural Log Transformation
def ln_normal(x):
    return np.log(x + 1)  # adding 1 here to deal with zero values

# 3. Inverse Square Root Transformation
def inv_sqrt_norm(x):
    return 1 / np.sqrt(x + 1)

# Normalization Functions:
# 1. Min-Max Normalization Function
#   values will be between 0 and 1
def minmax_norm(x):
    return ((x - np.min(x)) / (np.max(x) - np.min(x)))

# 2. Z-score Standardization
#   neg values are below the mean
#   pos values are above the mean
def zscore_scale(x):
    m = np.mean(x)
    s = np.std(x, ddof=0)  # population standard deviation
    zscore = (x - m) / s
    return zscore
