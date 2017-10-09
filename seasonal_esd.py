import scipy.stats as stats
import numpy as np

def calculate_test_statistic(ts):
    zscores = stats.zscore(ts)
    max_idx = np.argmax(zscores)
    return max_idx

def calculate_critical_value(ts, alpha):
    size   = len(ts)
    t_dist = stats.t.sf(alpha / (2 * size), size - 2)
    
    numerator   = (size - 1) * t_dist
    denominator = np.sqrt(size ** 2 - size * 2 + size * t ** 2)


def seasonal_esd(ts, max_anomalies=10, alpha=0.05):
    pass


def esd(ts, max_anomalies=10, alpha=0.05):
