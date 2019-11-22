import numpy as np
import scipy.stats as stats
from stldecompose import decompose


def calculate_zscore(ts, hybrid=False):
    if hybrid:
        median = np.ma.median(ts)
        mad = np.ma.median(np.abs(ts - median))
        return (ts - median) / mad
    else:
        return stats.zscore(ts, ddof=1)


def calculate_test_statistic(ts, test_statistics, hybrid=False):
    """
    Calculate the test statistic defined by being the top z-score in the time series.

    Args:
        ts (list or np.array): The time series to compute the test statistic.
        test_statistics: The test statistics
        hybrid: A flag that determines the type of z-score. See the paper.

    Returns:
        tuple(int, float): The index of the top z-score and the value of the top z-score.

    """
    corrected_ts = np.ma.array(ts, mask=False)
    for anomalous_index in test_statistics:
        corrected_ts.mask[anomalous_index] = True
    z_scores = abs(calculate_zscore(corrected_ts, hybrid=hybrid))
    max_idx = np.argmax(z_scores)
    return max_idx, z_scores[max_idx]


def calculate_critical_value(size, alpha):
    """
    Calculate the critical value with the formula given for example in
    https://en.wikipedia.org/wiki/Grubbs%27_test_for_outliers#Definition

    Args:
        size: The current size of the time series
        alpha (float): The significance level.

    Returns:
        float: The critical value for this test.

    """
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)

    numerator = (size - 1) * t_dist
    denominator = np.sqrt(size ** 2 - size * 2 + size * t_dist ** 2)

    return numerator / denominator


def seasonal_esd(ts, seasonality=None, hybrid=False, max_anomalies=10, alpha=0.05):
    """
    Compute the Seasonal Extreme Studentized Deviate of a time series.
    The steps taken are first to to decompose the time series into STL
    decomposition (trend, seasonality, residual). Then, calculate
    the Median Absolute Deviate (MAD) if hybrid (otherwise the median)
    and perform a regular ESD test on the residual, which we calculate as:
                    R = ts - seasonality - MAD or median

    Note: The statsmodel library requires a seasonality to compute the STL
    decomposition, hence the parameter seasonality. If none is given,
    then it will automatically be calculated to be 20% of the total
    timeseries.

    Args:
    ts (list or np.array): The timeseries to compute the ESD.
    seasonality (int): Number of time points for a season.
    hybrid (bool): See Twitter's research paper for difference.
    max_anomalies (int): The number of times the Grubbs' Test will be applied to the ts.
    alpha (float): The significance level.

    Returns:
    list int: The indices of the anomalies in the timeseries.

    """
    ts = np.array(ts)
    seasonal = seasonality or int(0.2 * len(ts))  # Seasonality is 20% of the ts if not given.
    decomposition = decompose(ts, period=seasonal)
    residual = ts - decomposition.seasonal - np.median(ts)
    outliers = esd(residual, max_anomalies=max_anomalies, alpha=alpha, hybrid=hybrid)
    return outliers


def esd(ts, max_anomalies=10, alpha=0.05, hybrid=False):
    """
    Compute the Extreme Studentized Deviate of a time series.
    A Grubbs Test is performed max_anomalies times with the caveat
       that each time the top value is removed. For more details visit
       http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

    Args:
        ts (list or np.array): The time series to compute the ESD.
        max_anomalies (int): The number of times the Grubbs' Test will be applied to the ts.
        alpha (float): The significance level.
        hybrid: A flag that determines the type of z-score. See the paper.

    Returns:
        list int: The indices of the anomalies in the time series.

    """
    ts = np.copy(np.array(ts))
    test_statistics = []
    total_anomalies = 0
    for curr in range(max_anomalies):
        test_idx, test_val = calculate_test_statistic(ts, test_statistics, hybrid=hybrid)
        critical_value = calculate_critical_value(len(ts) - len(test_statistics), alpha)
        if test_val > critical_value:
            total_anomalies = curr
        test_statistics.append(test_idx)
    anomalous_indices = test_statistics[:total_anomalies + 1] if total_anomalies else []
    return anomalous_indices
