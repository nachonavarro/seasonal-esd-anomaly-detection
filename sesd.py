import numpy as np
import scipy.stats as stats
from statsmodels.tsa.seasonal import STL


def calculate_test_statistic(ts, hybrid=False):
    """
    Calculate the test statistic defined by being the top z-score in the time series.

    Args:
        ts (list or np.array): The time series to compute the test statistic.
        hybrid: A flag that determines the type of z-score. See the paper.

    Returns:
        tuple(int, float): The index of the top z-score and the value of the top z-score.

    """
    if hybrid:
        median = np.ma.median(ts)
        mad = np.ma.median(np.abs(ts - median))
        scores = np.abs((ts - median) / mad)
    else:
        scores = np.abs((ts - ts.mean()) / ts.std())
    max_idx = np.argmax(scores)
    return max_idx, scores[max_idx]


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


def seasonal_esd(ts, periodicity=None, hybrid=False, max_anomalies=10, alpha=0.05):
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
    periodicity (int): Number of time points for a season.
    hybrid (bool): See Twitter's research paper for difference.
    max_anomalies (int): The number of times the Grubbs' Test will be applied to the ts.
    alpha (float): The significance level.

    Returns:
    list int: The indices of the anomalies in the timeseries.

    """
    if max_anomalies >= len(ts) / 2:
        raise ValueError('The maximum number of anomalies must be less than half the size of the time series.')

    ts = np.array(ts)
    period = periodicity or int(0.2 * len(ts))  # Seasonality is 20% of the ts if not given.
    stl = STL(ts, period=period, robust=True)
    decomposition = stl.fit()
    residual = ts - decomposition.seasonal - np.median(ts)
    outliers = generalized_esd(residual, max_anomalies=max_anomalies, alpha=alpha, hybrid=hybrid)
    return outliers


def generalized_esd(ts, max_anomalies=10, alpha=0.05, hybrid=False):
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
    ts = np.ma.array(ts)  # A masked array needed to ignore outliers in subsequent ESD tests.
    test_statistics = []
    num_outliers = 0
    for curr in range(max_anomalies):
        test_idx, test_val = calculate_test_statistic(ts, hybrid=hybrid)
        critical_val = calculate_critical_value(len(ts) - curr, alpha)
        if test_val > critical_val:
            num_outliers = curr
        test_statistics.append(test_idx)
        ts[test_idx] = np.ma.masked  # Mask this index so that we don't consider it in subsequent ESD tests.
    anomalous_indices = test_statistics[:num_outliers + 1] if num_outliers > 0 else []
    return anomalous_indices
