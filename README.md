# Anomaly Detection: Seasonal ESD

Note: All credit goes to Jordan Hochenbaum, Owen S. Vallis and Arun Kejariwa at Twitter, Inc. Any errors in the code are, of course, my mistake. Feel free to fix them.

## Intro
Seasonal ESD is an anomaly detection algorithm implemented at Twitter https://arxiv.org/pdf/1704.07706.pdf. What better definition than the one they use in their paper:

> "we developed two novel statistical techniques
> for automatically detecting anomalies in cloud infrastructure
> data. Specifically, the techniques employ statistical learning
> to detect anomalies in both application, and system metrics.
> Seasonal decomposition is employed to filter the trend and
> seasonal components of the time series, followed by the use
> of robust statistical metrics – median and median absolute
> deviation (MAD) – to accurately detect anomalies, even in
> the presence of seasonal spikes."

## Installation

To install `sesd`, use pip:

```python
pip install sesd
```


### Explanation
The algorithm uses the Extreme Studentized Deviate test to calculate the anomalies. In fact, the novelty doesn't come
in the fact that ESD is used, but rather on _what_ it is tested.

The problem with the ESD test on its own is that it assumes a normal data distribution, while real world data can have a multimodal distribution. To circumvent this, STL decomposition is used. Any time series can be decomposed with STL decomposition into a seasonal, trend, and residual component. The key is that the residual has a unimodal distribution that ESD can test. 

However, there is still the problem that extreme, spurious anomalies can corrupt the residual component. To fix it, the paper proposes to use the median to represent the "stable" trend, instead of the trend found by means of STL decomposition.

Finally, for data sets that have a high percentage of anomalies, the research papers proposes to use the median and Median Absolute Deviate (MAD) instead of the mean and standard deviation to compute the zscore. Using MAD enables a more consistent measure of central tendency of a time series with a high percentage of anomalies.

---

## Usage

```python
import numpy as np
import sesd
ts = np.random.random(100)
# Introduce artificial anomalies
ts[14] = 9
ts[83] = 10
outliers_indices = sesd.seasonal_esd(ts, hybrid=True, max_anomalies=2)
for idx in outliers_indices:
    print("Anomaly index: {}, anomaly value: {}".format(idx, ts[idx]))

>>> Anomaly index: 83, anomaly value: 10.0
>>> Anomaly index: 14, anomaly value: 9.0
```

--- 

## Documentation


* `seasonal_esd(seasonality=None, hybrid=False, max_anomalies=10, alpha=0.05)`: Computes the Seasonal Extreme Studentized Deviate of a time series. The steps taken are first to to decompose the time series into STL decomposition (trend, seasonality, residual). Then, calculate the Median Absolute Deviate (MAD) if hybrid (otherwise the median) and perform a regular ESD test on the residual, which we calculate as: `R = ts - seasonality - MAD or median.

    * Arguments

        * `ts`: The time series to compute the SESD.
        * `seasonality`: The statsmodel library requires a seasonality to compute the STL decomposition If none is given, then it will automatically be calculated to be 20% of the total time series.
        * `hybrid`: See Twitter’s research paper for the difference.
        max_anomalies: The number of times the Grubbs’ Test will be applied to the time series.
        * `alpha`: the significance level.
    
    * Returns

        * The indices of the anomalies in the time series.

* `esd(timeseries, max_anomalies=10, alpha=0.05, hybrid=False)`: Computes the Extreme Studentized Deviate of a time series. A Grubbs Test is performed max_anomalies times with the caveat that each time the top value is removed. For more details visit http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

    * Arguments

        * `ts`: The time series to compute the ESD.
        max_anomalies: The number of times the Grubbs’ Test will be applied to the time series.
        * `alpha`: the significance level.
        * `hybrid`: If set to false then the mean and standard deviation will be used to calculate the zscores in the Grubbs test. If set to true, then median and MAD will be used.
    
    * Returns

        * The indices of the anomalies in the time series.

