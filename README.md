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

### Explanation
The algorithm uses the Extreme Studentized Deviate test to calculate the anomalies. In fact, the novelty doesn't come
in the fact that ESD is used, but rather on _what_ it is tested.

The problem with the ESD test on its own is that it assumes a normal data distribution, while real world data can have a multimodal distribution. To circumvent this, STL decomposition is used. Any time series can be decomposed with STL decomposition into a seasonal, trend, and residual component. The key is that the residual has a unimodal distribution that ESD can test. 

However, there is still the problem that extreme, spurious anomalies can corrupt the residual component. To fix it, the paper proposes to use the median to represent the "stable" trend, instead of the trend found by means of STL decomposition.

Finally, for data sets that have a high percentage of anomalies, the research papers proposes to use the Median Absolute Deviate (MAD) instead of the median when computing the residual. Using MAD enables a more consistent measure of central tendency of a time series with a high percentage of anomalies. 

---

## Usage

```python
import numpy as np
import seasonal_esd as sesd
ts = np.random.random(100)
# Introduce artificial anomalies
ts[14] = 9
ts[83] = 10
outliers_indices = sesd.seasonal_esd(ts, hybrid=True, max_anomalies=2)
for idx in outliers_indices:
	print "Anomaly index: {0}, anomaly value: {1}".format(idx, ts[idx])

>>> Anomaly index: 83, anomaly value: 10.0
>>> Anomaly index: 14, anomaly value: 9.0
```
