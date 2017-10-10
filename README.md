# Anomaly Detection: Seasonal ESD

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
