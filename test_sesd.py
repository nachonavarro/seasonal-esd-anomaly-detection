import sesd
import numpy as np

def test_sesd_finds_expected_number_of_anomalies():
    ts = np.random.random(1000)
    # Introduce 4 artificial anomalies
    expected_anomaly_indices = [14, 83, 250, 540]
    for idx in expected_anomaly_indices:
        ts[idx] = 100

    outliers_indices = sesd.seasonal_esd(ts, hybrid=True, max_anomalies=4)
    for idx in outliers_indices:
        assert idx in expected_anomaly_indices

def test_sesd_does_not_find_more_anomalies_than_max_allowed():
    ts = np.random.random(1000)
    # Introduce 4 artificial anomalies
    expected_anomaly_indices = [14, 83, 250, 540]
    for idx in expected_anomaly_indices:
        ts[idx] = 100
    
    max_anomalies = 3
    outliers_indices = sesd.seasonal_esd(ts, hybrid=True, max_anomalies=max_anomalies)

    assert len(outliers_indices) == max_anomalies

    for idx in outliers_indices:
        assert idx in expected_anomaly_indices

def test_sesd_does_not_find_anomalies():
    ts = np.random.random(1000)
    outliers_indices = sesd.seasonal_esd(ts, hybrid=True, max_anomalies=3)
    assert len(outliers_indices) == 0