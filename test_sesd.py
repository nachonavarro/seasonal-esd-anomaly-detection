import sesd
import numpy as np
import unittest


class TestSESD(unittest.TestCase):

    def test_sesd_finds_expected_number_of_anomalies(self):
        ts = np.random.random(1000)
        # Introduce 4 artificial anomalies
        expected_anomaly_indices = [14, 83, 250, 540]
        for idx in expected_anomaly_indices:
            ts[idx] = 100

        outliers_indices = sesd.seasonal_esd(ts, hybrid=True, max_anomalies=4, periodicity=50)
        for idx in outliers_indices:
            self.assertIn(idx, expected_anomaly_indices)

    def test_sesd_does_not_find_more_anomalies_than_max_allowed(self):
        ts = np.random.random(1000)
        # Introduce 4 artificial anomalies
        expected_anomaly_indices = [14, 83, 250, 540]
        for idx in expected_anomaly_indices:
            ts[idx] = 100

        outliers_indices = sesd.seasonal_esd(ts, hybrid=True, max_anomalies=3, periodicity=50)

        self.assertEqual(3, len(outliers_indices))

        for idx in outliers_indices:
            self.assertIn(idx, expected_anomaly_indices)

    def test_sesd_does_not_find_anomalies(self):
        ts = np.arange(1, 10)  # Make a linear time series, in which there should be no anomalies.
        outliers_indices = sesd.seasonal_esd(ts, hybrid=True, max_anomalies=3, periodicity=3)
        self.assertEqual(0, len(outliers_indices))
