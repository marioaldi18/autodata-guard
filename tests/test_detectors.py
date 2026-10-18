import unittest

import numpy as np

from autodataguard.config import resolve_config
from autodataguard.models.detectors import IsolationForestDetector, LOFDetector, RobustZScoreDetector, build_detectors


def data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(500, 4))
    return X, np.vstack([X[:5], [[12.0, 12.0, 12.0, 12.0]]])


class DetectorTests(unittest.TestCase):
    def test_outlier_gets_highest_score(self):
        X, Xt = data()
        for det in (RobustZScoreDetector(), IsolationForestDetector(0, 100), LOFDetector(20)):
            s = det.fit(X).score(Xt)
            self.assertEqual(s.shape, (6,))
            self.assertEqual(int(np.argmax(s)), 5, det.name)

    def test_constant_column_does_not_break_zscore(self):
        X, Xt = data()
        X[:, 0] = 1.0
        Xt[:, 0] = 1.0
        s = RobustZScoreDetector().fit(X).score(Xt)
        self.assertTrue(np.isfinite(s).all())

    def test_seeded_isolation_forest_is_deterministic(self):
        X, Xt = data()
        a = IsolationForestDetector(3, 50).fit(X).score(Xt)
        b = IsolationForestDetector(3, 50).fit(X).score(Xt)
        np.testing.assert_allclose(a, b)

    def test_build_detectors_names(self):
        names = [d.name for d in build_detectors(resolve_config())]
        self.assertEqual(names, ["robust_zscore_baseline", "isolation_forest", "local_outlier_factor"])


if __name__ == "__main__":
    unittest.main()
