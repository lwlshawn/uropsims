from unittest import TestCase

from main import modified_distance_statistic
from test.test_distributions import DIST1, DIST3, DELTA, DIST2, DIST4


class Test(TestCase):
    def test_modified_distance_statistic1(self):
        # numerical accuracy appears very poor which is concerning.
        self.assertAlmostEqual(modified_distance_statistic(DIST3, DIST1), 0.5616477702, delta=0.005)

    def test_modified_distance_statistic2(self):
        # numerical accuracy here improves again greatly when fractions are simpler
        self.assertAlmostEqual(modified_distance_statistic(DIST2, DIST4), 2/27, delta=DELTA)

