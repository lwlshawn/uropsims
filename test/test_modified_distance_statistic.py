from unittest import TestCase

from main import modified_distance_statistic
from test.test_distributions import DIST1, DIST3, DELTA


class Test(TestCase):
    def test_modified_distance_statistic(self):
        self.assertAlmostEqual(modified_distance_statistic(DIST1, DIST3), 3731/3600, delta = 0.01)
