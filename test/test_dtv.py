from unittest import TestCase

from main import dtv
from test.test_distributions import DIST1, DELTA, DIST2, DIST3, DIST4


class Test(TestCase):
    def test_dtv1(self):
        self.assertAlmostEqual(dtv(DIST1, DIST1), 0, delta=DELTA)
        self.assertAlmostEqual(dtv(DIST2, DIST2), 0, delta=DELTA)
        self.assertAlmostEqual(dtv(DIST3, DIST3), 0, delta=DELTA)
        self.assertAlmostEqual(dtv(DIST4, DIST4), 0, delta=DELTA)

    def test_dtv2(self):
        self.assertAlmostEqual(dtv(DIST1, DIST2), 7/36, delta=DELTA)

    def test_dtv3(self):
        true_dtv = 0
        true_dtv += abs(2/9 - 9/32)
        true_dtv += abs(1/9 - 3/32)
        true_dtv += abs(1/18 - 1/32)
        true_dtv += abs(1/9 - 3/32)
        true_dtv += abs(1/9 - 3/32)
        true_dtv += abs(1/18 - 1/32)
        true_dtv += abs(1/9 - 3/32)
        true_dtv += abs(2/9 - 9/32)
        true_dtv *= 0.5

        self.assertAlmostEqual(dtv(DIST2, DIST4), true_dtv, delta=DELTA)
