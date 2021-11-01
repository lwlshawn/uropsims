import math
from unittest import TestCase

from main import dkl
from test.test_distributions import DELTA


class Test(TestCase):
    def test_dkl1(self):
        check = 0.1 * math.log(0.1/0.05, 2) + 0.9 * math.log(0.9/0.95, 2)
        self.assertAlmostEqual(dkl(1/10, 1/20), check, delta=DELTA)

    def test_dkl2(self):
        check = 0.01 * math.log(0.01/0.005, 2) + 0.99 * math.log(0.99/0.995, 2)
        self.assertAlmostEqual(dkl(1/100, 1/200), check, delta=DELTA)

    def test_dkl4(self):
        self.assertEqual(dkl(0, 0.5), 1)

    def test_dkl6(self):
        check = (3/8) * math.log((3/8)/(1/4), 2) + (5/8) * math.log((5/8)/(3/4), 2)
        self.assertAlmostEqual(dkl(3/8, 1/4), check, delta=DELTA)

    def test_dkl7(self):
        self.assertEqual(dkl(5/8, 3/4), dkl(3/8, 1/4))

    def test_dkl8(self):
        check = (1/8) * math.log((1/8)/(1/4), 2) + (7/8) * math.log((7/8)/(3/4), 2)
        self.assertAlmostEqual(dkl(1/8, 1/4), check, delta=DELTA)

    def test_dkl9(self):
        self.assertEqual(dkl(7/8, 3/4), dkl(1/8, 1/4))

    def test_dkl8(self):
        self.assertEqual(dkl(0.5, 0.5), 0)

    # Testing failure cases
    # q = 0
    def test_dkl3(self):
        self.assertRaises(AssertionError, dkl, 0.5, 0)

    # q = 1
    def test_dkl5(self):
        self.assertRaises(AssertionError, dkl, 0, 1)
