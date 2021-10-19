from unittest import TestCase

import generate
from main import compute_probability
from test.test_distributions import DIST4, DELTA, DIST3


class Test(TestCase):
    # uniform distribution
    def test_compute_probability1(self):
        dist = generate.uniform_bn(3)
        bit_strings = generate.bit_strings(3)
        for s in bit_strings:
            self.assertEqual(compute_probability(dist, s), 1 / 2 ** 3)

    def test_compute_probability2(self):
        self.assertAlmostEqual(compute_probability(DIST3, "000"), 9 / 35, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST3, "001"), 3 / 70, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST3, "010"), 1 / 6, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST3, "011"), 1 / 30, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST3, "100"), 2 / 7, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST3, "101"), 1 / 21, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST3, "110"), 5 / 36, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST3, "111"), 1 / 36, delta=DELTA)

    def test_compute_probability3(self):
        self.assertAlmostEqual(compute_probability(DIST4, "000"), 9 / 32, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST4, "001"), 3 / 32, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST4, "010"), 1 / 32, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST4, "011"), 3 / 32, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST4, "100"), 3 / 32, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST4, "101"), 1 / 32, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST4, "110"), 3 / 32, delta=DELTA)
        self.assertAlmostEqual(compute_probability(DIST4, "111"), 9 / 32, delta=DELTA)
