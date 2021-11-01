from unittest import TestCase

import generate
from main import compute_marginals
from test.test_distributions import DIST1, DIST2, DIST3, DELTA


class Test(TestCase):
    def test_compute_marginals1(self):
        # test one: uniform_bn marginals are all 0.5
        marginals = compute_marginals(DIST1)

        for i in range(3):
            for j in range(2):
                self.assertEqual(marginals[i] * (1 - j) + (1 - marginals[i]) * j, 0.5)

    def test_compute_marginals2(self):
        marginals = compute_marginals(DIST2)

        for i in range(3):
            for j in range(2):
                self.assertEqual(marginals[i] * (1 - j) + (1 - marginals[i]) * j, 0.5)

    def test_compute_marginals3(self):
        marginals = compute_marginals(DIST3)

        self.assertAlmostEqual((1 - marginals[0]), 0.5, delta=DELTA)
        self.assertAlmostEqual(marginals[0], 0.5, delta=DELTA)
        self.assertAlmostEqual((1 - marginals[1]), 19/30, delta=DELTA)
        self.assertAlmostEqual(marginals[1], 11/30, delta=DELTA)
        self.assertAlmostEqual((1 - marginals[2]), 1069/1260, delta=DELTA)
        self.assertAlmostEqual(marginals[2], 191/1260, delta=DELTA)

    # verifying property that constructed bns have marginal of 1/2 everywhere
    def test_compute_marginals4(self):
        marginals = compute_marginals(generate.constructed_reference_bn(10))
        for i in range(10):
            for j in range(2):
                self.assertEqual(marginals[i] * (1 - j) + (1 - marginals[i]) * j, 0.5)

    def test_compute_marginals5(self):
        marginals = compute_marginals(generate.constructed_random_bn(10))
        for i in range(10):
            for j in range(2):
                self.assertEqual(marginals[i] * (1 - j) + (1 - marginals[i]) * j, 0.5)
