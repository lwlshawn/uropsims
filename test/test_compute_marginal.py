from unittest import TestCase

import generate
from main import compute_marginal
from test.test_distributions import DIST3, DIST2, DIST1, DELTA


class Test(TestCase):
    def test_compute_marginal1(self):
        # test one: uniform_bn marginals are all 0.5
        marginals = []
        for i in range(3):
            marginals.append(compute_marginal(DIST1, i))

        for i in range(3):
            for j in range(2):
                self.assertEqual(marginals[i][j], 0.5)

    def test_compute_marginal2(self):
        # test two: sutanu's construction for 3 nodes, marginals are still all 0.5
        marginals = []
        for i in range(3):
            marginals.append(compute_marginal(DIST2, i))

        for i in range(3):
            for j in range(2):
                self.assertEqual(marginals[i][j], 0.5)

    def test_compute_marginal3(self):
        # test three: detailed in sims test note.
        marginals = []
        for i in range(3):
            marginals.append(compute_marginal(DIST3, i))

        self.assertAlmostEqual(marginals[0][0], 0.5, delta=DELTA)
        self.assertAlmostEqual(marginals[0][1], 0.5, delta=DELTA)
        self.assertAlmostEqual(marginals[1][0], 19/30, delta=DELTA)
        self.assertAlmostEqual(marginals[1][1], 11/30, delta=DELTA)
        self.assertAlmostEqual(marginals[2][0], 1069/1260, delta=DELTA)
        self.assertAlmostEqual(marginals[2][1], 191/1260, delta=DELTA)
