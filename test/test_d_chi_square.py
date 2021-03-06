from unittest import TestCase

from main import d_chi_square, compute_marginals
from test.test_distributions import DIST1, DELTA


class Test(TestCase):
    def test_d_chi_square_1(self):
        marginals = compute_marginals(DIST1)
        for i in range(len(marginals)):
            self.assertEqual(d_chi_square(marginals[i], marginals[i]), 0)

    def test_d_chi_square_2(self):
        self.assertAlmostEqual(d_chi_square(1 / 3, 4 / 5), 49 / 36, delta=DELTA)

    def test_d_chi_square_3(self):
        self.assertAlmostEqual(d_chi_square(1 / 2, 2 / 5), 1 / 24, delta=DELTA)

    def test_d_chi_square_4(self):
        self.assertAlmostEqual(d_chi_square(1 / 2, 1 / 3), 1 / 8, delta=DELTA)

    def test_d_chi_square_5(self):
        self.assertAlmostEqual(d_chi_square(1 / 2, 1 / 7), 25 / 24, delta=DELTA)

    def test_d_chi_square_6(self):
        self.assertAlmostEqual(d_chi_square(1 / 2, 1 / 6), 4 / 5, delta=DELTA)

    def test_d_chi_square_7(self):
        self.assertAlmostEqual(d_chi_square(1 / 3, 1 / 4), 1 / 27, delta=DELTA)

