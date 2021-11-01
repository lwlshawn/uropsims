from unittest import TestCase

import generate
from main import compute_kl_factorisation, dkl
from test.test_distributions import DELTA, DIST3, DIST4


class Test(TestCase):
    def test_compute_kl_factorisation1(self):
        p = generate.constructed_random_bn(4, 0)
        q = generate.constructed_reference_bn(4)
        check = dkl(3/8, 1/4) + 2 * dkl(1/8, 1/4)
        self.assertAlmostEqual(compute_kl_factorisation(p, q), check, delta=DELTA)

    def test_compute_kl_factorisation2(self):
        p = generate.constructed_reference_bn(10)
        q = p
        self.assertEqual(compute_kl_factorisation(p, q), 0)

    def test_compute_kl_factorisation3(self):
        c1 = (19/30) * dkl(2/5, 1/4)
        c2 = (11/30) * dkl(1/3, 3/4)
        c3 = (1069/1260) * dkl(1/7, 1/4)
        c4 = (191/1260) * dkl(1/6, 3/4)
        check = c1 + c2 + c3 + c4
        self.assertAlmostEqual(compute_kl_factorisation(DIST3, DIST4), check, delta=DELTA)
