from unittest import TestCase
from numpy.random import default_rng

from main import sample_and_update
from test.test_distributions import DIST1, DIST2


class Test(TestCase):
    # freq_table[i][0] = # of times v_i is 0. freq_table[i][1] = # of times v_i is 1
    # cond_freq_table[i][j][k] = # of times v_i is k, given v_(i - 1) is j

    def test_sample_and_update1(self):
        n = len(DIST1[0])
        f_table = [[0] * 2 for _ in range(n)]
        cf_table = [[[2, 2], [2, 2]] for _ in range(n)]
        rng = default_rng(seed=0)
        # samples drawn are:
        # 011, 100, 000, 001, 010, 100, 111, 100, 010, 000

        for i in range(10):
            sample_and_update(DIST1, f_table, cf_table, rng)

        self.assertEqual(cf_table[1][0][0], 5)
        self.assertEqual(cf_table[1][0][1], 5)
        self.assertEqual(cf_table[1][1][0], 5)
        self.assertEqual(cf_table[1][1][1], 3)
        self.assertEqual(cf_table[2][0][0], 7)
        self.assertEqual(cf_table[2][0][1], 3)
        self.assertEqual(cf_table[2][1][0], 4)
        self.assertEqual(cf_table[2][1][1], 4)

    def test_sample_and_update2(self):
        n = len(DIST1[0])
        f_table = [[0] * 2 for _ in range(n)]
        cf_table = [[[2, 2], [2, 2]] for _ in range(n)]
        rng = default_rng(seed=73)
        # samples drawn are:
        # 111, 011, 010, 011, 101, 101, 010, 111, 001, 101

        for i in range(10):
            sample = sample_and_update(DIST1, f_table, cf_table, rng)

        self.assertEqual(cf_table[1][0][0], 3)
        self.assertEqual(cf_table[1][0][1], 6)
        self.assertEqual(cf_table[1][1][0], 5)
        self.assertEqual(cf_table[1][1][1], 4)
        self.assertEqual(cf_table[2][0][0], 2)
        self.assertEqual(cf_table[2][0][1], 6)
        self.assertEqual(cf_table[2][1][0], 4)
        self.assertEqual(cf_table[2][1][1], 6)

    # tests verifying that samples match what we expect statistically
    def test_sample_and_update3(self):
        n = len(DIST1[0])
        f_table = [[0] * 2 for _ in range(n)]
        cf_table = [[[2, 2], [2, 2]] for _ in range(n)]
        rng = default_rng()
        samples = {}

        for i in range(80000):
            sample = sample_and_update(DIST1, f_table, cf_table, rng)
            if sample not in samples:
                samples[sample] = 1
            else:
                samples[sample] += 1

        check = 0
        for sample in samples:
            check += samples[sample]
        self.assertEqual(check, m)

        for sample in samples:
            self.assertAlmostEqual(samples[sample], 10000, delta=200)

    def test_sample_and_update4(self):
        n = len(DIST2[0])
        m = 100000
        f_table = [[0] * 2 for _ in range(n)]
        cf_table = [[[2, 2], [2, 2]] for _ in range(n)]
        rng = default_rng()
        samples = {}

        for i in range(m):
            sample = sample_and_update(DIST2, f_table, cf_table, rng)
            if sample not in samples:
                samples[sample] = 1
            else:
                samples[sample] += 1

        check = 0
        for sample in samples:
            check += samples[sample]
        self.assertEqual(check, m)

        self.assertAlmostEqual(samples["000"], m * 2/9, delta=200)
        self.assertAlmostEqual(samples["001"], m * 1/9, delta=200)
        self.assertAlmostEqual(samples["010"], m * 1/18, delta=200)
        self.assertAlmostEqual(samples["011"], m * 1/9, delta=200)
        self.assertAlmostEqual(samples["100"], m * 1/9, delta=200)
        self.assertAlmostEqual(samples["101"], m * 1/18, delta=200)
        self.assertAlmostEqual(samples["110"], m * 1/9, delta=200)
        self.assertAlmostEqual(samples["111"], m * 2/9, delta=200)
