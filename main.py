from numpy.random import default_rng
import math

import experiments
import generate

# Only Bayesian networks considered in this script are path graphs
# They are represented as a 2 by n array.
# dist[j][i] = Pr(v_i = 1 | v_(i-1) = j)
# e.g. dist[0][i] = Pr(v_i = 1 | v_(i-1) = 0)
# special case is dist[0][0] which is the probability v_i = 0, and dist[1][0] = Pr(v_i = 1)

# given a bayes net, draws a length n bitstring representing a sample
def sample_from_dist(dist):
    rng = default_rng()
    n = len(dist[0])

    prev = None
    sample = ""
    for i in range(n):
        if i == 0:
            # if rng.random() < dist[0][i], v0 = 1. else v0 = 0.
            coin = rng.random() < dist[0][i]
            prev = int(coin)
            sample += str(prev)
        else:
            coin = rng.random() < dist[prev][i]
            prev = int(coin)
            sample += str(prev)

    return sample


def draw_m_samples(m, dist):
    dct = {}
    for i in range(m):
        sample = sample_from_dist(dist)
        if sample not in dct:
            dct[sample] = 1
        else:
            dct[sample] += 1

    for key in dct:
        dct[key] = dct[key] / m
    return dct
# ============================== NOT NECESSARY TO TEST FUNCTIONS ABOVE THIS LINE ==============================


def compute_probability(dist, x):
    """
    Generates the probability under dist, of seeing the value x
    :param dist: Bayes net with n nodes
    :param x: 0-1 bitstring of length n
    :return: probability of x under this bayes net
    """
    acc = 1
    for i in range(len(x)):
        if i == 0:
            if x[i] == '0':
                acc *= dist[0][0]
            elif x[i] == '1':
                acc *= dist[1][0]
        else:
            if x[i - 1] == '0':  # previous character is 0
                if x[i] == '1':  # current character is 1
                    acc *= dist[0][i]
                elif x[i] == '0':  # current character is 0
                    acc *= (1 - dist[0][i])
            elif x[i - 1] == '1': # previous character is 1
                if x[i] == '1':
                    acc *= dist[1][i]
                elif x[i] == '0':
                    acc *= (1 - dist[1][i])

    return acc


def compute_marginals(dist):
    n = len(dist[0])
    marginals = [0] * n
    # marginals[i] = Pr(vi = 1) is enough
    for i in range(n):
        if i == 0:
            marginals[0] = dist[1][0]
        else:
            marginals[i] = dist[0][i] * (1 - marginals[i - 1]) + dist[1][i] * (marginals[i - 1])
    return marginals


def dtv(p, q):
    """
    Computes and returns the dTV between distributions p and q
    """
    n = len(p[0])
    _Omega = generate.bit_strings(n)
    _sum = 0

    for x in _Omega:
        px = compute_probability(p, x)
        qx = compute_probability(q, x)
        _sum += abs(px - qx)

    return 0.5 * _sum


def d_chi_square(pi, qi):  # computes d_chi_square between two bernoulli random variables
    first = (pi - qi)**2 / qi
    second = ((1 - pi) - (1 - qi))**2 / (1 - qi)
    return first + second


def modified_distance_statistic(p, q):
    # this computes the KL factorisation, substituted with chi-square distance as per the journal
    n = len(p[0])
    _sum = 0
    marginals = compute_marginals(p)
    for i in range(n):
        if i == 0:
            _sum += d_chi_square(p[1][0], q[1][0])
        else:
            for a in range(2):
                _sum += d_chi_square(p[a][i], q[a][i]) * (marginals[i] * a + (1 - marginals[i]) * (1 - a))

    return _sum


# freq_table[i][0] = # of times v_i is 0. freq_table[i][1] = # of times v_i is 1
# cond_freq_table[i][j][k] = # of times v_i is k, given v_(i - 1) is j
def sample_and_update(dist, freq_table, cond_freq_table, rng):
    n = len(dist[0])

    prev = None
    sample = ""
    for i in range(n):
        if i == 0:
            coin = rng.random() < dist[0][i]  # the value of the current node
            freq_table[i][coin] += 1
            prev = int(coin)
            sample += str(prev)
        else:
            coin = rng.random() < dist[prev][i]
            cond_freq_table[i][prev][coin] += 1
            freq_table[i][coin] += 1
            prev = int(coin)
            sample += str(prev)

    return sample


def dkl(pi, qi):  # computes dkl between two bernoulli random variables
    # assumes that qi != 0, 1
    assert qi != 0 and qi != 1

    if pi == 0:
        first = 0
    else:

        first = pi * math.log(pi/qi, 2)

    if pi == 1:
        second = 0
    else:
        second = (1 - pi) * math.log((1 - pi)/(1 - qi), 2)

    return first + second


def compute_kl_factorisation(p, q):
    n = len(p[0])
    _sum = 0
    marginals = compute_marginals(p)
    for i in range(n):
        if i == 0:
            _sum += dkl(p[1][0], q[1][0])
        else:
            for a in range(2):
                _sum += dkl(p[a][i], q[a][i]) * (marginals[i] * a + (1 - marginals[i]) * (1 - a))
    return _sum


# ============================== TESTED FUNCTIONS ABOVE THIS LINE ==============================
# Basic compute_statistic function, on which the others build on
# cond_freq_table[i][j][k] = # of times v_i is k, given v_(i - 1) is j
# freq_table[i][j] = # of times v_i = j
# dist[j][i] = Pr(v_i = 1 | v_(i-1) = j)
# special case is dist[0][0] which is the probability v_i = 0, and dist[1][0] = Pr(v_i = 1)
def compute_statistic(eps, dist, ref):
    true_dst = modified_distance_statistic(dist, ref)  # what we hope our statistic is close to approximating
    true_dtv = dtv(dist, ref)
    n = len(ref[0])
    m = math.ceil(10 * (1 / eps**2) * math.sqrt(n))  # draw 1000 rt(n) / eps**2 samples

    # next, it estimates it using sampling, in the style of the statistic we developed.
    # our frequency table needs to count # of times first node is 0 and first node is 1
    # also needs to track conditional counts
    # start all counts from 2, to avoid divide by 0 errors
    f_table = [[2] * 2 for _ in range(n)]
    cf_table = [[[2, 2], [2, 2]] for _ in range(n)]
    rng = default_rng()
    for i in range(m):
        sample_and_update(dist, f_table, cf_table, rng)

    # finally, it computes the statistic
    _sum = 0
    for i in range(n):
        if i == 0:
            for k in range(2):
                nk = f_table[0][k]  # number of times v0 = k
                qk = dist[k][0]  # probability v0 = k under q
                num = (nk - m*qk)**2 - nk
                den = (m - 1) * qk
                _sum += num/den + m/(m - 1)

        else:
            for j in range(2):
                for k in range(2):
                    # qkj should be Pr_Q(vi = k | v_(i - 1) = j)
                    # dist[j][i] = Pr(vi_1 = 1 | v_(i - 1) = j
                    if k == 0:
                        qkj = 1 - ref[j][i]
                    else:
                        qkj = ref[j][i]

                    # nkj = # of times vi = k, given v_(i - 1) = j = cf_table[i][j][k]
                    nkj = cf_table[i][j][k]

                    num = (nkj - f_table[i - 1][j] * qkj)**2 - nkj
                    den = (f_table[i - 1][j] - 1) * qkj
                    _sum += num/den + (f_table[i-1][j] / (f_table[i-1][j] - 1))

    # re-scale the sum to check against our distance statistic
    _sum = _sum / m
    absolute_error = abs(true_dst - _sum)
    relative_error = abs(true_dst - _sum) / _sum

    # print_summary()
    def print_summary():
        print(f"actual d* between distributions is: {true_dst}")
        print(f"statistic value is: {_sum}")
        print(f"absolute error is: {absolute_error}")
        print(f"relative error is: {relative_error}")
        print()
        print(f"epsilon is: {eps}")
        print(f"actual dTV between distributions is: {true_dtv}")

    print_summary()


# Modified to remove dTV computation to allow for larger values of n.
def compute_statistic1(eps, dist, ref):
    true_dst = modified_distance_statistic(dist, ref)
    true_dkl = compute_kl_factorisation(dist, ref)

    n = len(ref[0])
    m = math.ceil(10 * (1 / eps**2) * math.sqrt(n))

    f_table = [[2] * 2 for _ in range(n)]
    cf_table = [[[2, 2], [2, 2]] for _ in range(n)]
    rng = default_rng()
    for i in range(m):
        sample_and_update(dist, f_table, cf_table, rng)

    _sum = 0
    for i in range(n):
        if i == 0:
            for k in range(2):
                nk = f_table[0][k]  # number of times v0 = k
                qk = dist[k][0]  # probability v0 = k under q
                num = (nk - m*qk)**2 - nk
                den = (m - 1) * qk
                _sum += num/den + m/(m - 1)

        else:
            for j in range(2):
                for k in range(2):
                    # qkj should be Pr_Q(vi = k | v_(i - 1) = j)
                    # dist[j][i] = Pr(vi_1 = 1 | v_(i - 1) = j
                    # TBD UPDATE GENERATE FUNCTIONS TO NEVER GENERATE 0
                    if k == 0:
                        qkj = 1 - ref[j][i]
                    else:
                        qkj = ref[j][i]

                    # nkj = # of times vi = k, given v_(i - 1) = j = cf_table[i][j][k]
                    nkj = cf_table[i][j][k]

                    num = (nkj - f_table[i - 1][j] * qkj)**2 - nkj
                    den = (f_table[i - 1][j] - 1) * qkj
                    _sum += num/den + (f_table[i-1][j] / (f_table[i-1][j] - 1))

    # re-scale the sum to check against our distance statistic
    _sum = _sum / m
    absolute_error = abs(_sum - true_dst)
    relative_error = abs(_sum - true_dst) / true_dst

    return absolute_error, relative_error, true_dst, true_dkl


if __name__ == '__main__':
    experiments.graph_numerical_dtv(20)



