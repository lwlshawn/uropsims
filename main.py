from numpy.random import default_rng
import matplotlib.pyplot as plt
import math

# Only Bayesian networks considered in this script are path graphs
# They are represented as a 2 by n array.
# dist[j][i] = Pr(v_i = 1 | v_(i-1) = j)
# e.g. dist[0][i] = Pr(v_i = 1 | v_(i-1) = 0)
# special case is dist[0][0] which is the probability v_i = 0, and dist[1][0] = Pr(v_i = 1)
import generate


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


def graph_numerical_dtv(n):
    xi = [i for i in range(1, n)]
    yi = [dtv(generate.constructed_reference_bn(i), generate.constructed_random_bn(i)) for i in range(1, n)]

    plt.plot(xi, yi)
    plt.show()

# ================================== TESTED FUNCTIONS ABOVE THIS LINE ====================================


def dkl(pi, qi):  # computes dkl between two bernoulli random variables
    first = (pi - qi)**2 / qi
    second = ((1 - pi) - (1 - qi))**2 / (1 - qi)
    return first + second


def modified_distance_statistic(p, q):
    # this computes the KL factorisation, substituted with chi-square distance as per the journal
    n = len(p[0])
    _sum = 0
    for i in range(n):
        for a in range(2):
            if i == 0:
                _sum += dkl(p[a][i], q[a][i])  # compute the dKL of P(i|a), Q(i|a)
            else:
                _sum += dkl(p[a][i], q[a][i]) * compute_marginal(p, i - 1)[a]

    return _sum


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


# modified version that also updates a frequency table
# freq_table[i][0] = # of times v_i is 0. freq_table[i][1] = # of times v_i is 1
# cond_freq_table[i][j][k] = # of times v_i is k, given v_(i - 1) is j
def sample_and_update(dist, freq_table, cond_freq_table):
    rng = default_rng()
    n = len(dist[0])

    prev = None
    sample = ""
    for i in range(n):
        if i == 0:
            coin = rng.random() < dist[0][i] #the value of the current node
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


# cond_freq_table[i][j][k] = # of times v_i is k, given v_(i - 1) is j
# freq_table[i][j] = # of times v_i = j
# dist[j][i] = Pr(v_i = 1 | v_(i-1) = j)
def compute_statistic(eps, dist, ref):
    dst = modified_distance_statistic(dist, ref)  # what we hope our statistic is close to approximating
    n = len(ref[0])
    m = math.ceil(1000 * (1 / eps**2) * math.sqrt(n))  # draw 10000 rt(n) / eps**2 samples

    # next, it estimates it using sampling, in the style of the statistic we developed.
    # our frequency table needs to count # of times first node is 0 and first node is 1
    # also needs to track conditional counts
    f_table = [[0] * 2 for _ in range(n)]
    cf_table = [[[2,2], [2,2]] for _ in range(n)]  # start all conditional counts with 2
    for i in range(m):
        sample_and_update(dist, f_table, cf_table)

    # finally, it computes the statistic
    _sum = 0
    for i in range(n):
        for j in range(2):
            for k in range(2):
                if i == 0:
                    pass
                else:
                    qkj = None  # qkj = Pr_q(vi = k | v_(i-1) = j)
                    if k == 0:
                        qkj = 1 - ref[j][k]
                    else:
                        qkj = ref[j][k]

                    num = (cf_table[i][j][k] - f_table[i - 1][j] * qkj)**2 - cf_table[i][j][k]
                    den = (f_table[i - 1][j] - 1) * qkj
                    _sum += num/den + (f_table[i-1][j] / f_table[i-1][j] - 1)

    # re-scale the sum to check against our distance statistic
    _sum = _sum / m

    # at this point we have the statistic stored in _sum, and the estimate dst
    # output relative error, and absolute error
    print(f"epsilon is: {eps}")
    print(f"true value of distance statistic is: {dst}")
    print(f"actual dTV between distributions is: {dtv(dist, ref)}")
    print(f"statistic value is: {_sum}")
    print(f"relative error is: {abs(dst - _sum) / _sum}")
    print(f"absolute error is: {abs(dst - _sum)}")
    return _sum

# assuming there are no indices mistakes (which there almost certainly are), i need to write a new function that


if __name__ == '__main__':
    p = generate.constructed_reference_bn(6)
    q = generate.constructed_random_bn(6)

    # q = generate.close_to_uniform_bn(6)
    compute_statistic(0.1, q, p)

    #graph_numerical_dtv(10)
