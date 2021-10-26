from numpy.random import default_rng
import matplotlib.pyplot as plt
import math
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


def graph_numerical_dtv(n):
    xi = [i for i in range(1, n)]
    yi = [dtv(generate.constructed_reference_bn(i), generate.constructed_random_bn(i)) for i in range(1, n)]

    plt.plot(xi, yi)
    plt.show()


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
# ============================== TESTED FUNCTIONS ABOVE THIS LINE ==============================


def dkl(pi, qi):  # computes dkl between two bernoulli random variables
    # assumes that qi != 0
    if pi == 0:
        first = 0
    else:
        first = pi * math.log(pi/qi)

    if pi == 1:
        second = 0
    else:
        second = (1 - pi) * math.log((1 - pi)/(1 - qi))

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


# cond_freq_table[i][j][k] = # of times v_i is k, given v_(i - 1) is j
# freq_table[i][j] = # of times v_i = j
# dist[j][i] = Pr(v_i = 1 | v_(i-1) = j)
# special case is dist[0][0] which is the probability v_i = 0, and dist[1][0] = Pr(v_i = 1)
def compute_statistic(eps, dist, ref):
    true_dst = modified_distance_statistic(dist, ref)  # what we hope our statistic is close to approximating
    true_dtv = dtv(dist, ref)
    n = len(ref[0])
    m = math.ceil(1 * (1 / eps**2) * math.sqrt(n))  # draw 1000 rt(n) / eps**2 samples

    # next, it estimates it using sampling, in the style of the statistic we developed.
    # our frequency table needs to count # of times first node is 0 and first node is 1
    # also needs to track conditional counts
    best_relative_error = 1
    absolute_error = float('inf')
    for run in range(3):
        f_table = [[0] * 2 for _ in range(n)]
        cf_table = [[[2, 2], [2, 2]] for _ in range(n)]  # start all conditional counts with 2
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
        if abs(true_dst - _sum) / _sum <= best_relative_error:
            absolute_error = abs(true_dst - _sum)
            best_relative_error = abs(true_dst - _sum) / _sum
            final_sum = _sum

    # at this point we have the statistic stored in _sum, and the estimate dst
    # output relative error, and absolute error
    def print_summary():
        print(f"epsilon is: {eps}")
        print(f"true value of distance statistic is: {true_dst}")
        print(f"sqrt of distance statistic is: {math.sqrt(true_dst)}")
        print(f"actual dTV between distributions is: {true_dtv}")

        print(f"statistic value is: {final_sum}")
        print(f"sqrt of statistic value is {math.sqrt(final_sum)}")
        print(f"best relative error is: {best_relative_error}")
        print(f"absolute error is: {absolute_error}")

    print_summary()

    if math.sqrt(final_sum) > eps / 10:
        decision_close = False
    else:
        decision_close = True

    print(f"decision_close is: {decision_close}")

    if true_dtv > eps:
        actual_close = False
    else:
        actual_close = True

    print(f"actual_close is: {actual_close}")
    if actual_close == decision_close:
        return True
    else:
        return False


# ============================== EXPERIMENTS BEGIN HERE ==============================
def compute_statistic1(eps, dist, ref, runs):
    true_dst = modified_distance_statistic(dist, ref)
    true_dtv = dtv(dist, ref)
    n = len(ref[0])
    m = math.ceil(1000 * (1 / eps**2) * math.sqrt(n))

    best_relative_error = 1
    absolute_error = float('inf')
    for run in range(runs):
        f_table = [[0] * 2 for _ in range(n)]
        cf_table = [[[2, 2], [2, 2]] for _ in range(n)]  # start all conditional counts with 2
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
        if abs(true_dst - _sum) / _sum <= best_relative_error:
            absolute_error = abs(true_dst - _sum)
            best_relative_error = abs(true_dst - _sum) / _sum
            final_sum = _sum

    # at this point we have the statistic stored in _sum, and the estimate dst
    # output relative error, and absolute error
    def print_summary():
        print(f"epsilon is: {eps}")
        print(f"true value of distance statistic is: {true_dst}")
        print(f"sqrt of distance statistic is: {math.sqrt(true_dst)}")
        print(f"actual dTV between distributions is: {true_dtv}")

        print(f"statistic value is: {final_sum}")
        print(f"sqrt of statistic value is {math.sqrt(final_sum)}")
        print(f"best relative error is: {best_relative_error}")
        print(f"absolute error is: {absolute_error}")

    print_summary()
    return best_relative_error, absolute_error


def update_index(li, ri):
    if li == 0 and ri == 0:
        return li, ri + 1

    elif li == 0 and ri == 1:
        return li + 1, ri - 1

    elif li == 1 and ri == 0:
        return li, ri + 1

    else:
        return 1, 1


def first_experiment():
    num = 3
    fig, (ax1, ax2) = plt.subplots(1, 2)

    xi = [i for i in range(1, num)]
    y1, y2 = [], []
    q = generate.uniform_bn(10)
    p1 = generate.normal_around_uniform(10, 0.01, seed=0)
    p2 = generate.normal_around_uniform(10, 0.15, seed=0)
    for k in range(1, num):
        (relative_error, absolute_error) = compute_statistic1(0.1, p1, q, k)
        y1.append(relative_error)
        y2.append(absolute_error)

    ax1.plot(xi, y1, label="relative error")
    ax1.plot(xi, y2, label="absolute error")
    ax1.set_xlabel("k")
    ax1.set_ylabel("error")
    ax1.set_title("sigma = 0.01")

    y1, y2 = [], []
    for k in range(1, num):
        (relative_error, absolute_error) = compute_statistic1(0.1, p2, q, k)
        y1.append(relative_error)
        y2.append(absolute_error)

    ax2.plot(xi, y1, label="relative error")
    ax2.plot(xi, y2, label="absolute error")
    ax2.set_xlabel("k")
    ax2.set_ylabel("error")
    ax2.set_title("sigma = 0.15")

    plt.legend()
    plt.show()


# STEP 2: GRAPH THE ACTUAL DISTANCE STATISTIC, VS THE DTV STATISTIC
def second_experiment():
    deltas = [0.01, 0.05, 0.1, 0.15]
    pairs = []

    titles = [["", ""], ["", ""]]
    titles[0][0] = "sigma = 0.01"
    titles[0][1] = "sigma = 0.05"
    titles[1][0] = "sigma = 0.10"
    titles[1][1] = "sigma = 0.15"

    g = lambda sigma: (lambda n: generate.normal_around_uniform(n, sigma))
    for delta in deltas:
        pairs.append((g(delta), generate.uniform_bn))

    figure, axis = plt.subplots(2,2)
    xi = [x for x in range(3, 11)]

    li = 0
    ri = 0
    for (fp, fq) in pairs:
        y1, y2, y3, y4, y5 = [], [], [], [], []
        for n in range(3, 11): #number of nodes is the x axis
            p = fp(n)
            q = fq(n)
            true_dtv = dtv(p, q)
            true_dst_statistic = modified_distance_statistic(p, q)
            true_dst_statistic_sqrt = math.sqrt(true_dst_statistic)
            true_kl_factorisation = compute_kl_factorisation(p, q)
            y1.append(true_dtv)
            y2.append(true_dst_statistic)
            y3.append(true_dst_statistic_sqrt)
            y4.append(true_kl_factorisation)
            y5.append(math.sqrt(true_kl_factorisation))

        axis[li, ri].plot(xi, y1, label='dTV')
        # axis[li, ri].plot(xi, y2, label='distance statistic')
        axis[li, ri].plot(xi, y3, label='sqrt of distance statistic')
        # axis[li, ri].plot(xi, y4, label='kl factorisation')
        axis[li, ri].plot(xi, y5, label='sqrt of kl factorisation')
        axis[li, ri].set_title(titles[li][ri])
        li, ri = update_index(li, ri)

        print(f'li is: {li}, ri is: {ri}')

    plt.legend()
    plt.show()
"""
Passes sanity checks: 
- new distance metric is consistently larger than actual KL factorisation
- sqrt of new distance metric > sqrt of actual kl factorisation > dtv
"""

def third_experiment():
    pass
    # now we need to find a good break point
    # goal is to distinguish between the cases where
    # KL(P, Q) <= eps^2 / 10 and dtv(P, Q) > eps with good probability




# STEP 3: EXPERIMENT WITH DERIVED BREAK POINTS TO FIND A GOOD PLACE




# need to find a break point that performs well.
# I probably want to fix the distributions I use so that I can consistently test against them
# So find distributions that are varying amounts of dTV away from uniform, and experimentally find
# a good break point


# what we do at least have is that our statistic properly finds an amount such that when we
# take the square root, it is > then the total variation distance.

if __name__ == '__main__':
    p = generate.constructed_random_bn(10)
    q = generate.constructed_reference_bn(10)
    compute_statistic(0.1, p, q)

    first_experiment()
