from numpy.random import default_rng

# Only Bayesian networks considered in this script are path graphs
# They are represented as a 2 by n array.
# dist[j][i] = Pr(v_i = 1 | v_(i-1) = j)
# e.g. dist[0][i] = Pr(v_i = 1 | v_(i-1) = 0)
# special case is dist[0][0] which is the probability v_i = 0, and dist[1][0] = Pr(v_i = 1)


def bit_strings(n):
    """
    Generates all the binary strings of length n
    :param n: positive integer
    :return: array containing 2^n 0-1 bit strings
    """
    if n == 1:
        return ["0", "1"]
    else:
        prev = bit_strings(n - 1)
        prev_copy = prev.copy()
        _next = []
        for s in prev:
            _next.append(s + "0")
        for s in prev_copy:
            _next.append(s + "1")
        return _next


def uniform_bn(n):
    return [[0.5] * n for _ in range(2)]


def massonone_bn(n):
    return [[0] * n, [1] * n]


# These two are based on Sutanu's construction
def constructed_reference_bn(n):
    dist = [[0] * n for _ in range(2)]
    _1byn = 1/n
    _1byn_prime = 1 - _1byn
    for i in range(n):
        if i == 0:  # first bit is uniform
            dist[0][i] = 0.5
            dist[1][i] = 0.5
        else:
            dist[0][i] = _1byn
            dist[1][i] = _1byn_prime

    return dist


def constructed_random_bn(n):
    dist = [[0] * n for _ in range(2)]
    c1 = 1 / (2 * n)
    c1_prime = 1 - c1

    c2 = 3 / (2 * n)
    c2_prime = 1 - c2

    rng = default_rng()
    for i in range(n):
        if i == 0:
            dist[0][i] = 0.5
            dist[1][i] = 0.5
        else:
            if rng.random() < 0.5:
                dist[0][i] = c1
                dist[1][i] = c1_prime
            else:
                dist[0][i] = c2
                dist[0][i] = c2_prime

    return dist


def normal_around_uniform(n, sigma, seed=None):
    dist = [[0] * n for _ in range(2)]
    if seed is None:
        rng = default_rng()
    else:
        rng = default_rng(seed)

    mu, sigma = 0.5, sigma
    for i in range(n):
        p = rng.normal(mu, sigma)
        if p > 1:
            p = 1

        if p < 0:
            p = 0

        dist[0][i] = p
        dist[1][i] = 1 - p

    return dist
