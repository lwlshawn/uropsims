import generate
from main import compute_statistic, compute_statistic1, dtv
import matplotlib.pyplot as plt


def graph_numerical_dtv(n):
    xi = [i for i in range(1, n + 1)]
    yi = [dtv(generate.constructed_reference_bn(i), generate.constructed_random_bn(i)) for i in range(1, n + 1)]

    plt.title("dTV against number of nodes")
    plt.plot(xi, yi)
    plt.xticks(xi)
    plt.show()


def unf_normunf(n, sigma):
    p = generate.normal_around_uniform(n, sigma)
    q = generate.uniform_bn(n)
    compute_statistic(0.1, p, q)


def _foundation(fp, fq, title, eps):
    xi = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    y1, y2, y3, y4 = [], [], [], []
    for ni in xi:
        p = fp(ni)
        q = fq(ni)
        (abs_err, rel_err, true_dstar, true_dkl) = compute_statistic1(eps, p, q)
        y1.append(abs_err)
        y2.append(rel_err)
        y3.append(true_dstar)
        y4.append(true_dkl)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    ax1.plot(xi, y1, label="abs_err")
    # ax1.plot(xi, y2, label="rel_err")
    ax1.set_title("error against number of nodes")
    ax1.legend()

    ax2.plot(xi, y3, label="d*")
    ax2.plot(xi, y4, label="dkl")
    ax2.set_title("dkl and d* against number of nodes")
    ax2.legend()

    plt.show()


def run_experiment_0():
    # show difference in performance in epsilon = 0.25 vs epsilon = 0.1.
    xi = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    y1, y2, y3, y4 = [], [], [], []
    for ni in xi:
        p = generate.normal_around_uniform(ni, 0.01)
        q = generate.uniform_bn(ni)
        (abs_err1, _, true_dstar, true_dkl) = compute_statistic1(0.25, p, q)
        (abs_err2, _, _, _) = compute_statistic1(0.1, p, q)
        y1.append(abs_err1)
        y2.append(abs_err2)
        y3.append(true_dstar)
        y4.append(true_dkl)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Comparison of eps=0.1 and eps=0.25")
    ax1.plot(xi, y2, label="abs_err_10")
    ax1.plot(xi, y1, label="abs_err_25")
    ax1.set_title("error against number of nodes")
    ax1.legend()

    ax2.plot(xi, y3, label="d*")
    ax2.plot(xi, y4, label="dkl")
    ax2.set_title("dkl and d* against number of nodes")
    ax2.legend()

    plt.show()


def run_experiment_1():
    _foundation(generate.constructed_random_bn, generate.constructed_reference_bn,
                "Constructed distributions for lower bound conjecture", 0.1)

# uniform, against normal around uniform
def run_experiment_2(sigma):
    _foundation(lambda ni: generate.normal_around_uniform(ni, sigma),
                generate.uniform_bn,
                f"Uniform against normal around uniform distribution, sigma={sigma}", 0.1)

# normal around uniform, against normal around uniform
def run_experiment_3(sigma):
    _foundation(lambda ni: generate.normal_around_uniform(ni, sigma),
                lambda ni: generate.normal_nonzero_around_uniform(ni, sigma),
                f"Comparison of two normal around distributions, sigma={sigma}", 0.1)

