import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy


def draw_distributions():
    """Draws the distributions used in our experiment."""
    x = np.linspace(-3, 3, 1000)
    normal = scipy.stats.norm.pdf(x)
    exp = scipy.stats.expon.pdf(x)
    beta1 = scipy.stats.beta.pdf(x, 1, 1)
    logn = scipy.stats.lognorm.pdf(x, 1)
    lapl = scipy.stats.laplace.pdf(x)
    beta2 = scipy.stats.beta.pdf(x, 10, 2)
    for y, label in zip([normal, exp, beta1, beta2, logn, lapl], ['N(0, 1)', 'Exp(1)', 'U(0, 1)', 'B(10, 2)',
                                                                  'LN(0, 1)', 'Laplace(0, 1)']):
        plt.plot(x, y, label=label)

    plt.legend(title='distribution')
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.tight_layout()
    plt.show()


def draw_histogram_distances():
    """Draws a histogram of difference to exact intervals from double and standard."""
    intervals = pd.read_csv('results/intervals.csv')
    for (dgp, statistic, n, alpha) in [('DGPExp_1', 'mean', 16, 0.975), ('DGPExp_1', 'mean', 32, 0.975),
                                       ('DGPExp_1', 'mean', 64, 0.95), ('DGPExp_1', 'mean', 16, 0.95),
                                       ('DGPExp_1', 'mean', 32, 0.95), ('DGPExp_1', 'mean', 64, 0.95),
                                       ('DGPNorm_0_1', 'mean', 256, 0.975), ('DGPBeta_1_1', 'mean', 16, 0.25),
                                       ('DGPBeta_1_1', 'mean', 32, 0.975)]:
        int_exp = intervals[
            (intervals['dgp'] == dgp) & (intervals['statistic'] == statistic) & (intervals['n'] == n) & (
                        intervals['alpha'] == alpha) & (intervals['B'] == 1000)]
        int_exp['dist_d'] = int_exp['double'] - int_exp['exact']
        int_exp['dist_s'] = int_exp['standard'] - int_exp['exact']
        minn = int_exp[['dist_s', 'dist_d']].min().min()
        maxx = int_exp[['dist_s', 'dist_d']].max().max()
        plt.hist(int_exp['dist_d'].values, label='double', bins=100, alpha=0.8, range=(minn, maxx))
        plt.hist(int_exp['dist_s'].values, label='standard', bins=100, alpha=0.8, range=(minn, maxx))
        plt.axvline(x=0, linestyle='--', c='gray')
        plt.title(dgp + ', ' + statistic + ', ' + str(n) + ', ' + str(alpha))
        plt.legend()
        plt.savefig(f'images/hist_{dgp}_{statistic}_{n}_{alpha}.png')
        plt.close()


def draw_hist_with_exact():
    """Draws a histogram of double, standard and exact intervals, together with true value of the statistic."""
    intervals = pd.read_csv('results/intervals.csv')
    for (dgp, statistic, n, alpha, true_val) in [('DGPExp_1', 'mean', 16, 0.975, 1)]:
        int_exp = intervals[
            (intervals['dgp'] == dgp) & (intervals['statistic'] == statistic) & (intervals['n'] == n) & (
                        intervals['alpha'] == alpha) & (intervals['B'] == 1000)]
        int_exp['dist_d'] = int_exp['double']
        int_exp['dist_s'] = int_exp['standard']
        minn = int_exp[['dist_s', 'dist_d', 'exact']].min().min()
        maxx = int_exp[['dist_s', 'dist_d', 'exact']].max().max()
        plt.hist(int_exp['dist_d'].values, label='double', bins=100, alpha=0.8, range=(minn, maxx))
        plt.hist(int_exp['dist_s'].values, label='standard', bins=100, alpha=0.8, range=(minn, maxx))
        plt.hist(int_exp['exact'].values, label='exact', bins=100, alpha=0.8, range=(minn, maxx))
        plt.axvline(x=true_val, linestyle='--', c='gray', label='true value')
        plt.title(dgp + ', ' + statistic + ', ' + str(n) + ', ' + str(alpha))
        plt.legend()
        plt.savefig(f'images/hist3_{dgp}_{statistic}_{n}_{alpha}.png')
        plt.close()

