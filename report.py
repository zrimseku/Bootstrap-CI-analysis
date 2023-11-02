import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from results_visualizations import kl


def draw_distributions():
    """Draws the distributions used in our experiment."""
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    x = np.linspace(-3, 3, 1000)
    normal = scipy.stats.norm.pdf(x)
    exp = scipy.stats.expon.pdf(x)
    beta1 = scipy.stats.beta.pdf(x, 1, 1)
    logn = scipy.stats.lognorm.pdf(x, 1)
    lapl = scipy.stats.laplace.pdf(x)
    beta2 = scipy.stats.beta.pdf(x, 10, 2)
    for i, (y, label, ax) in enumerate(zip([normal, lapl, logn, exp, beta1, beta2], ['N(0, 1)', 'Laplace(0, 1)',
                                                                                     'LN(0, 1)', 'Exp(1)', 'U(0, 1)',
                                                                                     'B(10, 2)'], axes)):
        ax.plot(x, y, label=label)
        ax.set_title(label)
        if label[0] in ['B', 'U']:
            ax.set_ylim(0, 4.5 if label == 'B(10, 2)' else 2)
        else:
            ax.set_ylim(0, 1)

        if i % 2 == 0:
            ax.set_ylabel('P(x)')

        if i > 3:
            ax.set_xlabel('x')

    plt.tight_layout()
    plt.savefig(f'magistrska/distributions.png')
    plt.close()


def draw_bernoulli_se(r=10000, alphas=[0.025, 0.05, 0.25]):
    x = np.linspace(0, 1, 1000)

    def se_bern(p):
        return (p * (1 - p) / r) ** 0.5

    y = se_bern(x)

    fig, ax = plt.subplots()

    ax.plot(x, y)

    for alpha in alphas:
        plt.vlines(x=[alpha, 1 - alpha], color='gray', linestyle='--', ymin=0, ymax=se_bern(alpha))

    # plt.hlines(y=[se_bern(a) for a in alphas], xmin=-0.05, xmax=[1 - a for a in alphas], color='gray', linestyle='--')
    plt.xlim(-0.05, 1.05)
    plt.ylim(0, 0.0055)

    # Set the additional y-ticks and their labels
    additional_ticks = [round(se_bern(a), 5) for a in alphas]
    additional_labels = [str(at) for at in additional_ticks]

    # Set the y-ticks for both the default ticks and additional ticks
    yticks = np.append(plt.yticks()[0], additional_ticks)
    yticks = [0] + additional_ticks + [0.005]
    yticklabels = [str(at) for at in yticks]

    # Specify the red color for the additional ticks
    plt.yticks(yticks, yticklabels)

    plt.rcParams['mathtext.fontset'] = 'cm'

    xticks = alphas + [0.5] + [1 - a for a in alphas[::-1]]
    xticksl = [0.05, 0.975]
    xticksr = [0.025, 0.95]
    plt.xticks(xticks, xticks)

    tick_labels = ax.set_xticklabels(xticks)
    tick_labels[1].set_horizontalalignment('left')
    tick_labels[-1].set_horizontalalignment('left')
    tick_labels[-2].set_horizontalalignment('right')
    tick_labels[0].set_horizontalalignment('right')

    plt.xlabel(r'$p$')
    plt.ylabel(r'Bernoulli $se$')

    # plt.xticks(alphas + [1 - a for a in alphas[::-1]])
    plt.savefig(f'magistrska/bernoulli_se.png')
    plt.close()


def draw_kl_se(r=10000, xval='kl', scales='log'):
    x = np.linspace(0, 1, 1000)

    def se_kl(p, a):
        fst_order = np.abs(np.log2(p*(1-a)) - np.log2(a*(1-p))) * (p * (1 - p) / r) ** 0.5
        # snd_order = np.abs((1 - 2*p) / (np.log(2) * p * (1 - p))) * (p * (1 - p) / r) / 2
        return fst_order

    fig, ax = plt.subplots()

    alphas = np.array([0.025, 0.05, 0.25] + ([] if xval == 'kl' else [0.75, 0.95, 0.975]))

    for alpha in alphas:
        y = se_kl(x, alpha)
        # plt.vlines(x=[alpha, 1 - alpha], color='gray', linestyle='--', ymin=0, ymax=se_bern(alpha))
        ax.plot(kl(x, alpha) if xval == 'kl' else x, y,
                label=(str(alpha) + ' / ' + str(1 - alpha)) if xval == 'kl' else alpha)

    # plt.xlim(0, 1)
    plt.xlabel(r'$D_{KL}(p, \alpha)$' if xval == 'kl' else r'$p$')
    plt.ylabel(r'se of $D_{KL}(p, \alpha)$')
    plt.xscale(scales)
    plt.yscale(scales)
    plt.legend(title=r'$\alpha$', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # plt.xticks(alphas + [1 - a for a in alphas[::-1]])
    plt.savefig(f'magistrska/kl_se_{xval}_{scales}.png')
    plt.close()


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
        plt.hist(int_exp['dist_d'].values, label='double', bins=100, alpha=0.5, range=(minn, maxx))
        plt.hist(int_exp['dist_s'].values, label='standard', bins=100, alpha=0.5, range=(minn, maxx))
        plt.hist(int_exp['exact'].values, label='exact', bins=100, alpha=0.5, range=(minn, maxx))
        plt.axvline(x=true_val, linestyle='--', c='gray', label='true value')
        # plt.title(dgp + ', ' + statistic + ', ' + str(n) + ', ' + str(alpha))
        plt.legend()
        plt.savefig(f'images/hist3_{dgp}_{statistic}_{n}_{alpha}.png')
        plt.close()
        cov_ex = np.mean(int_exp['exact'] > int_exp['true_value'])
        cov_double = np.mean(int_exp['double'] > int_exp['true_value'])
        cov_std = np.mean(int_exp['standard'] > int_exp['true_value'])
        print(f'Coverages: exact = {cov_ex}, double = {cov_double}, standard = {cov_std}')

        d_double = np.mean(int_exp['double'] - int_exp['exact'])
        d_std = np.mean(int_exp['standard'] - int_exp['exact'])
        print(f'Distances: double = {d_double}, standard = {d_std}')


if __name__ == '__main__':
    # ggplot okej, whitegrid okej
    # sns.set_style('whitegrid')
    plt.style.use('seaborn')
    # draw_distributions()
    # draw_bernoulli_se()
    # draw_kl_se(xval='p', scales='linear')
    # draw_kl_se(scales='linear')
    draw_hist_with_exact()

