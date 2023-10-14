from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from ci_methods import Bootstrap
from generators import DGP, DGPNorm, DGPExp, DGPBeta, DGPBiNorm, DGPLogNorm, DGPLaplace, DGPBernoulli


def draw_bootstrap_comparison(dgps, ns, statistics, alphas=None, b=1000, methods=None, name=None):

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for dgp, n, statistic, ax in zip(dgps, ns, statistics, axes.flatten()):
        data = dgp.sample(sample_size=n)

        if alphas is None:
            alphas = [0.025, 0.975]
        # compute bootstrap intervals
        bts = Bootstrap(data, statistic)
        bts.sample(b)
        bts.evaluate_statistic()

        if methods is None:
            methods = ['percentile', 'basic', 'bca', 'bc', 'standard', 'smoothed', 'double', 'studentized']

        computed_intervals = {m: defaultdict(list) for m in methods}

        for method in methods:
            cis = bts.ci(coverages=alphas, side='one', method=method)
            for a, ci in zip(alphas, cis):
                computed_intervals[method][a].append(ci)

        # plotting
        colors = iter(plt.cm.jet(np.linspace(0.05, 0.95, len(methods))))
        ax.hist(bts.statistic_values, bins=30, label='statistic')
        if 'smoothed' in methods:
            if np.nan in bts.statistic_values_noise or np.inf in bts.statistic_values_noise:
                print('skipped drawing of smoothed values because of nan values.')  # TODO why are they here?
            else:
                ax.hist(bts.statistic_values_noise, bins=30, label='smoothed stat.', alpha=0.3)
        for method in methods:
            col = next(colors)
            for alpha in alphas:
                if alpha == alphas[0]:  # label only the first line of a method to avoid duplicates in legend
                    ax.axvline(computed_intervals[method][alpha][-1], linestyle='--', label=method, color=col,
                               alpha=0.75)
                else:
                    ax.axvline(computed_intervals[method][alpha][-1], linestyle=':', color=col, alpha=0.75)

    plt.legend([], [], frameon=False)
    handles, labels = plt.gca().get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='center left', title="Method", bbox_to_anchor=(0.9, 0.5))

    plt.savefig(f'magistrska/ci_comparison_{name}.png', bbox_inches='tight')
    plt.close()


def draw_length_comparison():
    lengths = pd.read_csv('results_10000_reps/length.csv')
    for stat in lengths['statistic'].unique():
        for dgp in lengths['dgp'].unique():
            df = lengths[(lengths['statistic'] == stat) & (lengths['dgp'] == dgp)].dropna(axis=1, how='all')
            if df.shape[0] == 0:
                continue
            df = df.dropna()        # drops all rows where any of the methods has nans
            df_long = pd.melt(df, id_vars=['n'], value_vars=df.columns[6:], var_name='method', value_name='length')

            if np.nanquantile(df_long['length'], 0.05) < 0:     # TEST
                print(dgp, stat, np.nanquantile(df_long['length'], 0.05))

            plot = sns.lineplot(data=df_long, x='n', y='length', hue='method')
            plot.set_xscale('log', base=2)
            plot.set_xticks(df_long['n'].unique())
            plot.set_xticklabels(df_long['n'].unique())
            plot.set_ylim(0, np.nanquantile(df_long['length'], 0.95))
            plt.savefig(f'magistrska/length_comparison_{stat}_{dgp}.png', bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    ns = [8, 8, 8, 8]
    dgps = [DGPNorm(0, 0, 1), DGPNorm(10, 0, 1), DGPNorm(20, 0, 1), DGPNorm(130, 0, 1)]
    stats = [np.mean, np.mean, np.median, np.median]
    # draw_bootstrap_comparison(dgps, ns, stats, name='test')
    draw_length_comparison()
