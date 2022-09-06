import time

import matplotlib.cm
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compare_all_cov_dis(df=None, comparing='coverage', methods=None, Bs=None, ns=None):
    if df is None:
        df = pd.read_csv(f'results/{comparing}.csv')

    if Bs is not None:
        df = df[df['B'].isin(Bs)]

    if ns is not None:
        df = df[df['n'].isin(ns)]

    if methods is not None:
        df = df[df['method'].isin(methods)]
        methods_order = methods
    else:
        methods_order = df['method'].unique()

    distributions = df['dgp'].unique()

    fig, axs = plt.subplots(ncols=len(distributions), figsize=(20, 8))

    for i in range(len(distributions)):
        dfi = df[df['dgp'] == distributions[i]]
        name = distributions[i][3:].split('_')[0]
        sns.boxplot(x="alpha", hue="method", y=comparing, data=dfi, ax=axs[i], hue_order=methods_order, fliersize=1)
        axs[i].title.set_text(name)
        if i != len(distributions) - 1:
            axs[i].get_legend().remove()

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def compare_alpha_cov_dis_by_n(df=None, comparing='coverage', alpha=0.95,  methods=None, Bs=None, ns=None):
    if df is None:
        df = pd.read_csv(f'results/{comparing}.csv')

    df = df[df['alpha'] == alpha]

    if Bs is not None:
        df = df[df['B'].isin(Bs)]

    if methods is not None:
        df = df[df['method'].isin(methods)]
        methods_order = methods
    else:
        methods_order = df['method'].unique()

    distributions = df['dgp'].unique()

    fig, axs = plt.subplots(ncols=len(distributions), figsize=(20, 8))

    for i in range(len(distributions)):
        dfi = df[df['dgp'] == distributions[i]]
        name = distributions[i][3:].split('_')[0]
        sns.boxplot(x="n", hue="method", y=comparing, data=dfi, ax=axs[i], hue_order=methods_order, fliersize=1)
        axs[i].title.set_text(name)
        if i != len(distributions) - 1:
            axs[i].get_legend().remove()

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def compare_cov_dis_grid(df=None, comparing='coverage', filter_by={'alpha': [0.95]}, x='n', row='statistic', col='dgp',
                         hue='method', save_add=None, title=None, ci=95, scale='linear'):
    if df is None:
        df = pd.read_csv(f'results/{comparing}.csv')

    for key in filter_by.keys():
        df = df[df[key].isin(filter_by[key])]

    nm = df['method'].nunique()
    if nm > 10:
        cols = plt.cm.tab20(np.linspace(0.05, 0.95, df['method'].nunique()))
    else:
        cols = plt.cm.tab10(np.linspace(0.05, 0.95, df['method'].nunique()))
    colors = {m: c for (m, c) in zip(df['method'].unique(), cols)}

    g = sns.FacetGrid(df, row=row, col=col, margin_titles=True, sharex=True, sharey='row', palette=colors)
    if comparing == 'coverage':
        g.map_dataframe(plot_coverage_bars, colors=cols, ci=ci, scale=scale)
    else:
        g.map(sns.boxplot, x, comparing, hue, hue_order=df[hue].unique(), fliersize=0, whis=[(100-ci)/2, 50 + ci/2],
              palette=colors)
        ylim = np.nanquantile(df['distance'], (0.01, 0.99))
        g.set(ylim=ylim)

        for axs in g.axes:
            for ax in axs:
                ax.axhline(0, linestyle='--', color='gray')

    g.add_legend(title='method')

    if title is not None:
        g.fig.subplots_adjust(top=0.95)
        g.fig.suptitle(title, fontsize=16)

    if save_add is not None:
        plt.savefig(f'images/comparison/only_bts/compare_{comparing}_{x}_{row}_{col}_{save_add}.png')
        plt.close()
    else:
        plt.show()


def plot_coverage_bars(data, **kwargs):
    colors = kwargs['colors']
    ci = kwargs['ci']
    scale = kwargs['scale']
    data['ci'] = np.sqrt(data['coverage'] * (1 - data['coverage']) / data['repetitions'])
    if ci != 'se':
        data['ci'] *= scipy.stats.norm.ppf(0.5 + ci/200)
    data['low'] = data['coverage'] - data['ci']

    n_levels = len(data['method'].unique())
    group_width = 0.8
    bar_width = group_width / n_levels
    offsets = np.linspace(0, group_width - bar_width, n_levels)
    offsets -= offsets.mean()

    bar_pos = np.arange(data['n'].nunique())
    for i, method in enumerate(data['method'].unique()):
        data_m = data[data['method'] == method]
        offset = bar_pos + offsets[i]
        plt.bar(offset, data_m['ci'], bar_width, bottom=data_m['coverage'], ec='k', label=method, color=colors[i])
        plt.bar(offset, data_m['ci'], bar_width, bottom=data_m['low'], ec='k', color=colors[i])

    a = data['alpha'].values[0]
    if a > 0.9:
        if scale == 'logit':
            ylim = (0.8, 0.99)
        else:
            ylim = (0.8, 1)
    elif a < 0.1:
        ylim = (0, 0.2)
    else:
        ylim = (a - 0.1, a + 0.1)

    ax = plt.gca()

    ax.set_yscale(scale)

    ax.set(ylim=ylim)

    ax.axhline(a, linestyle='--', color='gray')

    ax.set_xlabel('n')
    ax.set_ylabel('coverage')
    plt.xticks(bar_pos, sorted(data['n'].unique()))


def main_plot_comparison(B_as_method=False, filter_by={}, additional='', scale='linear'):
    for comparing in ['coverage']:
        df = pd.read_csv(f'results_lab2/{comparing}.csv')   # TODO change
        df = df[df['method'] != 'studentized']              # TODO delete
        for statistic in ['mean', 'median', 'std', 'percentile_5', 'percentile_95', 'corr']:
            if B_as_method:
                # a povprečit a vzet samo enega od B-jev za ostale metode?
                pass
            else:
                for B in [10, 100, 1000]:
                    df_part = df[(df['B'] == B) & (df['statistic'] == statistic)]
                    if df_part.shape[0] == 0:
                        print(statistic, B)
                        continue
                    title = f'{comparing}s for {statistic} using B = {B}'
                    compare_cov_dis_grid(df_part, comparing=comparing, filter_by=filter_by, x='n', row='alpha',
                                         col='dgp', title=title, save_add=f'{statistic}_{B}{additional}', scale=scale)


if __name__ == '__main__':
    cov = pd.read_csv('results_lab/coverage.csv')  # TODO change from lab and include studentized
    cov = cov[cov['method'] != 'studentized']
    bts_methods = ['percentile', 'standard', 'basic', 'bc', 'bca', 'double', 'smoothed']

    main_plot_comparison(filter_by={'method': bts_methods}, additional='_only_bts_logit', scale='logit')

    # compare_all_cov_dis(cov, 'coverage', methods, [10], [5, 10])
    # compare_alpha_cov_dis_by_n(cov, 'coverage', 0.95, methods, [10], [5, 10])

    # compare_cov_dis_grid(cov, filter_by={'method': methods}, x='n', row='alpha', col='dgp', title='kdsfl kdfjs lk')
    # compare_cov_dis_grid(cov, filter_by={'alpha': [0.95], 'method': methods})
    # compare_cov_dis_grid(cov, filter_by={'n': [100], 'method': methods}, x='alpha')
    # compare_cov_dis_grid(cov, filter_by={'method': methods}, x='alpha', row='n')

    # dis = pd.read_csv('results_lab/distance.csv')
    # dis = dis[dis['method'] != 'studentized']
    #
    # compare_all_cov_dis(dis, 'distance', methods, [10], [5, 10])
    # compare_alpha_cov_dis_by_n(dis, 'distance', 0.95, methods, [10], [5, 10])
    #
    # compare_cov_dis_grid(dis, 'distance', filter_by={'alpha': [0.95], 'method': methods})
    # compare_cov_dis_grid(dis, 'distance', filter_by={'n': [100], 'method': methods}, x='alpha')
    # compare_cov_dis_grid(dis, 'distance', filter_by={'method': methods}, x='alpha', row='n')






