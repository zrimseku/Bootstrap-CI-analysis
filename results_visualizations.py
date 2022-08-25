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

    plt.legend(loc="center right", bbox_to_anchor=(1, 1))
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


def compare_cov_dis_grid(df=None, comparing='coverage', filter_by={'alpha': [0.95]}, x='n', row='statistic', col='dgp', hue='method'):
    if df is None:
        df = pd.read_csv(f'results/{comparing}.csv')

    for key in filter_by.keys():
        df = df[df[key].isin(filter_by[key])]

    g = sns.FacetGrid(df, row=row, col=col, margin_titles=True)
    g.map(sns.boxplot, x, comparing, hue, hue_order=df[hue].unique())
    g.add_legend()
    plt.show()


if __name__ == '__main__':
    cov = pd.read_csv('results_lab/coverage.csv')  # TODO change from lab and include studentized
    cov = cov[cov['method'] != 'studentized']
    methods = ['percentile', 'bca', 'double', 'ttest', 'maritz_jarrett']
    # compare_all_cov_dis(cov, methods, [10], [5, 10])
    # compare_alpha_cov_dis_by_n(cov, 0.95, methods, [10], [5, 10])
    #
    # compare_cov_dis_grid(cov, filter_by={'alpha': [0.95], 'method': methods})
    # compare_cov_dis_grid(cov, filter_by={'n': [100], 'method': methods}, x='alpha')
    # compare_cov_dis_grid(cov, filter_by={'method': methods}, x='alpha', row='n')

    dis = pd.read_csv('results_lab/distance.csv')
    dis = dis[dis['method'] != 'studentized']

    compare_all_cov_dis(dis, 'distance', methods, [10], [5, 10])
    compare_alpha_cov_dis_by_n(dis, 'distance', 0.95, methods, [10], [5, 10])

    compare_cov_dis_grid(dis, 'distance', filter_by={'alpha': [0.95], 'method': methods})
    compare_cov_dis_grid(dis, 'distance', filter_by={'n': [100], 'method': methods}, x='alpha')
    compare_cov_dis_grid(dis, 'distance', filter_by={'method': methods}, x='alpha', row='n')






