import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compare_all_coverages(df=None, methods=None, Bs=None, ns=None):
    if df is None:
        cov = pd.read_csv('results/coverage.csv')
    else:
        cov = df

    if Bs is not None:
        cov = cov[cov['B'].isin(Bs)]

    if ns is not None:
        cov = cov[cov['n'].isin(ns)]

    if methods is not None:
        cov = cov[cov['method'].isin(methods)]
        methods_order = methods
    else:
        methods_order = cov['method'].unique()

    distributions = cov['dgp'].unique()

    fig, axs = plt.subplots(ncols=len(distributions), figsize=(20, 8))

    for i in range(len(distributions)):
        df = cov[cov['dgp'] == distributions[i]]
        name = distributions[i][3:].split('_')[0]
        sns.boxplot(x="alpha", hue="method", y="coverage", data=df, ax=axs[i], hue_order=methods_order, fliersize=1)
        axs[i].title.set_text(name)
        if i != len(distributions) - 1:
            axs[i].get_legend().remove()

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def compare_alpha_coverage_by_n(df=None, alpha=0.95,  methods=None, Bs=None, ns=None):
    if df is None:
        cov = pd.read_csv('results/coverage.csv')
    else:
        cov = df

    cov = cov[cov['alpha'] == alpha]

    if Bs is not None:
        cov = cov[cov['B'].isin(Bs)]

    if methods is not None:
        cov = cov[cov['method'].isin(methods)]
        methods_order = methods
    else:
        methods_order = cov['method'].unique()

    distributions = cov['dgp'].unique()

    fig, axs = plt.subplots(ncols=len(distributions), figsize=(20, 8))

    for i in range(len(distributions)):
        df = cov[cov['dgp'] == distributions[i]]
        name = distributions[i][3:].split('_')[0]
        sns.boxplot(x="n", hue="method", y="coverage", data=df, ax=axs[i], hue_order=methods_order, fliersize=1)
        axs[i].title.set_text(name)
        if i != len(distributions) - 1:
            axs[i].get_legend().remove()

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def compare_coverages_grid(cov, filter_by={'alpha': [0.95]}, x='n', row='statistic', col='dgp', hue='method'):
    for key in filter_by.keys():
        cov = cov[cov[key].isin(filter_by[key])]

    g = sns.FacetGrid(cov, row=row, col=col, margin_titles=True)
    g.map(sns.boxplot, x, 'coverage', hue, hue_order=cov[hue].unique())
    g.add_legend()
    plt.show()


if __name__ == '__main__':
    cov = pd.read_csv('results_lab/coverage.csv')  # TODO change from lab and include studentized
    cov = cov[cov['method'] != 'studentized']
    methods = ['percentile', 'bca', 'double', 'ttest', 'maritz_jarrett']
    compare_all_coverages(cov, methods, [10], [5, 10])
    compare_alpha_coverage_by_n(cov, 0.95, methods, [10], [5, 10])

    compare_coverages_grid(cov, filter_by={'alpha': [0.95], 'method': methods})






