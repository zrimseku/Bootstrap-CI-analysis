import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compare_all_coverages(methods=None, Bs=None, ns=None):
    cov = pd.read_csv('results_lab/coverage.csv')  # TODO change from lab and include studentized
    cov = cov[cov['method'] != 'studentized']

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



if __name__ == '__main__':
    compare_all_coverages(['percentile', 'bca', 'double', 'ttest', 'maritz_jarrett'], [10], [5, 10])






