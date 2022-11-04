import time
from collections import defaultdict
from os.path import exists

import matplotlib.cm
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compare_all_cov_dis(df=None, comparing='coverage', methods=None, Bs=None, ns=None, folder_add=''):
    if df is None:
        df = pd.read_csv(f'results{folder_add}/{comparing}.csv')

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


def compare_alpha_cov_dis_by_n(df=None, comparing='coverage', alpha=0.95,  methods=None, Bs=None, ns=None,
                               folder_add=''):
    if df is None:
        df = pd.read_csv(f'results{folder_add}/{comparing}.csv')

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
                         hue='method', save_add=None, title=None, ci=95, scale='linear', folder_add='', subfolder='',
                         set_ylim=False, colors=None):
    if df is None:
        df = pd.read_csv(f'results{folder_add}/{comparing}.csv')

    for key in filter_by.keys():
        df = df[df[key].isin(filter_by[key])]

    if colors is None:
        nm = df['method'].nunique()
        if nm > 10:
            cols = plt.cm.tab20(np.linspace(0.05, 0.95, df['method'].nunique()))
        else:
            cols = plt.cm.tab10(np.linspace(0.05, 0.95, df['method'].nunique()))
        colors = {m: c for (m, c) in zip(df['method'].unique(), cols)}

    g = sns.FacetGrid(df, row=row, col=col, margin_titles=True, sharex=True, sharey='row', palette=colors, aspect=2)
    if comparing == 'coverage':
        g.map_dataframe(plot_coverage_bars, colors=colors, ci=ci, scale=scale, set_ylim=set_ylim,
                        order=df[hue].unique(), hue=hue, x=x)
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
        plt.savefig(f'images{folder_add}/comparison/{subfolder}/compare_{comparing}_{x}_{row}_{col}_{save_add}.png')
        print('saved')
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

    n_levels = len(kwargs['order'])
    group_width = 0.8
    bar_width = group_width / n_levels
    offsets = np.linspace(0, group_width - bar_width, n_levels)
    offsets -= offsets.mean()

    bar_pos = np.arange(data[kwargs['x']].nunique())
    for i, method in enumerate(kwargs['order']):
        data_m = data[data[kwargs['hue']] == method]
        offset = bar_pos + offsets[i]
        # if data_m['ci'].shape[0] == 0:
        #     continue
        plt.bar(offset, data_m['ci'], bar_width, bottom=data_m['coverage'], label=method, color=colors[method],
                ec=colors[method])
        plt.bar(offset, data_m['ci'], bar_width, bottom=data_m['low'], color=colors[method], ec=colors[method])
        # TODO a rabmo še črtico dodatno?

    for p in bar_pos[:-1]:
        plt.axvline(p + 0.5, ls=':', alpha=0.2)

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

    if kwargs['set_ylim']:
        ax.set(ylim=ylim)

    ax.axhline(a, linestyle='--', color='gray')

    ax.set_xlabel(kwargs['x'])
    ax.set_ylabel('coverage')
    plt.xticks(bar_pos, sorted(data[kwargs['x']].unique()))


def main_plot_comparison(B_as_method=False, filter_by={}, additional='', scale='linear', folder_add='', set_ylim=True,
                         levels=None, stds=None):
    # for comparing in ['coverage', 'distance']:
    for comparing in ['coverage']:
        df = pd.read_csv(f'results{folder_add}/{comparing}.csv')
        if 'method' in filter_by:
            nm = len(filter_by['method'])
        else:
            nm = df['method'].nunique()
        if nm > 10:
            cols = plt.cm.tab20(np.linspace(0.05, 0.95, df['method'].nunique()))
        else:
            cols = plt.cm.tab10(np.linspace(0.05, 0.95, df['method'].nunique()))
        colors = {m: c for (m, c) in zip(df['method'].unique(), cols)}
        for statistic in ['mean', 'median', 'std', 'percentile_5', 'percentile_95', 'corr']:
            if B_as_method:
                # TODO if comparing more Bs for one method on one plot
                # a povprečit a vzet samo enega od B-jev za ostale metode?
                pass
            else:
                for B in [10, 100, 1000]:
                    df_part = df[(df['B'] == B) & (df['statistic'] == statistic)]

                    if additional == 'hierarchical':
                        for level in levels:
                            for std in stds:
                                df_part_part = df_part[(df_part['levels'] == level) & (df_part['std'] == std)]
                                if df_part_part.shape[0] == 0:
                                    print('Empty df for: ', statistic, B, level)
                                    continue
                                title = f'{comparing}s for {statistic} using B = {B}, {level} levels, std {std}'
                                compare_cov_dis_grid(df_part_part, comparing=comparing, filter_by=filter_by,
                                                     x='n_leaves', row='alpha', col='n_branches', title=title,
                                                     save_add=f'{statistic}_{B}_{level}_{std}{additional}', scale=scale,
                                                     folder_add=folder_add, set_ylim=set_ylim, colors=colors)

                    else:
                        if df_part.shape[0] == 0:
                            print('Empty df for: ', statistic, B)
                            continue
                        title = f'{comparing}s for {statistic} using B = {B}'
                        subfolder = '' if set_ylim else 'noylim'
                        compare_cov_dis_grid(df_part, comparing=comparing, filter_by=filter_by, x='n', row='alpha',
                                             col='dgp', title=title, save_add=f'{statistic}_{B}{additional}',
                                             scale=scale, folder_add=folder_add, set_ylim=set_ylim, subfolder=subfolder,
                                             colors=colors)
        del df      # clear space


def plot_times_lengths_grid(comparing='times', filter_by: dict = None, title=None, save_add=None, scale='linear',
                            ci=95, folder_add='', subfolder=''):
    df = pd.read_csv(f'results{folder_add}/{comparing}.csv')

    if filter_by is not None:
        for key in filter_by.keys():
            df = df[df[key].isin(filter_by[key])]

    id_vars = ['dgp', 'statistic', 'n', 'B', 'repetitions']
    if comparing == 'length':
        id_vars.append('CI')
    df_long = pd.melt(df, id_vars=id_vars).dropna()

    # df_long = df_long[df_long['variable'] != 'studentized']        # TODO DELETE

    nm = df_long['variable'].nunique()
    if nm > 10:
        cols = plt.cm.tab20(np.linspace(0.05, 0.95, df_long['variable'].nunique()))
    else:
        cols = plt.cm.tab10(np.linspace(0.05, 0.95, df_long['variable'].nunique()))
    colors = {m: c for (m, c) in zip(df_long['variable'].unique(), cols)}

    g = sns.FacetGrid(df_long, row='n', col='statistic', margin_titles=True, palette=colors)
    g.map(sns.boxplot, 'B', 'value', 'variable', hue_order=df_long['variable'].unique(), palette=colors, fliersize=0,
          whis=[(100-ci)/2, 50 + ci/2])
    g.add_legend(title='method')

    g.set(yscale=scale)
    g.set_axis_labels(y_var=comparing)

    if title is not None:
        g.fig.subplots_adjust(top=0.95)
        g.fig.suptitle(title, fontsize=16)

    if save_add is not None:
        plt.savefig(f'images{folder_add}/comparison/{subfolder}/{comparing}_{save_add}.png')
        plt.close()
    else:
        plt.show()


def compare_variances():
    intervals = pd.read_csv('results_hierarchical/intervals.csv')
    intervals['var_ratio'] = intervals['mean_var'] / intervals['gt_variance']

    intervals = intervals[intervals['method'] != 'exact']
    intervals['strat'] = intervals['method'].apply(lambda x: x.split('_')[1])
    res_int = intervals[['strat', 'var_ratio']].groupby('strat').mean().sort_values(by='var_ratio')

    coverage = pd.read_csv('results_hierarchical/coverage.csv')
    # cov1 = cov[::2]
    coverage['strat'] = coverage['method'].apply(lambda x: x.split('_')[1])
    res_cov = coverage[['strat', 'var_coverage']].groupby('strat').mean().sort_values(by='var_coverage')

    return res_int, res_cov


def aggregate_results(result_folder, methods=None):
    # reading and filtering coverage table
    coverage = pd.read_csv(f'{result_folder}/coverage.csv')
    coverage = coverage[coverage['B'] == 1000]
    coverage = coverage[~coverage['dgp'].isin(['DGPBernoulli_0.5', 'DGPBernoulli_0.95'])]
    #                       (coverage['statistic'].isin(['median', 'percentile_5', 'percentile_95'])))]
    if methods is None:
        methods = ['percentile', 'standard', 'basic', 'bc', 'bca', 'double', 'smoothed', 'studentized']
    coverage = coverage[coverage['method'].isin(methods)]

    # calculations for table of closeness to the best method
    coverage['difference'] = coverage['coverage'] - coverage['alpha']
    coverage['abs_difference'] = abs(coverage['difference'])
    min_distances = coverage[['alpha', 'coverage', 'abs_difference', 'dgp', 'statistic', 'n']]\
        .sort_values('abs_difference').groupby(['alpha', 'dgp', 'statistic', 'n']).first()
    coverage['min_distances'] = coverage.apply(
        lambda row: min_distances.loc[row['alpha'], row['dgp'], row['statistic'], row['n']]['abs_difference'], axis=1)
    coverage['best_coverage'] = coverage.apply(
        lambda row: min_distances.loc[row['alpha'], row['dgp'], row['statistic'], row['n']]['coverage'], axis=1)
    coverage['std'] = np.sqrt(coverage['best_coverage'] * (1 - coverage['best_coverage']) / coverage['repetitions'])
    coverage['near_best'] = abs(coverage['best_coverage'] - coverage['coverage']) < coverage['std']

    # calculation for ranks
    coverage['rank'] = coverage[['alpha', 'abs_difference', 'dgp', 'statistic', 'n', 'B']].groupby(
        ['alpha', 'dgp', 'statistic', 'n', 'B']).rank()

    # distances
    if not exists(f'{result_folder}/avg_abs_distances.csv'):
        average_distances_long(result_folder)
    avg_distances = pd.read_csv(f'{result_folder}/avg_abs_distances.csv')

    avg_distances = avg_distances[avg_distances['method'].isin(methods)]
    # TODO delete (potrebno samo za trenutno analizo z odvečnimi metodami - kasneje te izločimo vnaprej)
    avg_distances = avg_distances[~avg_distances['dgp'].isin(['DGPBernoulli_0.5', 'DGPBernoulli_0.95'])]
    #                                 (avg_distances['statistic'].isin(['median', 'percentile_5', 'percentile_95'])))]

    # tables
    near_best = coverage[['method', 'near_best']].groupby(['method']).sum()

    near_best_n = coverage[['method', 'near_best', 'n']].groupby(['method', 'n']).sum().unstack()
    near_best_n.columns = near_best_n.columns.droplevel()
    near_best = near_best.join(near_best_n)

    near_best_stat = coverage[['method', 'near_best', 'statistic']].groupby(['method', 'statistic']).sum().unstack()
    near_best_stat.columns = near_best_stat.columns.droplevel()
    near_best = near_best.join(near_best_stat).sort_values(by='near_best', ascending=False)

    avg_rank = coverage[['method', 'rank']].groupby(['method']).mean()

    avg_rank_n = coverage[['method', 'rank', 'n']].groupby(['method', 'n']).mean().unstack()
    avg_rank_n.columns = avg_rank_n.columns.droplevel()
    avg_rank = avg_rank.join(avg_rank_n)

    avg_rank_stat = coverage[['method', 'rank', 'statistic']].groupby(['method', 'statistic']).mean().unstack()
    avg_rank_stat.columns = avg_rank_stat.columns.droplevel()
    avg_rank = avg_rank.join(avg_rank_stat).sort_values(by='rank')

    dist_table = avg_distances[['method', 'avg_distance']].groupby(['method']).median()

    dist_table_n = avg_distances[['method', 'avg_distance', 'n']].groupby(['method', 'n']).median().unstack()
    dist_table_n.columns = dist_table_n.columns.droplevel()
    dist_table = dist_table.join(dist_table_n)

    dist_table_stat = avg_distances[['method', 'avg_distance', 'statistic']]\
        .groupby(['method', 'statistic']).median().unstack()
    dist_table_stat.columns = dist_table_stat.columns.droplevel()
    dist_table = dist_table.join(dist_table_stat).sort_values(by='avg_distance')

    # nans
    nans = avg_distances[['method', 'nans', 'repetitions']].groupby(['method']).sum()
    nans['nans'] /= nans['repetitions']
    nans = nans.drop('repetitions', axis=1)

    nans_n = avg_distances[['method', 'nans', 'n', 'repetitions']].groupby(['method', 'n']).sum().unstack()
    nans_n['nans'] /= nans_n['repetitions']
    nans_n = nans_n.drop('repetitions', axis=1)
    nans_n.columns = nans_n.columns.droplevel()
    nans = nans.join(nans_n)

    nans_stat = avg_distances[['method', 'nans', 'statistic', 'repetitions']].groupby(['method',
                                                                                       'statistic']).sum().unstack()
    nans_stat['nans'] /= nans_stat['repetitions']
    nans_stat = nans_stat.drop('repetitions', axis=1)
    nans_stat.columns = nans_stat.columns.droplevel()
    nans = nans.join(nans_stat)

    nans = nans[nans['nans'] > 0].sort_values(by='nans', ascending=False)

    # normalization
    for m in avg_rank.index:
        near_best.loc[m, 'near_best'] /= coverage[coverage['method'] == m].shape[0]
        for n in coverage['n'].unique():
            near_best.loc[m, n] /= coverage[(coverage['method'] == m) & (coverage['n'] == n)].shape[0]
        for stat in coverage['statistic'].unique():
            near_best.loc[m, stat] /= coverage[(coverage['method'] == m) & (coverage['statistic'] == stat)].shape[0]

    # combine tables, check differences in method ranking
    joined_df = coverage.merge(avg_distances, on=['method', 'alpha', 'dgp', 'statistic', 'n', 'repetitions'],
                               how='outer')
    joined_df['rank_dist'] = joined_df[['alpha', 'avg_distance', 'dgp', 'statistic', 'n', 'B']].groupby(
        ['alpha', 'dgp', 'statistic', 'n', 'B']).rank()

    # rank diff -> mean, median, kok kerih eksperimentov jih je med tistimi k majo > n razliko

    return near_best, avg_rank, dist_table, nans


def better_methods(method, result_folder):
    # reading and filtering coverage table
    coverage = pd.read_csv(f'{result_folder}/coverage.csv')
    coverage = coverage[coverage['B'] == 1000]
    coverage = coverage[~(coverage['dgp'].isin(['DGPBernoulli_0.5', 'DGPBernoulli_0.95']) &
                          (coverage['statistic'].isin(['median', 'percentile_5', 'percentile_95'])))]

    # calculations for table of closeness to the best method
    coverage['difference'] = coverage['coverage'] - coverage['alpha']
    coverage['abs_difference'] = abs(coverage['difference'])
    min_distances = coverage[['alpha', 'coverage', 'abs_difference', 'dgp', 'statistic', 'n']] \
        .sort_values('abs_difference').groupby(['alpha', 'dgp', 'statistic', 'n']).first()
    coverage['min_distances'] = coverage.apply(
        lambda row: min_distances.loc[row['alpha'], row['dgp'], row['statistic'], row['n']]['abs_difference'], axis=1)
    coverage['best_coverage'] = coverage.apply(
        lambda row: min_distances.loc[row['alpha'], row['dgp'], row['statistic'], row['n']]['coverage'], axis=1)
    coverage['std'] = np.sqrt(coverage['best_coverage'] * (1 - coverage['best_coverage']) / coverage['repetitions'])
    coverage['near_best'] = abs(coverage['best_coverage'] - coverage['coverage']) < coverage['std']

    # calculation for ranks
    coverage['rank'] = coverage[['alpha', 'abs_difference', 'dgp', 'statistic', 'n', 'B']].groupby(
        ['alpha', 'dgp', 'statistic', 'n', 'B']).rank()

    # tables for how many times another method is better
    coverage['better_diff'] = coverage[['method', 'alpha', 'coverage', 'abs_difference', 'dgp', 'statistic', 'n',
                                        'repetitions']].groupby(['alpha', 'dgp', 'statistic', 'n']) \
        .apply(better_diff_apply_fn, method)['better']

    # tables
    coverage = coverage[coverage['method'] != method]                       # filter out method we're comparing with
    better_diff = coverage[['method', 'better_diff']].groupby(['method']).sum()

    better_diff_n = coverage[['method', 'better_diff', 'n']].groupby(['method', 'n']).sum().unstack()
    better_diff_n.columns = better_diff_n.columns.droplevel()
    better_diff = better_diff.join(better_diff_n)

    better_diff_stat = coverage[['method', 'better_diff', 'statistic']].groupby(['method', 'statistic']).sum().unstack()
    better_diff_stat.columns = better_diff_stat.columns.droplevel()
    better_diff = better_diff.join(better_diff_stat)

    better_diff_dist = coverage[['method', 'better_diff', 'dgp']].groupby(['method', 'dgp']).sum().unstack()
    better_diff_dist.columns = better_diff_dist.columns.droplevel()
    better_diff = better_diff.join(better_diff_dist)

    # normalization
    for m in better_diff.index:
        better_diff.loc[m, 'better_diff'] /= coverage[coverage['method'] == m].shape[0]
        for n in coverage['n'].unique():
            better_diff.loc[m, n] /= coverage[(coverage['method'] == m) & (coverage['n'] == n)].shape[0]
        for stat in coverage['statistic'].unique():
            better_diff.loc[m, stat] /= coverage[(coverage['method'] == m) & (coverage['statistic'] == stat)].shape[0]
        for dgp in coverage['dgp'].unique():
            better_diff.loc[m, dgp] /= coverage[(coverage['method'] == m) & (coverage['dgp'] == dgp)].shape[0]

    return pd.melt(better_diff, ignore_index=False).sort_values(by='value', ascending=False)


def better_diff_apply_fn(df, method):
    val = df[df['method'] == method]['abs_difference'].values[0]            # value for method we're interested in
    # abs_difference + std of another method is less then abs difference of double
    df['better'] = (df['abs_difference'] + np.sqrt(df['coverage'] * (1 - df['coverage']) / df['repetitions'])) < val

    return df


def average_distances_long(folder):
    # for long tables (old results), skipping experiments with any nans
    dist_dict = defaultdict(lambda: [0, 0])         # dict that counts [sum of distances, #]
    nans = defaultdict(int)
    with open(f'{folder}/distance.csv') as f:
        f.readline()
        for line in f:
            method, alpha, distance, dgp, statistic, n, B, repetitions = line.strip('\n').split(',')
            alpha, n, B, repetitions = float(alpha), int(n), int(B), int(repetitions)
            # if B != '1000' or method == 'ci_corr_spearman' or (statistic in ['percentile_5', 'percentile_95', 'median
            #                                                    and dgp in ['DGPBernoulli_0.5', 'DGPBernoulli_0.95']):
            # TODO which results to include
            if B != 1000 or method == 'ci_corr_spearman' or (dgp in ['DGPBernoulli_0.5', 'DGPBernoulli_0.95']) \
                    or (statistic in ['percentile_95', 'percentile_5', 'corr'] and n == 4):
                continue
            if distance == '':
                nans[(method, alpha, dgp, statistic, n, repetitions)] += 1
                if (method, alpha, dgp, statistic, n, repetitions) not in dist_dict:
                    # add missing keys for dataframe iteration
                    dist_dict[(method, alpha, dgp, statistic, n, repetitions)] = [0, 0]
            else:
                distance = float(distance)
                dist_dict[(method, alpha, dgp, statistic, n, repetitions)][0] += abs(distance)
                dist_dict[(method, alpha, dgp, statistic, n, repetitions)][1] += 1

    avg_distances = pd.DataFrame(columns=['method', 'alpha', 'dgp', 'statistic', 'n', 'repetitions', 'avg_distance',
                                          'nans'])
    for i, experiment in enumerate(dist_dict.keys()):
        distance, reps = dist_dict[experiment]
        if reps == 0:
            avg_dist = np.nan
        else:
            avg_dist = distance / reps
        avg_distances.loc[i] = [*experiment, avg_dist, nans[experiment] / experiment[-1]]

    avg_distances.to_csv(f'{folder}/avg_abs_distances_long.csv', index=False)
    return avg_distances


def average_distances_wide(folder, include_nan_repetitions=False, combine=np.mean):
    # for wide tables (new results), skipping just replications that have nans
    dist_dict = defaultdict(list)
    nans = defaultdict(int)
    bts_methods = ['percentile', 'standard', 'basic', 'bc', 'bca', 'double', 'smoothed', 'studentized']
    stat_methods = {'mean': bts_methods + ['wilcoxon', 'ttest'],
                   'median': bts_methods + ['wilcoxon', 'ci_quant_param', 'ci_quant_nonparam', 'maritz_jarrett'],
                   'std': bts_methods + ['chi_sq'],
                   'percentile': bts_methods + ['ci_quant_param', 'ci_quant_nonparam', 'maritz_jarrett'],
                   'corr': bts_methods + ['ci_corr_pearson', 'ci_corr_spearman']}
    with open(f'{folder}/distance.csv') as f:
        keys = f.readline().strip('\n').split(',')      # header
        for line in f:
            line_dict = dict(zip(keys, line.strip('\n').split(',')))
            alpha, dgp, statistic, n, B, repetitions = [line_dict[name] for name in ['alpha', 'dgp', 'statistic', 'n',
                                                                                     'B', 'repetitions']]
            alpha, n, B, repetitions = float(alpha), int(n), int(B), int(repetitions)
            if B != 100:
                # TODO do we want to skip any more results?
                continue

            any_nan = False
            for method in stat_methods[line_dict['statistic'][:10]]:
                if line_dict[method] == '':
                    any_nan = True
                    nans[(method, alpha, dgp, statistic, n, repetitions)] += 1

                if (method, alpha, dgp, statistic, n, repetitions) not in dist_dict:
                    # add missing keys for dataframe iteration
                    dist_dict[(method, alpha, dgp, statistic, n, repetitions)] = []

            if (not any_nan) or include_nan_repetitions:
                for method in stat_methods[line_dict['statistic'][:10]]:
                    distance = line_dict[method]
                    if distance != '':
                        distance = float(distance)
                        dist_dict[(method, alpha, dgp, statistic, n, repetitions)].append(abs(distance))

                        # aggregating results as soon as we can to save memory
                        if len(dist_dict[(method, alpha, dgp, statistic, n, repetitions)]) == repetitions:
                            dist_dict[(method, alpha, dgp, statistic, n, repetitions)] = combine(
                                dist_dict[(method, alpha, dgp, statistic, n, repetitions)])

    avg_distances = pd.DataFrame(columns=['method', 'alpha', 'dgp', 'statistic', 'n', 'repetitions', 'avg_distance',
                                          'nans'])
    for i, experiment in enumerate(dist_dict.keys()):
        distances = dist_dict[experiment]
        if type(distances) == list:
            if len(distances) == 0:
                avg_dist = np.nan
            else:
                avg_dist = combine(distances)
        else:
            avg_dist = distances
        avg_distances.loc[i] = [*experiment, avg_dist, nans[experiment] / experiment[-1]]

    # normalization of distances based on the best method
    avg_distances['avg_distance'] = avg_distances[['alpha', 'dgp', 'statistic', 'n', 'avg_distance']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).transform(lambda x: x / x.min())

    avg_distances.to_csv(f'{folder}/avg_abs_distances_wide_{combine.__name__}.csv', index=False)
    return avg_distances


if __name__ == '__main__':
    # folder_add = '_hierarchical'
    # folder_add = '_10000_reps'
    folder_add = ''
    subfolder = ''
    # additional = 'hierarchical'
    additional = ''
    cov = pd.read_csv(f'results{folder_add}/coverage.csv')
    bts_methods = ['percentile', 'standard', 'basic', 'bc', 'bca', 'double', 'smoothed']

    # main_plot_comparison(filter_by={}, additional=additional, scale='linear', folder_add=folder_add, levels=[2, 3],
    #                      stds=[0.1, 1, 10], set_ylim=True)

    # for c in compare_variances():
    #     print(c)

    # for t in aggregate_results('results_10000_reps'):
    #     print(t)

    average_distances_wide('results', combine=np.median)

    # result_folder = 'results_10000_reps'
    # method = 'double'
    #
    # td = better_methods(method, result_folder)
    #
    # nb, ar, ad, na = aggregate_results('results_10000_reps')
    # debug = True

