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
                for B in [1000]:
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


def aggregate_results(result_folder, methods=None, combined_with='mean'):
    # reading and filtering coverage table
    results = pd.read_csv(f'{result_folder}/results_from_intervals_{combined_with}_bts.csv')    # changed for only wide!

    print('SPEARMAN', results[results['method'] == 'ci_corr_spearman'].shape)
    # coverage = coverage[coverage['B'] == 1000]
    # coverage = coverage[~coverage['dgp'].isin(['DGPBernoulli_0.5', 'DGPBernoulli_0.95'])]
    #                       (coverage['statistic'].isin(['median', 'percentile_5', 'percentile_95'])))]
    if methods is None:
        methods = ['percentile', 'standard', 'basic', 'bc', 'bca', 'double', 'smoothed', 'studentized']
    results = results[results['method'].isin(methods)]

    # calculations for table of closeness to the best method
    results['difference'] = results['coverage'] - results['alpha']
    results['abs_difference'] = abs(results['difference'])
    min_distances = results[['alpha', 'coverage', 'abs_difference', 'dgp', 'statistic', 'n']]\
        .sort_values('abs_difference').groupby(['alpha', 'dgp', 'statistic', 'n']).first()
    results['min_distances'] = results.apply(
        lambda row: min_distances.loc[row['alpha'], row['dgp'], row['statistic'], row['n']]['abs_difference'], axis=1)
    results['best_coverage'] = results.apply(
        lambda row: min_distances.loc[row['alpha'], row['dgp'], row['statistic'], row['n']]['coverage'], axis=1)
    results['std'] = np.sqrt(results['best_coverage'] * (1 - results['best_coverage']) / results['repetitions'])
    results['near_best'] = abs(results['best_coverage'] - results['coverage']) < results['std']

    # calculation for ranks
    results['rank'] = results[['alpha', 'abs_difference', 'dgp', 'statistic', 'n']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).rank()

    # tables
    near_best = results[['method', 'near_best']].groupby(['method']).sum()

    near_best_n = results[['method', 'near_best', 'n']].groupby(['method', 'n']).sum().unstack()
    near_best_n.columns = near_best_n.columns.droplevel()
    near_best = near_best.join(near_best_n)

    near_best_stat = results[['method', 'near_best', 'statistic']].groupby(['method', 'statistic']).sum().unstack()
    near_best_stat.columns = near_best_stat.columns.droplevel()
    near_best = near_best.join(near_best_stat).sort_values(by='near_best', ascending=False)

    avg_rank = results[['method', 'rank']].groupby(['method']).mean()

    avg_rank_n = results[['method', 'rank', 'n']].groupby(['method', 'n']).mean().unstack()
    avg_rank_n.columns = avg_rank_n.columns.droplevel()
    avg_rank = avg_rank.join(avg_rank_n)

    avg_rank_stat = results[['method', 'rank', 'statistic']].groupby(['method', 'statistic']).mean().unstack()
    avg_rank_stat.columns = avg_rank_stat.columns.droplevel()
    avg_rank = avg_rank.join(avg_rank_stat).sort_values(by='rank')

    dist_table = results[['method', 'avg_distance']].groupby(['method']).median()

    dist_table_n = results[['method', 'avg_distance', 'n']].groupby(['method', 'n']).median().unstack()
    dist_table_n.columns = dist_table_n.columns.droplevel()
    dist_table = dist_table.join(dist_table_n)

    dist_table_stat = results[['method', 'avg_distance', 'statistic']]\
        .groupby(['method', 'statistic']).median().unstack()
    dist_table_stat.columns = dist_table_stat.columns.droplevel()
    dist_table = dist_table.join(dist_table_stat).sort_values(by='avg_distance')

    # nans
    nans = results[['method', 'nans']].groupby(['method']).mean()

    nans_n = results[['method', 'nans', 'n']].groupby(['method', 'n']).mean().unstack()
    nans_n.columns = nans_n.columns.droplevel()
    nans = nans.join(nans_n)

    nans_stat = results[['method', 'nans', 'statistic']].groupby(['method', 'statistic']).mean().unstack()
    nans_stat.columns = nans_stat.columns.droplevel()
    nans = nans.join(nans_stat)

    nans = nans[nans['nans'] > 0].sort_values(by='nans', ascending=False)

    # normalization
    for m in avg_rank.index:
        near_best.loc[m, 'near_best'] /= results[results['method'] == m].shape[0]
        for n in results['n'].unique():
            near_best.loc[m, n] /= results[(results['method'] == m) & (results['n'] == n)].shape[0]
        for stat in results['statistic'].unique():
            near_best.loc[m, stat] /= results[(results['method'] == m) & (results['statistic'] == stat)].shape[0]

    # check for differences in method ranking
    results['rank_dist'] = results[['alpha', 'avg_distance', 'dgp', 'statistic', 'n']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).rank()

    # rank diff -> mean, median, kok kerih eksperimentov jih je med tistimi k majo > n razliko

    # gather all nans
    nans_all = results[['method', 'nans', 'statistic', 'repetitions', 'dgp', 'n']].groupby(['method', 'statistic',
                                                                                            'dgp', 'n']).mean()
    nans_all = nans_all[nans_all['nans'] > 0]['nans'].sort_values(ascending=False)

    # t = results[results['method'].isin(['double', 'standard'])] for finding experiment for histogram

    return near_best, avg_rank, dist_table, nans_all


def better_methods(method, result_folder, combined_with='mean'):
    """Finds better methods than proposed one - on experiment level."""
    # reading and filtering coverage table
    results = pd.read_csv(f'{result_folder}/results_combined_{combined_with}.csv')

    # calculations for table of (coverage) closeness to the best method
    results['difference'] = results['coverage'] - results['alpha']
    results['abs_difference'] = abs(results['difference'])
    min_distances = results[['alpha', 'coverage', 'abs_difference', 'dgp', 'statistic', 'n']] \
        .sort_values('abs_difference').groupby(['alpha', 'dgp', 'statistic', 'n']).first()
    results['min_distances'] = results.apply(
        lambda row: min_distances.loc[row['alpha'], row['dgp'], row['statistic'], row['n']]['abs_difference'], axis=1)
    results['best_coverage'] = results.apply(
        lambda row: min_distances.loc[row['alpha'], row['dgp'], row['statistic'], row['n']]['coverage'], axis=1)
    results['std'] = np.sqrt(results['best_coverage'] * (1 - results['best_coverage']) / results['repetitions'])
    results['near_best'] = abs(results['best_coverage'] - results['coverage']) < results['std']

    # tables for how many times another method is better for at least its std
    results['better_cov'] = results[['method', 'alpha', 'coverage', 'abs_difference', 'dgp', 'statistic', 'n',
                                     'repetitions']].groupby(['alpha', 'dgp', 'statistic', 'n']) \
        .apply(better_cov_apply_fn, method)['better']

    # get where distance is better
    results['better_dist'] = results[['method', 'alpha', 'coverage', 'avg_distance', 'dgp', 'statistic', 'n',
                                      'repetitions']].groupby(['alpha', 'dgp', 'statistic', 'n']) \
        .apply(better_dist_apply_fn, method)['better']

    # tables
    results = results[results['method'] != method]                       # filter out method we're comparing with
    better_cov = better_methods_by(results, 'better_cov')

    # DISTANCE FROM EXACT
    better_dist = better_methods_by(results, 'better_dist')

    return better_cov, better_dist


def better_cov_apply_fn(df, method):
    val = df[df['method'] == method]['abs_difference'].values[0]            # value for method we're interested in
    # abs_difference + std of another method is less then abs difference of proposed method
    df['better'] = (df['abs_difference'] + np.sqrt(df['coverage'] * (1 - df['coverage']) / df['repetitions'])) < val
    return df


def better_dist_apply_fn(df, method):
    val = df[df['method'] == method]['avg_distance'].values[0]            # value for method we're interested in
    # avg_distance is less then abs difference of proposed method
    df['better'] = df['avg_distance'] < val
    return df


def better_methods_by(df, criteria, threshold=0.33):
    # TODO join in aggregation function - use that everywhere
    better = df[['method', criteria]].groupby(['method']).sum()

    better_n = df[['method', criteria, 'n']].groupby(['method', 'n']).sum().unstack()
    better_n.columns = better_n.columns.droplevel()
    better = better.join(better_n)

    better_stat = df[['method', criteria, 'statistic']].groupby(['method', 'statistic']).sum().unstack()
    better_stat.columns = better_stat.columns.droplevel()
    better = better.join(better_stat)

    better_dist = df[['method', criteria, 'dgp']].groupby(['method', 'dgp']).sum().unstack()
    better_dist.columns = better_dist.columns.droplevel()
    better = better.join(better_dist)

    # normalization
    for m in better.index:
        better.loc[m, criteria] /= df[df['method'] == m].shape[0]
        for n in df['n'].unique():
            better.loc[m, n] /= df[(df['method'] == m) & (df['n'] == n)].shape[0]
        for stat in df['statistic'].unique():
            better.loc[m, stat] /= df[(df['method'] == m) & (df['statistic'] == stat)].shape[0]
        for dgp in df['dgp'].unique():
            better.loc[m, dgp] /= df[(df['method'] == m) & (df['dgp'] == dgp)].shape[0]

    melted = pd.melt(better, ignore_index=False).sort_values(by='value', ascending=False)
    better_dict = defaultdict(lambda: defaultdict(list))

    for method, values in melted.iterrows():
        if type(values['variable']) == int:
            dimension = 'n'
        elif values['variable'][:6] == 'better':
            dimension = 'all'
        elif values['variable'][:3] == 'DGP':
            dimension = 'dgp'
        else:
            dimension = 'statistic'

        if values['value'] > threshold or dimension == 'all':
            better_dict[method][dimension].append((values['variable'], round(values['value'], 3)))

    for method in better_dict:
        print(method, f'({round(better_dict[method]["all"][0][1], 3)})')
        for where in better_dict[method]:
            if where == 'all':
                continue
            print('     -', where, better_dict[method][where])

    return better


def better_methods_repetition_level(method, results_folder, combined_with='mean'):
    """Finds better methods than proposed one - on repetition level."""
    # read_csv if possible to read whole, else adapt results_from_intervals
    intervals = pd.read_csv(f'{results_folder}/intervals.csv')

    combined = pd.read_csv(f'{results_folder}/results_from_intervals_{combined_with}.csv')

    combined['better_dist'] = combined.apply(better_dist_apply_repetitions, axis=1, intervals=intervals, method=method)
    # combined['better_cov'] = combined.apply(better_cov_apply_repetitions, axis=1, intervals=intervals, method=method)
    combined = combined[combined['method'] != method]       # no point in comparing method with itself
    return


def better_dist_apply_repetitions(row, intervals, method):
    experiment_intervals = intervals[(intervals['alpha'] == row['alpha']) & (intervals['dgp'] == row['dgp']) &
                                     (intervals['statistic'] == row['statistic']) & (intervals['n'] == row['n'])]
    # percentage of methods where another method is closer to exact intervals for at least its std
    perc_better = np.mean(abs(experiment_intervals[method] - experiment_intervals['exact']) <
                          (abs(experiment_intervals[row['method']] - experiment_intervals['exact']) + row['std']))
    return perc_better


def better_cov_apply_repetitions(row, intervals, method):
    # nesmiselno!!
    experiment_int = intervals[(intervals['alpha'] == row['alpha']) & (intervals['dgp'] == row['dgp']) &
                               (intervals['statistic'] == row['statistic']) & (intervals['n'] == row['n'])]
    experiment_int = experiment_int[experiment_int[row['method']] >= experiment_int['true_value']]
    # percentage of intervals that another method covers but the suggested one does not
    perc_bad = np.mean(experiment_int[method] < experiment_int['true_value'])
    return perc_bad


def average_distances_long(folder, combine_dist=np.mean):
    # for long tables (old results), skipping experiments with any nans
    # reading line by line because of 35GB dataframes
    dist_dict = defaultdict(list)         # dict that counts [sum of distances, #]
    nans = defaultdict(int)
    stds = {}
    with open(f'{folder}/distance.csv') as f:
        f.readline()
        for line in f:
            method, alpha, distance, dgp, statistic, n, B, repetitions = line.strip('\n').split(',')
            alpha, n, B, repetitions = float(alpha), int(n), int(B), int(repetitions)
            # if B != '1000' or method == 'ci_corr_spearman' or (statistic in ['percentile_5', 'percentile_95', 'median
            #                                                    and dgp in ['DGPBernoulli_0.5', 'DGPBernoulli_0.95']):
            # TODO which results to include
            if B != 1000 or method == 'ci_corr_spearman' or (dgp in ['DGPBernoulli_0.5', 'DGPBernoulli_0.95']):
                continue

            if distance == '':
                nans[(method, alpha, dgp, statistic, n, repetitions)] += 1
                if (method, alpha, dgp, statistic, n, repetitions) not in dist_dict:
                    # add missing keys for dataframe iteration
                    dist_dict[(method, alpha, dgp, statistic, n, repetitions)] = []
            else:
                distance = float(distance)
                dist_dict[(method, alpha, dgp, statistic, n, repetitions)].append(abs(distance))

                # aggregating results as soon as we can to save memory
                if len(dist_dict[(method, alpha, dgp, statistic, n, repetitions)]) == repetitions:
                    # change list of distances with mean/median distance
                    stds[(method, alpha, dgp, statistic, n, repetitions)] = np.std(
                        dist_dict[(method, alpha, dgp, statistic, n, repetitions)])
                    dist_dict[(method, alpha, dgp, statistic, n, repetitions)] = combine_dist(
                        dist_dict[(method, alpha, dgp, statistic, n, repetitions)])

    avg_distances = pd.DataFrame(columns=['method', 'alpha', 'dgp', 'statistic', 'n', 'repetitions', 'avg_distance',
                                          'std', 'nans'])
    for i, experiment in enumerate(dist_dict.keys()):
        distances = dist_dict[experiment]
        if type(distances) == list:
            if len(distances) == 0:
                avg_dist = np.nan
                std = np.nan
            else:
                avg_dist = combine_dist(distances)
                std = np.std(distances)
        else:
            avg_dist = distances
            std = stds[experiment]

        avg_distances.loc[i] = [*experiment, avg_dist, std, nans[experiment] / experiment[-1]]

    # normalization of distances based on the best method
    avg_distances['avg_distance'] = avg_distances[['alpha', 'dgp', 'statistic', 'n', 'avg_distance']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).transform(lambda x: x / x.min())

    avg_distances.to_csv(f'{folder}/avg_abs_distances_long_{combine_dist.__name__}.csv', index=False)
    return avg_distances


def results_from_intervals(folder, combine_dist=np.mean, include_nan_repetitions=False, only_bts=True):
    # for wide tables (new results), skipping just replications that have nans
    # we are reading results line by line because of possibility of very large dataframes that can't fit in memory
    dist_dict = defaultdict(list)
    nans = defaultdict(int)
    coverage_dict = defaultdict(lambda: [0, 0])
    stds = {}
    bts_methods = ['percentile', 'standard', 'basic', 'bc', 'bca', 'double', 'smoothed', 'studentized']
    stat_methods = {'mean': bts_methods + ['wilcoxon', 'ttest'],
                   'median': bts_methods + ['wilcoxon', 'ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett'],
                   'std': bts_methods + ['chi_sq'],
                   'percentile': bts_methods + ['ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett'],
                   'corr': bts_methods + ['ci_corr_pearson']}
    with open(f'{folder}/intervals.csv') as f:
        keys = f.readline().strip('\n').split(',')      # header
        for line in f:
            line_dict = dict(zip(keys, line.strip('\n').split(',')))
            alpha, dgp, statistic, n, B, repetitions, true_val, exact = [line_dict[name] for name in
                                                                         ['alpha', 'dgp', 'statistic', 'n', 'B',
                                                                          'repetitions', 'true_value', 'exact']]
            alpha, n, B, repetitions, true_val, exact = float(alpha), int(n), int(B), int(repetitions),\
                                                        float(true_val), float(exact)
            if B != 1000:
                # TODO do we want to skip any more results?
                continue

            any_nan = False
            all_methods = bts_methods if only_bts else stat_methods[line_dict['statistic'][:10]]
            for method in all_methods:
                if line_dict[method] == '':
                    any_nan = True
                    nans[(method, alpha, dgp, statistic, n, repetitions)] += 1

                if (method, alpha, dgp, statistic, n, repetitions) not in dist_dict:
                    # add missing keys for dataframe iteration
                    dist_dict[(method, alpha, dgp, statistic, n, repetitions)] = []

            if (not any_nan) or include_nan_repetitions:
                for method in stat_methods[line_dict['statistic'][:10]]:
                    pred = line_dict[method]
                    if pred != '':
                        pred = float(pred)
                        dist_dict[(method, alpha, dgp, statistic, n, repetitions)].append(abs(pred - exact))
                        coverage_dict[(method, alpha, dgp, statistic, n, repetitions)][0] += int(pred >= true_val)
                        coverage_dict[(method, alpha, dgp, statistic, n, repetitions)][1] += 1

                        # aggregating results as soon as we can to save memory
                        if len(dist_dict[(method, alpha, dgp, statistic, n, repetitions)]) == repetitions:
                            # change list of distances with mean/median distance
                            stds[(method, alpha, dgp, statistic, n, repetitions)] = np.std(
                                dist_dict[(method, alpha, dgp, statistic, n, repetitions)])
                            dist_dict[(method, alpha, dgp, statistic, n, repetitions)] = combine_dist(
                                dist_dict[(method, alpha, dgp, statistic, n, repetitions)])

    results = pd.DataFrame(columns=['method', 'alpha', 'dgp', 'statistic', 'n', 'repetitions', 'avg_distance',
                                          'std', 'coverage', 'nans'])
    for i, experiment in enumerate(dist_dict.keys()):
        distances = dist_dict[experiment]
        covers, count = coverage_dict[experiment]
        if type(distances) == list:
            if len(distances) == 0:
                avg_dist = np.nan
                std = np.nan
            else:
                avg_dist = combine_dist(distances)
                std = np.std(distances)
        else:
            avg_dist = distances
            std = stds[experiment]

        results.loc[i] = [*experiment, avg_dist, std, np.nan if count == 0 else covers / count,
                                nans[experiment] / experiment[-1]]

    # normalization of distances based on the best method
    results['avg_distance'] = results[['alpha', 'dgp', 'statistic', 'n', 'avg_distance']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).transform(lambda x: x / x.min())

    results.to_csv(f'{folder}/results_from_intervals_{combine_dist.__name__}{["", "_bts"][int(only_bts)]}'
                   f'{["", "_withnans"][int(include_nan_repetitions)]}.csv', index=False)
    return results


def combine_results(combine_dist='mean', only_bts=True):
    """Combining old (long df) and new results into one df, same shape as the new ones should be."""
    old = pd.read_csv(f'results_10000_reps/avg_abs_distances_long_{combine_dist}.csv')
    old_cov = pd.read_csv('results_10000_reps/coverage.csv')
    old_cov = old_cov[old_cov['B'] == 1000]
    old_cov = old_cov.drop('B', axis=1)
    old_cov = old_cov[(old_cov['method'] != 'ci_corr_spearman') &
                      ~old_cov['dgp'].isin(['DGPBernoulli_0.5', 'DGPBernoulli_0.95'])]
    old = old.merge(old_cov, how='outer', on=['method', 'alpha', 'dgp', 'statistic', 'n', 'repetitions'],
                    validate='one_to_one')
    if only_bts:
        old = old[old['method'].isin(['percentile', 'standard', 'basic', 'bc', 'bca', 'double', 'smoothed',
                                      'studentized'])]

    # delete results that we repeated and Bernoulli
    old = old[old['statistic'].isin(['mean', 'std']) |
              ((old['statistic'] == 'percentile_95') & (old['n'] > 64)) |
              ((old['statistic'] == 'percentile_5') & (old['n'] > 16)) |
              ((old['statistic'] == 'median') & (old['n'] > 4)) |
              ((old['statistic'] == 'corr') & (old['n'] > 8))]

    left_nans = old[old['nans'] > 0].shape[0]
    if left_nans != 0:
        print(left_nans, ' leftover experiments with nans.')

    new = pd.read_csv(f'results_wide_nans/results_from_intervals_{combine_dist}_bts.csv')
    new = new[new['method'] != 'ci_corr_spearman']      # TODO delete

    all_results = pd.concat([old, new])
    all_results.to_csv(f'results/results_combined_{combine_dist}{["", "_bts"][int(only_bts)]}.csv', index=False)


if __name__ == '__main__':
    # folder_add = '_hierarchical'
    # folder_add = '_10000_reps'
    folder_add = ''
    subfolder = ''
    # additional = 'hierarchical'
    additional = ''
    cov = pd.read_csv(f'results{folder_add}/coverage.csv')
    bts_methods = ['percentile', 'standard', 'basic', 'bc', 'bca', 'double', 'smoothed']

    main_plot_comparison(filter_by={}, additional=additional, scale='linear', folder_add=folder_add, set_ylim=True)

    for stat in [np.mean, np.median]:
        for only_bts in [True, False]:
            for include_nans in [True, False]:
                results_from_intervals('results', combine_dist=stat, only_bts=only_bts,
                                       include_nan_repetitions=include_nans)
                # combine_results(stat.__name__, only_bts=only_bts) not needed anymore with complete wide results

    # for stat in ['mean', 'median']:
    #     aggregate_results('results' + folder_add, combined_with=stat)
        # better_methods('double', 'results', stat)

    # better_methods_repetition_level('double', 'results_wide_nans')

    # result_folder = 'results_10000_reps'
    # method = 'double'
    #
    # td = better_methods(method, result_folder)
    #
    # nb, ar, ad, na = aggregate_results('results_10000_reps')
    # debug = True

