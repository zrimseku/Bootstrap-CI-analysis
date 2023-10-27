import time
from collections import defaultdict
from os.path import exists

import matplotlib.cm
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import ticker


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


def compare_alpha_cov_dis_by_n(df=None, comparing='coverage', alpha=0.95, methods=None, Bs=None, ns=None,
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
        g.map(sns.boxplot, x, comparing, hue, hue_order=df[hue].unique(), fliersize=0,
              whis=[(100 - ci) / 2, 50 + ci / 2],
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

    if 'cov_kind' in kwargs:                # for the possibility of plotting variance coverage with it
        cov_kind = kwargs['cov_kind']
    else:
        cov_kind = 'coverage'

    data['ci'] = np.sqrt(data[cov_kind] * (1 - data[cov_kind]) / data['repetitions'])
    if ci != 'se':
        data['ci'] *= scipy.stats.norm.ppf(0.5 + ci / 200)
    data['low'] = data[cov_kind] - data['ci']

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
        plt.bar(offset, data_m['ci'], bar_width, bottom=data_m[cov_kind], label=method, color=colors[method],
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
    plt.yticks(list(plt.yticks()[0]) + [a])

    ax.set_xlabel(kwargs['x'])
    ax.set_ylabel(cov_kind)
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
        del df  # clear space


def plot_times_line():
    df = pd.read_csv(f'results/times.csv')
    for stat in ['mean', 'median', 'std', 'percentile_5', 'percentile_95', 'corr']:
        filter_by = {'B': [1000], 'statistic': [stat]}
        df_part = df.copy()
        for key in filter_by.keys():
            df_part = df_part[df_part[key].isin(filter_by[key])]
            df_part[key] = np.nan
        df_na = df_part.dropna(axis=1, how='all')

        id_vars = [var for var in ['dgp', 'statistic', 'n', 'B', 'repetitions'] if var not in filter_by.keys()]
        df_long = pd.melt(df_na, id_vars=id_vars, value_name='t', var_name='method')  # .dropna()

        sns.lineplot(data=df_long, x='n', hue='method', y='t', ci=95)  # errorbar=('ci', 95)), ci is deprecated

        plt.xscale('log', base=2)
        xt = sorted(df_long['n'].unique())
        plt.xticks(xt, labels=[str(x) for x in xt])
        plt.yscale('log')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f'Times of CI calculation for {stat}')
        plt.tight_layout()
        plt.savefig(f'images/comparison/times_line_{stat}_ylog10.png')
        plt.close()


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
          whis=[(100 - ci) / 2, 50 + ci / 2])
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


def kl(p, q):
    """K-L divergence."""
    part1 = np.where(p == 0, 0, p * np.log2(p / q))
    part2 = np.where(p == 1, 0, (1 - p) * np.log2((1 - p) / (1 - q)))
    return part1 + part2


def aggregate_results(result_folder, methods=None, combined_with='mean', withnans=True, onlybts=True, r=10000):
    # reading and filtering coverage table
    results = pd.read_csv(f'{result_folder}/results_from_intervals_{combined_with}{["", "_bts"][int(onlybts)]}'
                          f'{["", "_withnans"][int(withnans)]}.csv')

    # coverage = coverage[coverage['B'] == 1000]
    # coverage = coverage[~coverage['dgp'].isin(['DGPBernoulli_0.5', 'DGPBernoulli_0.95'])]
    #                       (coverage['statistic'].isin(['median', 'percentile_5', 'percentile_95'])))]
    if onlybts:
        if methods is None:
            methods = ['percentile', 'standard', 'basic', 'bc', 'bca', 'double', 'smoothed', 'studentized']
        results = results[results['method'].isin(methods)]

    # calculations for table of closeness to the best method
    results['difference'] = results['coverage'] - results['alpha']
    results['abs_difference'] = abs(results['difference'])
    min_distances = results[['alpha', 'coverage', 'abs_difference', 'dgp', 'statistic', 'n']] \
        .sort_values('abs_difference').groupby(['alpha', 'dgp', 'statistic', 'n']).first()
    results['min_difference'] = results.apply(
        lambda row: min_distances.loc[row['alpha'], row['dgp'], row['statistic'], row['n']]['abs_difference'], axis=1)
    results['best_coverage'] = results.apply(
        lambda row: min_distances.loc[row['alpha'], row['dgp'], row['statistic'], row['n']]['coverage'], axis=1)
    results['std'] = np.sqrt(results['best_coverage'] * (1 - results['best_coverage']) / results['repetitions'])
    results['near_best'] = abs(results['best_coverage'] - results['coverage']) < results['std']

    # calculation for KL-divergence
    results['kl_div'] = kl(results['coverage'], results['alpha'])

    # calculation for ranks
    results['rank'] = results[['alpha', 'abs_difference', 'dgp', 'statistic', 'n']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).rank()
    results['rank_kl'] = results[['alpha', 'kl_div', 'dgp', 'statistic', 'n']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).rank()

    def agregate_n_stat(df, column, fun, asc=True):
        agg = fun(df[['method', column]].groupby(['method']))

        agg_n = fun(results[['method', column, 'n']].groupby(['method', 'n'])).unstack()
        agg_n.columns = agg_n.columns.droplevel()
        agg = agg.join(agg_n)

        agg_stat = fun(results[['method', column, 'statistic']].groupby(['method', 'statistic'])).unstack()
        agg_stat.columns = agg_stat.columns.droplevel()
        agg = agg.join(agg_stat).sort_values(by=column, ascending=asc)

        return agg

    # tables
    # first comparison by near best in avg rank, distance, nans
    near_best = agregate_n_stat(results, 'near_best', lambda x: x.sum(), asc=False)
    avg_rank = agregate_n_stat(results, 'rank', lambda x: x.mean())

    dist_table = agregate_n_stat(results, 'avg_distance', lambda x: x.median())
    nans = agregate_n_stat(results, 'nans', lambda x: x.mean())

    nans_a = results[['method', 'nans', 'alpha']].groupby(['method', 'alpha']).mean().unstack()
    nans_a.columns = nans_a.columns.droplevel()
    nans = nans.join(nans_a)

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
    nans_all = results[results['method'] == 'ci_quant_nonparam'][['method', 'nans', 'statistic', 'repetitions',
                                                                  'n', 'alpha']].groupby(['method', 'statistic',
                                                                                          'n', 'alpha']).mean()
    nans_all = nans_all[nans_all['nans'] > 0]['nans'].sort_values(ascending=False)

    # t = results[results['method'].isin(['double', 'standard'])] for finding experiment for histogram

    # current comparison by kl div
    kl_div = agregate_n_stat(results, 'kl_div', lambda x: x.mean())
    kl_div_se = agregate_n_stat(results, 'kl_div', lambda x: x.sem())
    kl_div_med = agregate_n_stat(results, 'kl_div', lambda x: x.median())
    kl_div_rank = agregate_n_stat(results, 'rank_kl', lambda x: x.mean())
    kl_rank_se = agregate_n_stat(results, 'rank_kl', lambda x: x.sem())

    def significantly_worse(df, se_df, values=False, experiment_se=None):
        # subtracting method differences and standard errors, to know which are significantly worse (positive ones)
        idx_min = df.idxmin()
        df_significant = df - df.min() - se_df
        for col in df_significant.columns:
            # subtracting se of the best method
            df_significant[col] -= se_df.loc[idx_min[col], col]
        df_significant = df_significant.loc[df.index, :]

        # checking if each separate experiment has too big standard error
        if experiment_se == 'kl':
            # for kl estimation (computed by delta method, upper limit at alpha 0.025)
            # ERROR we are doing se_kl(kl(p), a), not se_kl(p, a), cant get p from kl
            # can we get
            def se_kl(p, a):
                fst_order = np.abs(np.log2(p * (1 - a)) - np.log2(a * (1 - p))) * (p * (1 - p) / r) ** 0.5
                return fst_order

            # additional_se = se_kl(df.min(), 0.025) + se_kl(df, 0.025)   ERROR
            additional_se = 0

        elif experiment_se == 'rank':
            # for rank estimation, conservative estimation by each method taking max variability
            max_sd = np.sqrt(r/(r-1) * ((df_significant.shape[0] - 1) / 2)**2)
            additional_se = max_sd / np.sqrt(r) * 2

        else:
            additional_se = 0

        df_sig_each_exp = df_significant - additional_se

        return df_sig_each_exp if values else (df_sig_each_exp >= 0)

    kl_div_significant = significantly_worse(kl_div, kl_div_se, experiment_se='kl')
    kl_rank_significant = significantly_worse(kl_div_rank, kl_rank_se, experiment_se='rank')

    return nans_all, kl_div, kl_div_med, kl_div_rank, kl_div_se, kl_rank_se, kl_div_significant, kl_rank_significant


def better_methods(method, result_folder, combined_with='mean', withnans=True, onlybts=True):
    """Finds better methods than proposed one - on experiment level."""
    # reading and filtering coverage table
    # results = pd.read_csv(f'{result_folder}/results_combined_{combined_with}.csv')
    results = pd.read_csv(f'{result_folder}/results_from_intervals_{combined_with}{["", "_bts"][int(onlybts)]}'
                          f'{["", "_withnans"][int(withnans)]}.csv')

    # calculations for table of (coverage) closeness to the best method
    results['difference'] = results['coverage'] - results['alpha']
    results['abs_difference'] = abs(results['difference'])
    min_distances = results[['alpha', 'coverage', 'abs_difference', 'dgp', 'statistic', 'n']] \
        .sort_values('abs_difference').groupby(['alpha', 'dgp', 'statistic', 'n']).first()
    results['min_difference'] = results.apply(
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
    results = results[results['method'] != method]  # filter out method we're comparing with
    better_cov = better_methods_by(results, 'better_cov')

    # DISTANCE FROM EXACT
    better_dist = better_methods_by(results, 'better_dist')

    return better_cov, better_dist


def better_cov_apply_fn(df, method):
    val = df[df['method'] == method]['abs_difference'].values[0]  # value for method we're interested in
    # abs_difference + std of another method is less then abs difference of proposed method
    df['better'] = (df['abs_difference'] + np.sqrt(df['coverage'] * (1 - df['coverage']) / df['repetitions'])) < val
    return df


def better_dist_apply_fn(df, method):
    val = df[df['method'] == method]['avg_distance'].values[0]  # value for method we're interested in
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


def average_distances_long(folder, combine_dist=np.mean):
    # for long tables (old results), skipping experiments with any nans
    # reading line by line because of 35GB dataframes
    dist_dict = defaultdict(list)  # dict that counts [sum of distances, #]
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
    avg_distances['norm_distance'] = avg_distances[['alpha', 'dgp', 'statistic', 'n', 'avg_distance']].groupby(
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
        keys = f.readline().strip('\n').split(',')  # header
        for line in f:
            line_dict = dict(zip(keys, line.strip('\n').split(',')))
            alpha, dgp, statistic, n, B, repetitions, true_val, exact = [line_dict[name] for name in
                                                                         ['alpha', 'dgp', 'statistic', 'n', 'B',
                                                                          'repetitions', 'true_value', 'exact']]
            alpha, n, B, repetitions, true_val, exact = float(alpha), int(n), int(B), int(repetitions), \
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
    results['min_distance'] = results['avg_distance']
    results['min_distance'] = results[['alpha', 'dgp', 'statistic', 'n', 'min_distance']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).transform(lambda x: x.min())
    results['norm_distance'] = results[['alpha', 'dgp', 'statistic', 'n', 'avg_distance']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).transform(lambda x: x / x.min())

    results.to_csv(f'{folder}/results_from_intervals_{combine_dist.__name__}_wthmin_{["", "_bts"][int(only_bts)]}'
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
    new = new[new['method'] != 'ci_corr_spearman']  # TODO delete

    all_results = pd.concat([old, new])
    all_results.to_csv(f'results/results_combined_{combine_dist}{["", "_bts"][int(only_bts)]}.csv', index=False)


def separate_experiment_plots(result_folder='results', B=1000, reps=10000, showoutliers=False):
    coverages = pd.read_csv(f'{result_folder}/coverage.csv')
    lengths = pd.read_csv(f'{result_folder}/length.csv')

    coverages = coverages[(coverages['B'] == B) & (coverages['repetitions'] == reps)]
    lengths = lengths[(lengths['B'] == B) & (lengths['repetitions'] == reps)]

    if reps < 10000:
        distances = pd.read_csv(f'{result_folder}/distance.csv')
        distances = distances[(distances['B'] == B) & (distances['repetitions'] == reps)]
    else:  # empty dataframe for final results (wasn't enough space, didn't save distances separately)
        distances = pd.read_csv(f'{result_folder}/intervals.csv')
        distances = distances[(distances['B'] == B) & (distances['repetitions'] == reps)]
        for m in ['percentile', 'basic', 'bca', 'bc', 'standard', 'smoothed', 'double', 'studentized', 'ttest',
                  'wilcoxon', 'ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett', 'chi_sq', 'ci_corr_pearson']:
            distances[m] = distances[m] - distances['exact']

    for sided in ['onesided', 'twosided']:
        i = 0
        if sided == 'onesided':
            df_whole = distances
        else:
            df_whole = lengths
            df_whole.rename(columns={'CI': 'alpha'}, inplace=True)

        for [dgp, stat, alpha], df in df_whole.groupby(['dgp', 'statistic', 'alpha']):
            # plotting coverage
            au = alpha if sided == 'onesided' else 0.5 + alpha / 2
            cov_df = coverages[(coverages['dgp'] == dgp) & (coverages['statistic'] == stat) &
                               (coverages['alpha'] == au)]
            if sided == 'twosided':
                cov_df2 = coverages[(coverages['dgp'] == dgp) & (coverages['statistic'] == stat) &
                                    (coverages['alpha'] == round(1 - au, 5))]
                # merge dataframes to be able to subtract coverages of same experiments
                cov_df = pd.merge(cov_df, cov_df2, on=['method', 'dgp', 'statistic', 'n', 'B', 'repetitions'],
                                  suffixes=('_au', '_al'))
                cov_df['coverage'] = cov_df['coverage_au'] - cov_df['coverage_al']
                cov_df['alpha'] = cov_df['alpha_au'] - cov_df['alpha_al']

            nm = cov_df['method'].nunique()
            if nm > 10:
                cols = plt.cm.tab20(np.linspace(0.05, 0.95, cov_df['method'].nunique()))
            else:
                cols = plt.cm.tab10(np.linspace(0.05, 0.95, cov_df['method'].nunique()))
            colors = {m: c for (m, c) in zip(cov_df['method'].unique(), cols)}
            order = cov_df['method'].unique()

            val_name = 'distance' if sided == 'onesided' else 'length'

            fig = plt.figure(figsize=(6, 12), constrained_layout=True)
            txt = fig.suptitle(f"Coverage and {val_name} for {dgp}, {stat}, {alpha}", fontsize=14)

            plt.subplot(2, 1, 1)
            plot_coverage_bars(cov_df, colors=colors, ci=95, scale='linear', set_ylim=False, order=order, hue='method',
                               x='n')
            plt.title('Coverages')

            # long dataframe for distances
            cols_here = df.dropna(axis=1, how='all').columns.tolist()
            all_methods = ["percentile", "basic", "bca", "bc", "standard", "smoothed", "double", "studentized", "ttest",
                           "wilcoxon", "ci_quant_param", "ci_quant_nonparam", "maritz-jarrett", "chi_sq",
                           "ci_corr_pearson"]
            methods_here = [m for m in all_methods if m in cols_here]
            dist_df_long = pd.melt(df, id_vars=["alpha", "dgp", "n"], value_vars=methods_here, var_name="method",
                                   value_name=val_name)

            # plotting distances
            plt.subplot(2, 1, 2)
            plt.title('Distances' if sided == 'onesided' else 'Lengths')

            sns.boxplot(x="n", y=val_name, hue="method", data=dist_df_long, hue_order=order, palette=colors,
                        showfliers=showoutliers)
            plt.axhline(y=0, color="gray", linestyle="--")
            plt.yticks(list(plt.yticks()[0]) + [0])
            plt.legend([], [], frameon=False)
            plt.yscale(['symlog', 'log'][int(sided == 'twosided')])
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2f}'))
            # plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            handles, labels = plt.gca().get_legend_handles_labels()
            lgd = fig.legend(handles, labels, loc='center left', title="Method", bbox_to_anchor=(1, 0.5))

            plt.savefig(f'images/separate_experiments_{sided}{["", "_outliers"][int(showoutliers)]}/'
                        f'plots_{sided}_experiment_{dgp}_{stat}_{alpha}.png',
                        bbox_extra_artists=(lgd, txt), bbox_inches='tight')


def hierarchical_from_intervals(folder='results_hierarchical', bts_method='double', n_lvl=3, filenames=['intervals']):
    # for wide tables (new results), skipping just replications that have nans
    # we are reading results line by line because of possibility of very large dataframes that can't fit in memory

    def analyse_line(line_str, line_keys, method):
        line_dict = dict(zip(line_keys, line_str.strip('\n').split(',')))

        if line_dict['method'] == 'exact':
            # we don't need exact intervals in this study
            return False

        sampling, strategy, m = line_dict['method'].split('_')
        if int(line_dict['B']) != 1000 or m != method or len(strategy) != n_lvl or line_dict['statistic'] != 'mean':
            # skip the line, we are building each method's table separately, only for mean statistic
            return False

        line_dict['strategy'] = strategy
        line_dict['method'] = method

        return line_dict

    results = []

    for filename in filenames:
        with open(f'{folder}/{filename}.csv') as f:
            keys = f.readline().strip('\n').split(',')  # header
            for line in f:
                line_results = analyse_line(line, keys, bts_method)
                if not line_results:
                    # line_results == False
                    continue
                results.append(line_results)
                # df = pd.concat([df, pd.Series(line_results).to_frame().T], ignore_index=True)

    df = pd.DataFrame(results)
    df.to_csv(f'{folder}/hierarchical_nlvl{n_lvl}_{bts_method}.csv', index=False)


def separate_experiment_plots_hierarchical(result_folder='results_hierarchical', B=1000, reps=1000, std=1,
                                           method='double'):
    coverages = pd.read_csv(f'{result_folder}/coverage.csv')
    repetitions = {}
    nlvls = [2, 3, 4]
    for nlvl in nlvls:
        rdf = pd.read_csv(f'{result_folder}/hierarchical_nlvl{nlvl}_{method}.csv')
        repetitions[nlvl] = rdf[(rdf['B'] == B) & (rdf['repetitions'] == reps)]

    coverages = coverages[(coverages['B'] == B) & (coverages['repetitions'] == reps) & (coverages['std'] == std)
                          & (coverages['method'].str.split('_').str[2] == method) & (coverages['statistic'] == 'mean')]
    coverages['strategy'] = coverages['method'].str.split('_').str[1]
    coverages_a95 = coverages[(coverages['alpha'] == 0.95)]     # hack for variance coverage (always checking 95CI)

    for alpha in coverages['alpha'].unique():

        cov_df = coverages[(coverages['alpha'] == alpha)]

        fig = plt.figure(figsize=(10, 14), constrained_layout=True)
        # txt = fig.suptitle(f"Coverage and variance coverage for std {std}, {method}", fontsize=14)
        lgds = []

        for i, nlvl in enumerate(nlvls):
            cov_lvl_df = cov_df[cov_df['levels'] == nlvl]
            cov_lvl_95 = coverages_a95[coverages_a95['levels'] == nlvl]

            nm = cov_lvl_df['strategy'].nunique()
            if nm > 10:
                cols = plt.cm.tab20(np.linspace(0.05, 0.95, cov_lvl_df['strategy'].nunique()))
            else:
                cols = plt.cm.tab10(np.linspace(0.05, 0.95, cov_lvl_df['strategy'].nunique()))
            colors = {m: c for (m, c) in zip(cov_lvl_df['strategy'].unique(), cols)}
            order = cov_lvl_df['strategy'].unique()

            plt.subplot(3, 2, 2 * i + 1)
            plot_coverage_bars(cov_lvl_df, colors=colors, ci=95, scale='linear', set_ylim=False, order=order,
                               hue='strategy', x='n')
            if i == 0:
                plt.title('Accuracy')
            if i != 2:
                plt.xlabel('')

            plt.xticks(plt.xticks()[0], plt.xticks()[1], rotation=45)
            plt.ylim(max(plt.ylim()[0], 0), min(plt.ylim()[1], 1))

            # plotting boxplots of variance estimations
            # plt.subplot(3, 3, 4 + i)
            # # plt.title('Variance estimation')
            # rep_df = repetitions[nlvl]
            # rep_df = rep_df[rep_df['alpha'] == alpha]
            # sns.boxplot(x="n", y='mean_var', hue="strategy", data=rep_df, hue_order=order, palette=colors,
            #             showfliers=True)
            # plt.axhline(y=rep_df['gt_variance'].values[0], color="gray", linestyle="--")
            # # plt.yticks(list(plt.yticks()[0]) + [0])
            # plt.legend([], [], frameon=False)

            # # just check
            # if rep_df['gt_variance'].mean() != rep_df['gt_variance'].values[0]:
            #     print('variances not equal')

            # plotting distances
            ax = plt.subplot(3, 2, 2 * i + 2)
            plot_coverage_bars(cov_lvl_95, colors=colors, ci=95, scale='linear', set_ylim=False, order=order,
                               hue='strategy', x='n', cov_kind='var_coverage')
            handles, labels = ax.get_legend_handles_labels()
            lgds.append(fig.legend(handles, labels, loc='center left', title="Strategy", bbox_to_anchor=(1, (5-2*i)/6)))
            if i == 0:
                plt.title("Imitation of DGP's Variation Properties")
            if i != 2:
                plt.xlabel('')

            plt.xticks(plt.xticks()[0], plt.xticks()[1], rotation=45)
            plt.ylim(max(plt.ylim()[0], 0), min(plt.ylim()[1], 1))

        # handles, labels = plt.gca().get_legend_handles_labels()
        # lgd1 = fig.legend(handles, labels, loc='center left', title="Strategy", bbox_to_anchor=(1, 0.16))
        # lgd2 = fig.legend(handles, labels, loc='center left', title="Strategy", bbox_to_anchor=(1, 0.5))
        # lgd3 = fig.legend(handles, labels, loc='center left', title="Strategy", bbox_to_anchor=(1, 0.84))
        plt.savefig(f'images_hierarchical/separate_experiments/'
                    f'plots_experiment_{alpha}_{std}_{method}.png',
                    bbox_extra_artists=lgds, bbox_inches='tight')

            # plt.yticks(list(plt.yticks()[0]) + [0])
            # plt.legend([], [], frameon=False)
            # plt.yscale(['symlog', 'log'][int(sided == 'twosided')])
            # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2f}'))
            # # plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            #
            # handles, labels = plt.gca().get_legend_handles_labels()
            # lgd = fig.legend(handles, labels, loc='center left', title="Method", bbox_to_anchor=(1, 0.5))

            # plt.savefig(f'images/separate_experiments_{sided}{["", "_outliers"][int(showoutliers)]}/'
            #             f'plots_{sided}_experiment_{dgp}_{stat}_{alpha}.png',
            #             bbox_extra_artists=(lgd, txt), bbox_inches='tight')


if __name__ == '__main__':
    # folder_add = '_hierarchical'
    # folder_add = '_10000_reps'
    folder_add = '_final'
    subfolder = ''
    # additional = 'hierarchical'
    additional = ''
    # cov = pd.read_csv(f'results{folder_add}/coverage.csv')
    bts_methods = ['percentile', 'standard', 'basic', 'bc', 'bca', 'double', 'smoothed']

    # main_plot_comparison(filter_by={}, additional=additional, scale='linear', folder_add=folder_add, set_ylim=True)

    # for stat in [np.mean, np.median]:
    #     for only_bts in [False]:
    #         for include_nans in [True]:
    #             results_from_intervals('results', combine_dist=stat, only_bts=only_bts,
    #                                    include_nan_repetitions=include_nans)
    # combine_results(stat.__name__, only_bts=only_bts) not needed anymore with complete wide results

    # tables = {}
    # onlybts = True
    #
    # for stat in ['mean', 'median']:
    #     print('Aggregated with ', stat)
    #
    #     for table, name in zip(aggregate_results(f'results{folder_add}', combined_with=stat, withnans=True,
    #                                              onlybts=onlybts),
    #                            ['nans', 'kl div', 'kl div med', 'kl div rank',
    #                            'kl div se', 'kl rank se', 'kl_div_significant', 'kl_rank_significant']):
    #         print(name)
    #         if 'rank' in name:
    #             ff = "%.2f"
    #         else:
    #             ff = "%.4f"
    #         print(table.to_latex(float_format=ff))
    #         tables[name] = table


    # print('BETTER:')
    # cov_bet, dist_bet = better_methods('double', f'results{folder_add}', combined_with=stat, withnans=True,
    #                                    onlybts=True)
    # print(cov_bet.to_latex(float_format="%.2f"))

    # better_methods_repetition_level('double', 'results_wide_nans')

    plot_times_line()

    # separate_experiment_plots('results', showoutliers=False)

    # hierarchical results separation
    # for levels in [2, 3, 4]:
    #     for method in ['double', 'percentile', 'bca']:
    #         print(method, levels)
    #         hierarchical_from_intervals(folder='results_hierarchical', bts_method=method, n_lvl=levels,
    #                                     filenames=['intervals_first550experiments', 'intervals'])

    # plots for hierarchical experiments
    # separate_experiment_plots_hierarchical()

