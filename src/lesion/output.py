"""Presentation and visualization of lesson test results."""

import math
import itertools as it
from copy import deepcopy
import warnings
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import networkx as nx

from src.utils import compute_pvalue, heatmap_fixed
from src.lesion import perform_lesion_experiment


def layer_cluster_taxonomify(stats_row, with_proportion=True,
                             pvalue_threshod=1/100,
                             diff_threshold=-1/100,
                             diff_field='diff'):
    """Choose a category for a damaged layer-cluster based on:
    1. `label_in_layer_proportion` - The proportion of layer-cluster neurons
                                     in the layer.
    2. `n_layer_label` - The number of neurones in the layer-cluster.
    3. `corrected_pvalue`
    4. `diff` - The difference in overall accuracy
                between the damaged network to the undamaged one.

    See the code for the complete taxonomify

    Parameters:
    -----------
    stats_row :
        A DataFrame's row of dictionary with the statistics of the lesion test
        for a given layer-cluster.

    pvalue_threshod : float
        Define what is a small difference in a pvalue.
    
    diff_threshold : float
        Define what is a small difference in diff.
    """

    if with_proportion and stats_row['label_in_layer_proportion'] == 1:
        return 'complete'
    elif with_proportion and stats_row['n_layer_label'] <= 12:  # ~5% out of 256 neurons
        return 'small'
    elif stats_row['corrected_pvalue'] < pvalue_threshod and stats_row[diff_field] <= diff_threshold:
        return 'important'
    # elif stats_row['corrected_pvalue'] > (1-pvalue_threshod) and stats_row[diff_field] > diff_threshold:
    elif stats_row['corrected_pvalue'] == 1 and stats_row[diff_field] > diff_threshold:
        return 'prunable'
    # elif stats_row['corrected_pvalue'] > (1-pvalue_threshod) and stats_row[diff_field] < diff_threshold:
    elif stats_row['corrected_pvalue'] == 1 and stats_row[diff_field] <= diff_threshold:
        return 'least_important'
    elif stats_row['corrected_pvalue'] < pvalue_threshod and stats_row[diff_field] >= diff_threshold:
        return 'sig-but-not-diff'
    elif stats_row['corrected_pvalue'] >= pvalue_threshod and stats_row[diff_field] < diff_threshold:
        return 'diff-but-not-sig'

    else:
        return 'other'


def compute_damaged_cluster_stats(true_results, all_random_results, metadata, evaluation,
                                  pvalue_threshod=None,
                                  diff_threshold=-1/100,
                                  double_joint_df=None,
                                  single_df=None,
                                  diff_field='diff'):
    
    n_way = 2 if 'labels_in_layers' in true_results[0] else 1
    index = ['labels_in_layers'] if n_way == 2 else ['layer', 'label']

    assert diff_field in ('diff', 's_i|j')
    if diff_field == 's_i|j':
        assert n_way == 2
        assert single_df is not None

    if pvalue_threshod is None:
        pvalue_threshod = 1 / len(all_random_results)

    true_df = (pd.DataFrame(true_results)
              .set_index(index)
              .sort_index())

    all_random_df = pd.DataFrame(sum(all_random_results, []))

    all_random_layer_label = all_random_df.groupby(index)


    true_df = true_df['acc_overall']

    all_random_layer_label = all_random_layer_label['acc_overall']

    corrected_pvalue_by_groups = ((layer_label, compute_pvalue(true_df[layer_label], group))
                                           for layer_label, group in all_random_layer_label)

    layer_label_index, corrected_pvalue_by_layer_label = zip(*corrected_pvalue_by_groups)

    # make sure that the order of true_df and corrected_pvalue_by_layer_label is the same
    # before setting corrected_pvalues_df's index as same as true_df
    assert tuple(true_df.index) == layer_label_index
    
    
    corrected_pvalues_df = pd.DataFrame({'acc_overall': corrected_pvalue_by_layer_label},
                                        index=true_df.index)

    corrected_pvalues_df = (corrected_pvalues_df
                            .assign(value='corrected_pvalue')
                            .set_index(['value'], append=True))

    true_df = pd.DataFrame(true_df).assign(value='true').set_index(['value'], append=True)

    random_stats_df = pd.DataFrame({'acc_overall': all_random_layer_label
                       .agg(['mean', 'std'])
                       .stack()})

    z_score_df = ((true_df - random_stats_df.xs('mean', level=-1))
                  / random_stats_df.xs('std', level=-1))

    z_score_df.index = z_score_df.index.set_levels(['z_score'], level=-1)

    diff_df = (true_df - evaluation['acc_overall'])

    diff_df.index = diff_df.index.set_levels(['diff'], level=-1)

    stats_df = (pd.concat([true_df, corrected_pvalues_df,
                           # pvalues_df,
                           diff_df, z_score_df, random_stats_df])
                .sort_index())

    metadata_df = (pd.DataFrame(metadata)
                       .set_index(['layer', 'label']))

    overall_stats_df = stats_df.unstack()

    overall_stats_df.columns = overall_stats_df.columns.droplevel()

    if n_way == 1:
        overall_stats_df = pd.concat([overall_stats_df, metadata_df], axis=1)

    if diff_field == 's_i|j':
        overall_stats_df = enrich_score_double_conditional_df(overall_stats_df, single_df)
        
    overall_stats_df['taxonomy'] = overall_stats_df.apply(lambda r:
                                                          layer_cluster_taxonomify(r,
                                                                                   with_proportion=(n_way==1),
                                                                                   pvalue_threshod=pvalue_threshod,
                                                                                   diff_threshold=diff_threshold,
                                                                                   diff_field=diff_field),
                                                          axis=1)

    overall_columns = ['diff', 'corrected_pvalue']
    
    if n_way == 1: overall_columns.append('label_in_layer_proportion')
    overall_columns.append('true')
    
    overall_columns.extend(['taxonomy', 'mean', 'std', 'z_score'])
    
    if n_way == 1: overall_columns.append('n_layer_label')

    overall_stats_df = overall_stats_df[overall_columns]

    
    # adding the diagonal (i.e., single) to a conditional double using the joint double
    if double_joint_df is not None:
        assert n_way == 2, '`double_joint_df` should be given only for double'
        warnings.warn('Make sure that `n_shuffled` for conditional double results should be the same'
                      ' as the one for generating the joint double df!')
        
        double_same_pair_mask = [first == second for first, second in double_joint_df.index]
        double_same_pair_df = double_joint_df[double_same_pair_mask]
        overall_stats_df = pd.concat([overall_stats_df, double_same_pair_df]).sort_index()
    
    return overall_stats_df


def plot_damaged_cluster(index, true_results, all_random_results, metadata, evaluation,
                         figsize=(10, 5), alpha=0.1, ax=None):
    
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)

    layer_label_metadata = metadata[index]

    (pd.DataFrame([results[index] for results in all_random_results])
     .set_index(['label', 'layer'])
     .T
     .plot(color='k', alpha=alpha,
           legend=False, ax=ax))

    (pd.DataFrame(true_results)
      .set_index(['layer', 'label'])
      .iloc[index]
      .plot(color='r',
            legend=False, ax=ax))

    # this is hacky and annoying, but it accomplishes what it needs to to get the right ordering
    evaluation_copy = deepcopy(evaluation)
    del evaluation_copy['acc_overall']
    evaluation_copy['acc_overall'] = evaluation['acc_overall']
    (pd.Series(evaluation_copy).plot(color='g', legend=False, ax=ax))

    ax.set_xticks(range(11))
    ax.set_xticklabels(list('0123456789') + ['all'])

    layer_label_metadata = deepcopy(layer_label_metadata)
    layer_label_metadata['label_in_layer_precentage'] = (layer_label_metadata['label_in_layer_proportion']
                                                         * 100)
    ax.title.set_text('Layer {layer} Cluster {label}: {label_in_layer_precentage:.2f}%'
              .format(**layer_label_metadata))
    
    ax.set_ylim(0, 1)

    return ax


def plot_damaged_cluster_regression(index, true_results, all_random_results, metadata, evaluation,
                                    figsize=(10, 5), alpha=0.1, ax=None):
    n_inputs = 2
    coefs = (0, 1)
    exps = (0, 1, 2)
    n_terms = len(exps) ** n_inputs
    n_outputs = len(coefs) ** n_terms
    poly_coefs = np.zeros((n_outputs, n_terms))
    for poly_i, coef_list in enumerate(itertools.product(coefs, repeat=n_terms)):
        poly_coefs[poly_i] = np.array(coef_list)
    term_exps = [exs for exs in itertools.product(exps, repeat=n_inputs)]
    n_terms = len(term_exps)

    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)

    layer_label_metadata = metadata[index]
    n_shuffles = len(metadata)

    random_mses = np.array([result[index]['mses'] for result in all_random_results])
    true_mses = true_results[index]['mses']
    eval_mses = evaluation['mses']

    term_randoms = np.array(
        [np.mean(
            np.array(
                [random_mses[:, output_i] for output_i in range(n_outputs)
                 if poly_coefs[output_i][term_i] == 1]
            ), axis=0
        )
                             for term_i in range(n_terms)]
    )

    term_trues = np.array(
        [np.mean(
            np.array(
                [true_mses[output_i] for output_i in range(n_outputs)
                 if poly_coefs[output_i][term_i] == 1]
            )
        )
                           for term_i in range(n_terms)]
    )

    evals = np.array(
        [np.mean(np.array([eval_mses[output_i] for output_i in range(n_outputs)
                           if poly_coefs[output_i][term_i] == 1]))
         for term_i in range(n_terms)]
    )

    pd.DataFrame(term_randoms).plot(color='k', alpha=alpha, legend=False, ax=ax)
    pd.DataFrame(np.reshape(term_trues, (-1, 1))).plot(color='r', legend=False, ax=ax)
    pd.DataFrame(np.reshape(evals, (-1, 1))).plot(color='g', legend=False, ax=ax)

    ax.set_xticks(range(n_terms))
    ax.set_xticklabels([f'{te[0]},{te[1]}' for te in term_exps])

    lyr = metadata[index]['layer']
    lbl = metadata[index]['label']
    pct = round(metadata[index]['label_in_layer_proportion'] * 100, 2)

    ax.title.set_text(f'Layer {lyr} Cluster {lbl}: {pct}%')

    return ax


def plot_all_damaged_clusters(true_results, random_results, metadata, evaluation,
                              alpha=0.1, ncols=5, title=None):

    if 'output_props' in true_results[0].keys():
        for sm_i in range(len(true_results)):
            del true_results[sm_i]['output_props']
        for rand_i in range(len(random_results)):
            for sm_i in range(len(random_results[rand_i])):
                del random_results[rand_i][sm_i]['output_props']
        del evaluation['output_props']

    nrows = int(math.ceil(len(metadata) / ncols))
    figsize = (30, 4 * nrows)
    assert len(metadata) <= nrows * ncols

    if 'mses' in true_results[0]:
        plot_fn = plot_damaged_cluster_regression
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=figsize)
    else:
        plot_fn = plot_damaged_cluster
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize)

    if title is not None:
        fig.suptitle(title)

    axes_iter = it.chain(*axes)
    for cluster_index, ax in zip(range(len(metadata)), axes_iter):
        plot_fn(cluster_index, true_results, random_results, metadata, evaluation, alpha=alpha, ax=ax)

    for ax in axes_iter:
        ax.axis('off')
                         
    return axes


def plot_overall_damaged_clusters(true_results, all_random_results, metadata, evaluation,
                                  title=None, figsize=(11, 6), y_lim=(0, 1)):

    layer_cluster_random_results = zip(*all_random_results)
    
    df = pd.DataFrame({'{layer}-{label}\n({label_in_layer_proportion:.2f})'.format(**layer_cluster_metadata):
                       [result['acc_overall'] for result in layet_cluster_random_result]
                      for layer_cluster_metadata, layet_cluster_random_result
                      in zip(metadata, layer_cluster_random_results)})

    f, ax = plt.subplots(figsize=figsize)

    # random damaged
    sns.swarmplot(data=df)

    # damaged by spectral clustering
    ax.plot([result['acc_overall'] for result in true_results],
            color='k', linestyle='None', marker='X', markersize=10,
            zorder=20,
           label='Sub-module with lesion')

    # non-damaged model
    ax.axhline(evaluation['acc_overall'],
                color='k', linewidth=2, linestyle='--',
               zorder=10,
               label='Model without lesion')

    ax.set_ylim(*y_lim)

    minor_locator = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor')

    plt.xlabel('Sub-module (proportion in layer)')
    plt.ylabel('Accuracy')
    plt.xticks(rotation='vertical')

    plt.legend()
    
    if title is not None:
        ax.set_title(title)

    return ax


def plot_accuracy_profile(true_results, single_df,
                          figsize=(7, 5), title=None, ax=None):
    
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)
    
    taxonomy_translator = _build_taxonomy_translator(single_df)

    accuracy_profile_df = pd.DataFrame(true_results)

    accuracy_profile_df['taxonomy'] = (accuracy_profile_df
                                       .apply(lambda r: taxonomy_translator[(r['layer'], r['label'])],
                                       axis=1))


    accuracy_profile_df['layer-label'] = (accuracy_profile_df
                                          .apply(lambda r:
                                                 ('{layer:0.0f}-{label:0.0f} ({taxonomy})'
                                                  .format(**r)),
                                          axis=1))

    accuracy_profile_df = accuracy_profile_df.drop(['label', 'layer', 'taxonomy'], axis=1)
    accuracy_profile_df = accuracy_profile_df.set_index('layer-label').T

    accuracy_profile_df.plot(ax=ax)

    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5),
              fancybox=True, shadow=True, ncol=5)


    if title is not None:
        ax.set_title(title)

    plt.xticks(range(len(accuracy_profile_df)),
               list(accuracy_profile_df.index),
               rotation=45)
    
    return ax


def report_lesion_test(name, dataset_path, run_dir,
                       n_clusters=4, n_shuffles=100, n_way=1,
                       verbose=False,
                       with_random_lines_plot=False, with_overall_plot=True,
                       with_accuracy_profile=True, with_display_df=False, true_as_random=False,
                       to_save=False):

    assert not to_save, 'This function currently does not support Excel/CSV saving of results.'
    assert n_way == 1, 'Currently, Only single lesion test is supported.'

    (true_results,
     all_random_results,
     metadata,
     evaluation) = perform_lesion_experiment(dataset_path, run_dir,
                                                   n_clusters=n_clusters,
                                                   n_shuffles=n_shuffles,
                                                   n_way=n_way,
                                                   true_as_random=true_as_random,
                                                   verbose=verbose)
    
    if n_way == 1:
        
        if with_random_lines_plot:
            plot_all_damaged_clusters(true_results, all_random_results, metadata, evaluation,
                                      title=f'{name}');
        if with_overall_plot:
            plot_overall_damaged_clusters(true_results, all_random_results, metadata, evaluation,
                                          title=f'{name}');
        df = compute_damaged_cluster_stats(true_results, all_random_results, metadata, evaluation)

        if with_accuracy_profile:
            plot_accuracy_profile(true_results, df,
                                  title=f'{name}');
            
        if with_display_df:
            print(f'### {name}')
            display.display(df)

        return df


####################################
#######    Double Lesion     #######
####################################

def _build_taxonomy_translator(single_df):
    return single_df['taxonomy'].str.upper().str[:3].to_dict()


def build_double_mat(double_joint_df, single_df, col='true'):
    
    assert col in ('true', 'diff', 'corrected_pvalue', 's_i|j')
    
    taxonomy_translator = _build_taxonomy_translator(single_df)
    
    def _build_label_in_layer_tick(label_in_layer):
        label, layer = label_in_layer
        taxonomy_letter = taxonomy_translator[label_in_layer]
        return f'{label}-{layer} ({taxonomy_letter})'

    
    df = double_joint_df.reset_index()

    df[['first', 'second']] = (df['labels_in_layers']
                                               .apply((lambda x: pd.Series([x[0], x[1]]))))

    df['first'] = df['first'].apply(_build_label_in_layer_tick)
    df['second'] = df['second'].apply(_build_label_in_layer_tick)

    df = df.pivot('first', 'second', col)

    if col in ('true', 'diff', 's_i|j'):
        df *= 100

    return df


def build_double_joint_interaction_mat(double_joint_diff_mat, single_df):
    diag = np.diag(double_joint_diff_mat)
    diff_sum = diag + diag[:, None]
    df = (double_joint_diff_mat - diff_sum)
    np.fill_diagonal(df.values, single_df['true'] * 100)

    return df


def build_conditional_double_df(double_df, single_df):
    # the conditional df is not symmetric,
    # so we need to construct a full matrix
    df = double_df.copy()
    df[df.isnull()] = df.T

    diag = np.diag(df)
    # substruction is per row (first),
    # so the diagonal is fixed per column (second), and thus conditioned
    df = diag - df
    np.fill_diagonal(df.values, single_df['true'] * 100)
    
    return df


def plot_double_heatmap(mat, df=None, pvalue_threshod=1/50,
                        with_diag=True, is_trig=True, with_text_format=True,
                        metadata=None, with_masking=False,
                        figsize=(20, 15), diag_font_size=10, fmt='.2g',
                        vmin=0, vmax=100, ax=None):

    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)

    df = df.copy()
    
    #df
    #if with_masking:
    ax = heatmap_fixed(mat,
                     vmin=vmin, vmax=vmax,
                     annot=True,
                     linewidths=.5,
                     fmt=fmt)

    if with_diag:
        if is_trig:
            diagonal_diffs = np.arange(len(metadata) + 1)[::-1]
            diagonal_indices = np.zeros(len(metadata), dtype=int)
            diagonal_indices[1:] = np.cumsum(diagonal_diffs)[:-2]
        else:
            diagonal_indices = np.arange(0, len(metadata)**2,
                                         len(metadata) + 1)

        for text in np.take(ax.texts, diagonal_indices):
            text.set_size(diag_font_size)
            text.set_weight('bold')
            # text.set_style('italic')

    if not with_masking and with_text_format:
        for text, (_, row) in zip(ax.texts, df.iterrows()):
            if row['corrected_pvalue'] <= pvalue_threshod:
                text.set_text(text.get_text() + ' ★')
            elif row['corrected_pvalue'] == 1:
                text.set_text(text.get_text() + ' ?')

    return ax


def build_double_pvalue_mat(double_df, single_df, categorized=True,
                            pvalue_threshod=1/50):

    mat = build_double_mat(double_df, single_df, col='corrected_pvalue')
    
    if categorized:
        one_mask = (mat == 1)
        sig_mask = (mat <= pvalue_threshod)
        assert not np.all(one_mask & sig_mask)
        else_mask = ~(one_mask | sig_mask)

        mat = mat.astype(str)
        mat[one_mask] = 'Anti-Significant'
        mat[sig_mask] = 'Significant'
        mat[else_mask] = 'Non-Significant'
        np.fill_diagonal(mat.values, 'Single')

        return mat


def plot_double_pvalue_mat(mat, pvalue_threshod=1/50, value_to_color='pvalue',
                           figsize=(13, 10), ax=None):

    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)

    # https://stackoverflow.com/questions/36227475/heatmap-like-plot-but-for-categorical-variables-in-seaborn
    
    if value_to_color == 'pvalue':
        value_to_color = {'Non-Significant': 'white',
                  'Significant': 'black',
                  'Anti-Significant': 'grey',
                  'Single':'grey green'}
    elif value_to_color == 'bool':
        value_to_color = {'False': 'white',
                          'True': 'black',
                          'Single': 'grey green'}
        
    if value_to_color is not None:
        values, colors = zip(*value_to_color.items())
        value_to_int = {value: i for i, value in enumerate(values)}
        n = len(value_to_color)
        cmap = sns.xkcd_palette(colors)
        
    else:
        value_to_int = {j:i for i,j in enumerate(pd.unique(mat.values.ravel()))} # like you did
        n = len(value_to_int)     

        # discrete colormap (n samples from a given cmap)
        cmap = sns.color_palette('hls', n) 

    # ax = sns.heatmap(mat.replace(value_to_int), cmap=cmap) 
    ax = heatmap_fixed(mat.replace(value_to_int), cmap=cmap) 

    # modify colorbar:
    colorbar = ax.collections[0].colorbar 
    r = colorbar.vmax - colorbar.vmin 
    colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
    colorbar.set_ticklabels(list(value_to_int.keys()))                                          

    # ax.set_xlim(0, len(mat))
    # ax.set_ylim(0, len(mat))
    plt.yticks(rotation=0) 

    return ax


def plot_cluster_scatter(df, figsize=(10, 5), x='label_in_layer_proportion', y='diff',
                         ax=None, **kwargs):
    
    if not kwargs:
        kwargs = {'s': 200}

    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)

    df_flat = df.reset_index()

    df_flat['label_in_layer_proportion'] *= 100
    df_flat['diff'] *= 100

    hue_order = [cat for cat in ['important',
                               'sig-but-not-diff',
                               'diff-but-not-sig',
                               'small',
                               'least_important',
                               'complete',
                               'other']
                 if cat in set(df_flat['taxonomy'].values)]

    palette = sns.color_palette('cubehelix', len(hue_order))
                                
    g = sns.scatterplot(x,
                    y,
                    hue='taxonomy',
                    hue_order=hue_order,
                    palette=palette,
                    data=df_flat,
                    ax=ax,
                    **kwargs)

    # refactor! as parameter
    for lh in g.legend_.legendHandles: 
        lh._sizes = [150] 
    
    if x == 'label_in_layer_proportion':
        ax.set_xlabel('Proportion in Layer (%)')
    if y == 'diff':
        ax.set_ylabel('Difference (%)')
    elif y == 'z_score':
        ax.set_ylabel('Z Score')
    
    df_flat['text'] = df_flat.apply(lambda r: str(r['layer']) + '-' +  str(r['label']),
                                    axis=1)
    for line in range(0, df.shape[0]):
        if df_flat['text'][line] not in df_flat[df_flat['taxonomy'] == 'important']['text'].values:
            continue
        ax.text(df_flat[x][line] + 0.01, df_flat[y][line],
                df_flat['text'][line],
                horizontalalignment='left', size='small', color='black', weight='semibold')
 

    return ax


def build_double_joint_imp_grouped_df(double_joint_df, single_df):
    tw_joint_imp_df = double_joint_df.reset_index()


    tw_joint_imp_df[['first', 'second']] = (tw_joint_imp_df['labels_in_layers']
                                               .apply((lambda x: pd.Series([x[0], x[1]]))))

    tw_joint_imp_df['first_taxonomy'] = tw_joint_imp_df['first'].apply(lambda index: single_df.loc[index, 'taxonomy'])
    tw_joint_imp_df['second_taxonomy'] = tw_joint_imp_df['second'].apply(lambda index: single_df.loc[index, 'taxonomy'])

    tw_joint_imp_df['is_important'] = (tw_joint_imp_df['taxonomy'] == 'important')
    tw_joint_imp_df['first_is_important'] = (tw_joint_imp_df['first_taxonomy'] == 'important')
    tw_joint_imp_df['second_is_important'] = (tw_joint_imp_df['second_taxonomy'] == 'important')


    is_important_fields = ['first_is_important', 'second_is_important', 'is_important']

    tw_joint_imp_df[is_important_fields] = tw_joint_imp_df[is_important_fields].replace({True: '✔️', False: '❌'})

    tw_joint_imp_grouped_df = (tw_joint_imp_df
     .groupby(is_important_fields)
     .size()
     .sort_values(ascending=False))

    return tw_joint_imp_df, tw_joint_imp_grouped_df


def build_tw_cond_imp_merged_df(double_conditional_df, single_df):
    
    tw_cond_imp_df = double_conditional_df.reset_index()


    tw_cond_imp_df[['first', 'second']] = (tw_cond_imp_df['labels_in_layers']
                                               .apply((lambda x: pd.Series([x[0], x[1]]))))

    tw_cond_imp_df['first_taxonomy'] = tw_cond_imp_df['first'].apply(lambda index: single_df.loc[index, 'taxonomy'])
    tw_cond_imp_df['second_taxonomy'] = tw_cond_imp_df['second'].apply(lambda index: single_df.loc[index, 'taxonomy'])

    tw_cond_imp_df['is_important'] = (tw_cond_imp_df['taxonomy'] == 'important')
    tw_cond_imp_df['first_is_important'] = (tw_cond_imp_df['first_taxonomy'] == 'important')
    tw_cond_imp_df['second_is_important'] = (tw_cond_imp_df['second_taxonomy'] == 'important')

    # Remove single lesion test
    tw_cond_imp_df = tw_cond_imp_df[tw_cond_imp_df['first'] != tw_cond_imp_df['second']]

    # Remove pairs of layer-cluster from the same layer
    # because it is not clear how we should interpret the p-value
    tw_cond_imp_df = tw_cond_imp_df[tw_cond_imp_df['first'].apply(lambda x: x[0])
                                    != tw_cond_imp_df['second'].apply(lambda x: x[0])]

    # Take rows that both of the layer-clusters ar important
    tw_cond_imp_df = tw_cond_imp_df[((tw_cond_imp_df['first_taxonomy'] == 'important')
                                    & (tw_cond_imp_df['second_taxonomy'] == 'important'))]

    tw_cond_imp_df['sorted'] = tw_cond_imp_df['labels_in_layers'].apply(sorted).apply(tuple)

    tw_cond_imp_merged_df = pd.merge(tw_cond_imp_df[tw_cond_imp_df['first'] < tw_cond_imp_df['second']],
                                     tw_cond_imp_df[tw_cond_imp_df['second'] < tw_cond_imp_df['first']],
                                     on='sorted', suffixes=('_X', '_Y'))

    tw_cond_imp_merged_df = tw_cond_imp_merged_df.rename({'s_i|j_X': 's_X|Y', 's_i|j_Y': 's_Y|X',
                                                          'is_important_X': 'X|Y', 'is_important_Y': 'Y|X'}, axis=1)

    tw_cond_imp_merged_df = tw_cond_imp_merged_df[['sorted', 'X|Y', 'Y|X', 's_X|Y', 's_Y|X']]

    return tw_cond_imp_merged_df


def enrich_score_double_conditional_df(double_conditional_df, single_df):
    df = double_conditional_df.reset_index()
    df[['first', 'second']] = (df['labels_in_layers']
                                               .apply((lambda x: pd.Series([x[0], x[1]]))))

    s_ij = (df['diff']
            - single_df.loc[df['second'], 'diff'].values)
    
    s_ij[df['first'] == df['second']] = single_df.loc[df['second'], 'diff'].values
    
    s_ij.index = double_conditional_df.index

    double_conditional_df['s_i|j'] = s_ij
    return double_conditional_df


def draw_tw_cond_dependency_graph(tw_cond_imp_merged_df, single_df,
                                  ax=None):
    
    dependency_type_mask = tw_cond_imp_merged_df.apply(lambda r: not (r['X|Y'] and r['Y|X']),
                                                       axis=1)
    
    dependency_edges = tw_cond_imp_merged_df.loc[dependency_type_mask, 'sorted'].values
    
    dependency_edges = [[f'{start}-{end}' for start, end in node]
                    for node in dependency_edges]
    
    important_nodes = [f'{start}-{end}'
                   for start, end
                   in single_df.index[single_df['taxonomy'] == 'important'].values]
    
    if ax is None:
        _, ax = plt.subplots(1)

    G = nx.DiGraph()
    G.add_nodes_from(important_nodes)
    G.add_edges_from(dependency_edges)

    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')

    nx.draw(G, pos,
            with_labels=True,
            font_weight='bold',
            font_size=25,
            node_size=50,
            # font_color='white',
            width=2,
            ax=ax
           )
    
    return ax