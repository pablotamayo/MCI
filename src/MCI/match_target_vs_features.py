import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.sandbox.stats.multicomp import multipletests
import pyximport
import numpy as np
from time import process_time
from sys import float_info

pyximport.install(setup_args={"include_dirs":np.get_include()}, inplace=True)
 
from .MCI_assoc import MCI_assoc
from .MCI import MCI
from .association_matrix2 import association_matrix2
from .write_dataset import write_dataset
from .cluster_2d_array_slices import cluster_2d_array_slices

def match_target_vs_features(
    target,
    features,
    target_type = 'continuous',
    target_ascending = False,
    normalize_target = True,
    features_type = 'binary',
    features_scores_ascending = False,
    metric = MCI_assoc,
    n_features_plot = 10,
    filter_flat_features = False,
    normalize_features = True,
    truncate_feature_names_n_chars = 130,
    normalize_target_clims = [-3.0, 3.0],
    normalize_features_clims = [-3.0, 3.0],
    add_noise_to_cont_target_to_break_ties = True,
    add_noise_to_cont_features_to_break_ties = True,
    dict_to_map_categorical_to_numeric_vals = None,
    title = None,
    title_font_size = 12,
    CI_n_bootstraps = 100,
    p_val_n_permutations = 1,
    method_to_compute_p_vals = 'direct',
    one_dist_skew_t_method_to_compute_p_vals = False,
    plot_skew_t_distributions = False,  
    plot_histograms_of_p_vals = False,
    continuous_cmap = mpl.cm.get_cmap('bwr'),
    plot_bin_or_categ_target_values = True,
    cluster_within_category = True,
    target_binary_cmap = mpl.colors.ListedColormap(["white", "#FCCE8E", "#0CAFA9"]),
    features_binary_cmap = mpl.colors.ListedColormap(["#bef4ff", "black", "black"]),
    categorical_cmap = mpl.cm.get_cmap('bwr'),
    save_random_perm_data = False,
    save_random_perm_data_file = None,
    figsize = 'auto',
    plot_heatmap = True,
    plot_dpi = 300,
    font_scale = 0.65,
    font_family = "Arial Narrow",
    row_compression = 1.0,
    plot_samples_names = True,
    sample_names_font_size = 8,
    features_fontsize = 9,
    features_fontweight = 'bold',
    use_line_plot_for_binary_features = True,
    use_line_plot_for_binary_features_linewidth = 'auto',
    plot_cont_phen_legend = False,
    fill_na_with_zeroes = False,
    random_num_gene_seed = 1729,
    print_timings = True,
    savefig_format = 'png', # 'png', 'ps', 'pdf', 'svg'
    filepath_prefix = None,
    *args, **kwargs):

    init_start = process_time()

    mpl.rcParams['text.color'] = 'black'
    np.random.seed(random_num_gene_seed)

    if metric == MCI_assoc:
        metric_shortname = 'MCI'
        if target_type == 'binary' or target_type == 'categorical':
            binary_x = True
        else:
            binary_x = False
        if features_type == 'binary' or features_type == 'categorical':
            binary_y = True
        else:
            binary_y = False            

    if isinstance(target, pd.Series):
        target = target.to_frame().T

    if CI_n_bootstraps is None:
        CI_n_bootstraps = 0

    if p_val_n_permutations is None:
        p_val_n_permutations = 0
        
    non_nan_cols = list(target.columns[pd.notnull(target.iloc[0,:])])
    target = target.loc[:, non_nan_cols]
        
    non_nan_cols = list(features.columns[pd.notnull(features.iloc[0,:])])
    features = features.loc[:, non_nan_cols]

    if fill_na_with_zeroes == True:
        target.fillna(value = 0, inplace = True)
        features.fillna(value = 0, inplace = True)
        
    overlap = []
    for s in target.columns:  # this way to compute overlap preserves the target ordering
        if s in features.columns:
            overlap.append(s)
            
    target = target.loc[:, overlap]
    orig_target = target

    target_recoded_flag = False
    if target_type == 'continuous':
        if add_noise_to_cont_target_to_break_ties == True:
            noise = np.random.normal(0, float_info.epsilon, target.shape)
            target = target + noise
        
        if normalize_target == True:
            target = (target - np.float64(target.mean(axis = 1)))/np.float64(target.std(axis = 1))
            target = target.clip(lower = normalize_target_clims[0], upper = normalize_target_clims[1])
            
    elif target_type == 'categorical':  # recode categorical strings as numbers

        target_recoded = pd.DataFrame(0, index = target.index, columns = target.columns)
        if dict_to_map_categorical_to_numeric_vals is None:
            ind = 0.0
            orig_target_vals = {}
            for v in list(target.iloc[0, :].unique()):
                target_recoded.iloc[0, target.iloc[0, :] == v] = ind
                orig_target_vals[ind] = v
                ind = ind + 1.0
        else:
            for s in target.columns:
                target_recoded.loc[target.index[0], s] = dict_to_map_categorical_to_numeric_vals[target.loc[target.index[0], s]]
            orig_target_vals = {}
            for v in dict_to_map_categorical_to_numeric_vals.keys():
                orig_target_vals[dict_to_map_categorical_to_numeric_vals[v]] = v

        target = target_recoded
        target_recoded_flag = True

    if target_ascending == True:
        target.sort_values(by = target.index[0], axis = 1, ascending = True, inplace = True)
        orig_target.sort_values(by = orig_target.index[0], axis = 1, ascending = True, inplace = True)
    elif target_ascending == False :
        target.sort_values(by = target.index[0], axis = 1, ascending = False, inplace = True)
        orig_target.sort_values(by = orig_target.index[0], axis = 1, ascending = False, inplace = True)

    features = features.loc[:, target.columns]

    if filter_flat_features == True:
        features_to_keep = []
        for f in features.index:
            if len(set(features.loc[f,:].values)) != 1:
                features_to_keep.append(f)

        print('Number of features before filtering: {}'.format(features.shape[0]))
        features = features.loc[features_to_keep, :].copy()
        print('Number of features after filtering: {}'.format(features.shape[0]))
    
    if truncate_feature_names_n_chars is not None:
        new_feat_names = []
        if truncate_feature_names_n_chars < 15:
            truncate_feature_names_n_chars = 10
            postfix = 3
        else:
            postfix = 15
            
        for f in features.index:
            if len(f) > truncate_feature_names_n_chars:
                new_feat_names.append(f[0:(truncate_feature_names_n_chars - postfix)] + ' .... ' + f[len(f) - postfix:len(f)])
            else:
                new_feat_names.append(f)

        features.index = new_feat_names

    if features_type == 'continuous':
 
        if add_noise_to_cont_target_to_break_ties == True:
            noise = np.random.normal(0, float_info.epsilon, features.shape)
            features = features + noise

        if normalize_features == True:
             
            D = features.T            
            D1 = ((D - D.mean(axis=0))/D.std(axis=0)).clip(lower = normalize_features_clims[0],
                                                           upper = normalize_features_clims[1])
            features = D1.T

    print('target size: {} features size: {}'.format(target.shape, features.shape))
 
    start = process_time()
   
    if metric == MCI_assoc:

        target_vs_features = MCI_assoc(target.iloc[0,:], features, binary_x = binary_x, binary_y = binary_y)
        target_vs_features = pd.Series(target_vs_features, index = features.index)
        
    else:
        target_vs_features = association_matrix2(target, features, axis = 1,
                                            assoc_function = metric,  
                                                 make_plot = False, *args, **kwargs).iloc[0, :]

    if features_scores_ascending == True:
        target_vs_features = target_vs_features.sort_values(ascending = True) 
    elif features_scores_ascending == False:
        target_vs_features = target_vs_features.sort_values(ascending = False)
        
    A = features.copy()
    N = A.shape[0]

    if isinstance(n_features_plot, int):
        if n_features_plot > np.floor(N/2):
            n_features_plot = np.int_(np.floor(N/2))
        
        features_plot = list(target_vs_features.index[range(n_features_plot)]) + list(target_vs_features.index[range(N - n_features_plot, N)])
    else:
        n_features_plot = target_vs_features.size
        features_plot = target_vs_features.index

    target_vs_features_plot = target_vs_features.loc[features_plot]
    target_vs_features2_plot = target_vs_features_plot.to_frame().copy()

    A = A.loc[features_plot, :]
    A.index.name = ' '

    if (cluster_within_category and target_type in ('binary', 'categorical')):

        print("Clustering heat map within category ...")

        clustered_indices = cluster_2d_array_slices(A, axis = 1, groups = target.iloc[0, :], raise_for_bad = False)

        target = target.iloc[0, clustered_indices].to_frame().T

        A = A.iloc[:, clustered_indices]

    if print_timings is True:
        print('time computing observed scores: {} secs.'.format(np.round(process_time() - start, 3)))

    # Compute confidence intervals

    start = process_time()
        
    if CI_n_bootstraps > 0:
        start = process_time()
        bootstrap_ICs = pd.DataFrame(0.0, index = A.index, columns = range(CI_n_bootstraps))
        IC_CI = pd.Series(0.0, index = A.index)

        for k in range(CI_n_bootstraps):
            bootstrap_samples = list(np.random.choice(target.columns, size = target.size, replace = True))

            if metric == MCI_assoc:
                bootstrap_ICs.loc[:, k] = MCI_assoc(target.loc[target.index[0], bootstrap_samples], A.loc[:, bootstrap_samples])
                
            else:
            
                bootstrap_ICs.loc[:, k] = association_matrix2(target.loc[target.index[0], bootstrap_samples].to_frame().T, 
                                                      A.loc[:, bootstrap_samples], axis = 1, assoc_function = metric,
                                                      make_plot = False, *args, **kwargs).iloc[0, :]  
        for f in A.index:
            bootstraps = bootstrap_ICs.loc[f, pd.notnull(bootstrap_ICs.loc[f, :])]
            IC_CI.loc[f] = np.abs(target_vs_features_plot.loc[f] - np.percentile(bootstraps, 0.90))

    if print_timings is True:
        print('time computing bootstrap scores: {} secs.'.format(np.round(process_time() - start, 3)))

    # Compute p-values and FDRs

    start = process_time()
        
    if p_val_n_permutations > 0:
            
        if save_random_perm_data == True: 
            random_perm_data = pd.DataFrame(0, index = features.index, columns = range(p_val_n_permutations))

        null_metric_dist = []
        
        for k in range(p_val_n_permutations):
 
            rand_target = pd.Series(target.iloc[0, np.random.permutation(range(target.shape[1]))].values, index = target.columns)
            rand_target = rand_target.to_frame().T
 
            if metric == MCI_assoc:

                    scores = MCI_assoc(rand_target.iloc[0,:], features)
                    rand_perm_scores = pd.Series(scores, index = features.index)

            else:

                rand_perm_scores = association_matrix2(rand_target,
                                                  features, axis = 1, assoc_function = metric, *args, **kwargs).iloc[0, :]
            
            null_metric_dist.extend(rand_perm_scores)
            
            if save_random_perm_data == True:
                random_perm_data.loc[:, k] = rand_perm_scores

        if save_random_perm_data_file is not None:
            write_dataset(random_perm_data, save_random_perm_data_file)
            
        if print_timings is True:                
            print('time computing random permutation scores: {} secs.'.format(np.round(process_time() - start, 3)))

            # Compute histogram of null dist

        start = process_time()

        null_metric_dist = [x for x in null_metric_dist if not np.isnan(x)]
        
        target_vs_features2 = target_vs_features.values.flatten()
        n_bins = np.int_(np.sqrt(len(null_metric_dist)))
        hist, bins = np.histogram(null_metric_dist, bins = n_bins, density = False)
        bin_width = np.abs(bins[0] - bins[1])/2
        bins = np.append(np.insert(bins, 0, bins[0] - bin_width), bins[-1] + bin_width)
        mid_bins = [(x + y)/2 for x, y in zip(bins[:-1], bins[1:])]
        hist = np.append(np.insert(hist, 0, 0), 0)
        hist = hist/hist.sum()
        if plot_histograms_of_p_vals == True:
                plt.xlabel('Metric Scores')
                plt.ylabel('Density')
                plt.title('Histogram of null distribution of metric scores')
                plt.plot(mid_bins, hist)
                plt.show()

        # Compute p-values

        if method_to_compute_p_vals == 'direct':
           
           # Compute p-vals directly from null distribution
            
            null_metric_dist = pd.Series(null_metric_dist)
            p_vals_up = pd.Series([(null_metric_dist >= x).sum()/null_metric_dist.size for x in target_vs_features],
                              index = target_vs_features.index)
            p_vals_up = np.where(p_vals_up == 0, 1.0/null_metric_dist.size, p_vals_up)
           
            p_vals_dn = pd.Series([(null_metric_dist <= x).sum()/null_metric_dist.size for x in target_vs_features],
                              index = target_vs_features.index)
            p_vals_dn = np.where(p_vals_dn == 0, 1.0/null_metric_dist.size, p_vals_dn)
           
            p_vals = pd.Series(np.where(p_vals_up < p_vals_dn, p_vals_up, p_vals_dn), index = target_vs_features.index)

        elif method_to_compute_p_vals == 'histogram':
            
            # Compute p-vals from null distribution histogram
            
            hist_cumsum = np.round(hist.cumsum(), 10)
            hist_cumsum_rev = 1 - hist_cumsum
            p_vals_up = []
            p_vals_dn = []
            p_vals = []
            for i in range(len(target_vs_features)):
                diffs = list(np.abs(target_vs_features2[i] - mid_bins))
                min_index = diffs.index(min(diffs))
                p_vals_up.append(hist_cumsum_rev[min_index])
                p_vals_dn.append(hist_cumsum[min_index])
    
            p_vals = [x if x < y else y for x, y in zip(p_vals_up, p_vals_dn)]
            p_vals = [1.0/len(null_metric_dist) if x == 0.0 else x for x in p_vals]

            p_vals = pd.Series(p_vals, index = target_vs_features.index)

        else:
            raise('unknown method to compute p-vals: {}'.format(method_to_compute_p_vals))
            
        if plot_histograms_of_p_vals == True:
            plt.xlabel('Metric Scores')
            plt.ylabel('Log(p-value)')
            plt.title('Plot of p-values vs. metric scores')
            plt.plot(target_vs_features, np.log(p_vals))
            plt.show()

        if print_timings is True:                            
            print('time computing p-vals: {} secs.'.format(np.round(process_time() - start, 3)))

        start = process_time()

        FDRs_up = multipletests(p_vals_up, method = 'fdr_bh')[1]
        FDRs_dn = multipletests(p_vals_dn, method = 'fdr_bh')[1]
        FDRs = pd.Series(np.where(FDRs_up < FDRs_dn, FDRs_up, FDRs_dn), index = target_vs_features.index)
        
        if print_timings is True:                            
            print('time computing FDRs: {} secs.'.format(np.round(process_time() - start, 3)))
        
    # Prepare scores and stats for heatmap
    if CI_n_bootstraps > 0 and p_val_n_permutations > 0: 
        target_vs_features_names_plot = pd.Series(['{:.2f} ({:.3f})   {:.2e}    {:.2e}'.format(x, y, z, q) 
                                            for x, y, z, q in zip(target_vs_features_plot,
                                            IC_CI,
                                            p_vals.loc[features_plot],
                                            FDRs.loc[features_plot])],
                                            index = target_vs_features_plot.index)
    elif CI_n_bootstraps == 0 and p_val_n_permutations > 0:
       target_vs_features_names_plot = pd.Series(['{:.2f}   {:.2e}    {:.2e}'.format(x, z, q) 
                                            for x, z, q in zip(target_vs_features_plot,
                                            p_vals.loc[features_plot],
                                            FDRs.loc[features_plot])],
                                            index = target_vs_features_plot.index)
    elif CI_n_bootstraps > 0 and p_val_n_permutations == 0:
        target_vs_features_names_plot = pd.Series(['{:.2f} ({:.3f}) '.format(x, y) 
                                            for x, y, in zip(target_vs_features_plot,
                                            IC_CI)],
                                            index = target_vs_features_plot.index)
    elif CI_n_bootstraps == 0 and p_val_n_permutations == 0:
        target_vs_features_names_plot = pd.Series(['{:.2f} '.format(x) 
                                           for x in target_vs_features_plot],
                                           index = target_vs_features_plot.index)
    else:
        raise('ERROR: invalid values for CI_n_bootstraps ({}) and/or p_val_n_permutations ({})'.format(CI_n_bootstraps, p_val_n_permutations))
       
    if CI_n_bootstraps > 0 and p_val_n_permutations > 0:

        res_table = pd.concat([target_vs_features, IC_CI, p_vals, FDRs], axis = 1)
        res_table.columns = [metric_shortname, '$\pm \Delta^{90\%}_{CI}$', 'p-val', 'FDR']
    elif CI_n_bootstraps == 0 and p_val_n_permutations > 0:
        res_table = pd.concat([target_vs_features, p_vals, FDRs], axis = 1)
        res_table.columns = [metric_shortname, 'p-val', 'FDR']
    elif CI_n_bootstraps > 0 and p_val_n_permutations == 0:
        res_table = pd.concat([target_vs_features, IC_CI], axis = 1)
        res_table.columns = [metric_shortname, '$\pm \Delta^{90\%}_{CI}$']
    elif CI_n_bootstraps == 0 and p_val_n_permutations == 0:
        res_table = target_vs_features.to_frame()
        res_table.columns = [metric_shortname]
       
    # plot heatmap
    if plot_heatmap == True:
        
        if figsize == 'auto':
            x_size = 10
            y_size = row_compression * (2 + n_features_plot/3.75)
        else:
            x_size, y_size = figsize

        sns.set(font_scale = font_scale)
    
        #plt.tight_layout(pad = 0, w_pad  =0, h_pad = 0)
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [font_family]

        if plot_cont_phen_legend == False:
    
            fig, axs = plt.subplots(2, 2, figsize = (x_size, y_size), dpi = plot_dpi,
                        gridspec_kw={'height_ratios': [1.5, np.int_(2*n_features_plot)], 
                        'width_ratios': [26, 1], 'hspace': (1.75/n_features_plot), 'wspace': 0.025})
        else:
            fig, axs = plt.subplots(3, 2, figsize = (x_size, y_size), dpi = plot_dpi,
                        gridspec_kw={'height_ratios': [1.5, np.int_(2*n_features_plot), 1], 
                        'width_ratios': [16, 1], 'hspace': (1.75/n_features_plot), 'wspace': 0.025})
   
        if target_type == 'continuous':
            target_cmap = continuous_cmap
        elif target_type == 'binary':
            target_cmap = target_binary_cmap
        else:
            target_cmap = target_binary_cmap
    
        g = sns.heatmap(target, cmap = target_cmap, ax = axs[0][0],
                        linewidths = 0, linecolor = 'black', robust = False, 
                        annot = False, vmin = -2.0, vmax = 2.0, cbar = False, square = False, xticklabels = False, 
                            yticklabels = list(target.index), mask = None)
        g.set_yticklabels(g.get_yticklabels(),  rotation = 0,
                              fontdict = {'fontsize': features_fontsize, 'fontweight': features_fontweight})

        if plot_bin_or_categ_target_values == True and target_type in ('binary', 'categorical'):
            target_vals = list(target.iloc[0, :].unique())
            loc = 0
            for v in target_vals:
                v_size = np.sum(target.iloc[0, :] == v)
                if target_recoded_flag is True:
                    val = orig_target_vals[v]
                else:
                    val = v
                
                g.text(loc + v_size/2, 1, val, horizontalalignment='center', verticalalignment='bottom',
                           fontsize = features_fontsize) #, fontweight = 'bold')
                loc = loc + v_size
    
        if title != None:
            g.text(target.size/2, -1, title, horizontalalignment='center', verticalalignment='bottom',
                           fontsize = title_font_size, fontweight = 'bold')

        axs[0][1].set_visible(False)

        if features_type == 'continuous':
            features_cmap = continuous_cmap
        elif features_type == 'binary':
            features_cmap = features_binary_cmap
        else:
            features_cmap = categorical_cmap    

        if features_type == 'continuous':
            fea_vmin = normalize_features_clims[0]
            fea_vmax = normalize_features_clims[1]
        elif features_type == 'binary':
            fea_vmin = -0.2
            fea_vmax = 1

        if plot_samples_names == True:
            xticklabels = 'auto'
        else:
            xticklabels = False

        if features_type == 'binary':

            if use_line_plot_for_binary_features:

                # make tickness of the lines inversely proportional to the density of lines
                 
                A_null = pd.DataFrame(0, index = A.index, columns = A.columns)
                h = sns.heatmap(A_null, cmap = features_cmap, ax = axs[1][0], linewidths = 0, linecolor = 'black', robust = False, 
                    annot = False, vmin = fea_vmin, vmax = fea_vmax, cbar = False, square = False, 
                    xticklabels = xticklabels, yticklabels = 'auto', mask = None)

                for i in range(A.shape[0]):

                    if use_line_plot_for_binary_features_linewidth == 'auto':
                        density = A.iloc[i, :].sum()/A.shape[1]
                        if density > 0.3:
                            density = 0.3
                        computed_linewidth = 1/(25 * density + 1)
                    else:
                        computed_linewidth = use_line_plot_for_binary_features_linewidth
                    
                    for j in range(A.shape[1]):
                        if A.iloc[i, j] == 1:
                            h.plot((j, j), (i, i+1), color = 'black', linewidth = computed_linewidth)

            else:

                h = sns.heatmap(A, cmap = features_cmap, ax = axs[1][0], linewidths = 0, linecolor = 'black', robust = False, 
                    annot = False, vmin = fea_vmin, vmax = fea_vmax, cbar = False, square = False, 
                    xticklabels = xticklabels, yticklabels = 'auto', mask = None)

        else:
            
            h = sns.heatmap(A, cmap = features_cmap, ax = axs[1][0], linewidths = 0, linecolor = 'black', robust = False, 
                annot = False, vmin = fea_vmin, vmax = fea_vmax, cbar = False, square = False, 
                xticklabels = xticklabels, yticklabels = 'auto', mask = None)

        h.set_yticklabels(h.get_yticklabels(), rotation = 0, minor = False,
                               fontdict = {'fontsize': features_fontsize, 'fontweight': features_fontweight})
        h.set_xticklabels(h.get_xticklabels(), minor = False, fontsize = sample_names_font_size)

        f = sns.heatmap(target_vs_features2_plot, yticklabels = target_vs_features_names_plot,
                    cmap = continuous_cmap, ax = axs[1][1], linewidths = 0, linecolor = 'black', robust = False, 
                    annot = False, vmin = -1, vmax = 1, cbar = False, square = False, xticklabels = False, mask = None)
        
        f.yaxis.tick_right()
        f.set_yticklabels(f.get_yticklabels(), rotation = 0, minor = False) # fontsize = 8, 
        f.tick_params(axis='both', which='both', length=0)
 
        if CI_n_bootstraps > 0 and p_val_n_permutations > 0: 
            ylabel = r'              {} ($\pm \Delta$)    p-val         FDR'.format(metric_shortname)
        if CI_n_bootstraps == 0 and p_val_n_permutations > 0:    
            ylabel = r'   {}         p-val         FDR'.format(metric_shortname)
        if CI_n_bootstraps > 0 and p_val_n_permutations == 0:             
            ylabel = r'   {} ($\pm \Delta$)  '.format(metric_shortname)
        if CI_n_bootstraps == 0 and p_val_n_permutations == 0:
            ylabel = r' {} '.format(metric_shortname)

        f.set_ylabel('')
        f.text(4*font_scale, 0, ylabel, horizontalalignment='center', verticalalignment='bottom',
                           fontsize = font_scale*12, fontweight = 'bold')
            
        if plot_cont_phen_legend == True:

            g = sns.heatmap(target, cmap = target_cmap, ax = axs[2][0],
                            linewidths = 0, linecolor = 'black', robust = False, 
                            annot = False, vmin = -2.0, vmax = 2.0, cbar = False, square = False,
                            xticklabels = np.round(orig_target.values[0], 2),
                            yticklabels = list(target.index), mask = None)

            axs[2][1].set_visible(False)
            
        plt.show()
        
        if filepath_prefix is not None:
            fig.savefig(fname = '{}.{}'.format(filepath_prefix, savefig_format), dpi = plot_dpi) # , bbox_inches = 'tight'

    if filepath_prefix is not None:
        write_dataset(res_table, '{}.gct'.format(filepath_prefix))

    if print_timings is True:                            
            print('Total time: {} secs.'.format(np.round(process_time() - init_start, 3)))

    return(res_table)

