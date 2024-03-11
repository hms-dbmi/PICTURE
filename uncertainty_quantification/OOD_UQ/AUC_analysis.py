
from sklearn import metrics

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import numpy as np
from argparse import ArgumentParser
import scipy.stats as st
import os
import glob
import seaborn as sns
from typing import Literal
import scipy
import copy

ID_COL = 'patient_id'  # column name for patient ID
SCORE_COL = 'prob1'  # column name for class probability
LABEL_COL = 'label'

parser = ArgumentParser()
parser.add_argument('--files', type=str,
                    default= glob.glob('hydra_logs_CV/wMoreBenign/fold*/train/runs/*_best/data/epoch_*/postnet_*.csv'),
                    help="input csv files")
parser.add_argument('--save_folder', type=str, default='OOD_AUC_analysis')
parser.add_argument('--save_name', type=str,
                    default='OOD_roc_curve', help="file name for figure")
parser.add_argument('--n_bootstraps', type=int, default=1000,
                    help='number of bootstrap samples for calculating confidence interval')
parser.add_argument('--interp_fpr', action='store_true',
                    default=True, help='No use for now')
parser.add_argument('--aggregate', action='store_true',
                    default=True, help='Aggregate predictions of subjects')

parser.add_argument('--method', type=str, default='mean', choices=[
                    'mean', 'quantile'], help='method for aggregating predictions of subjects')
parser.add_argument('--threshold', type=int, default=0.5, help='threshold')
parser.add_argument('--conf_threshold', type=float, default=0.05,
                    help='confidence threshold for filtering')
parser.add_argument('--conf_col', type=str, help='column name for confidence score',
                    choices=['pred_uncertainty', 'y_score_aleatoric', 'y_score_epistemic'], default='pred_uncertainty')
parser.add_argument('--filter_mode', type=str, default='LP',
                    choices=['LP', 'HP'], help='filter mode for confidence score')
parser.add_argument('--sort_tpr_on', type=str, default='tpr',
                    choices=['auc', 'tpr'], help='sort tpr based on AUC value or TPR value')
parser.add_argument('--plt_style', type=str, default='bmh',
                    help='pyplot style for plotting')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

ZOOM_RANGE = 0.2

CLASS_DICT = {
    1: 'ID',
    0: 'OoD',
}

KW_LABEL_DICT = {
    'GBM': 'GBM',
    'PCNSL': 'PCNSL',
    'OOD': 'OoD',
    'Benign': 'Benign',
    'Wien-Autopsies': 'Benign'}

SUBTYPE_COARSE_DICT = {
    'GBM': 'GBM',
    'PCNSL': 'PCNSL',
    'OoD': 'OoD',
    'Benign': 'OoD'}

SUBTYPE_LABEL_DICT = {
    'GBM': 'ID',
    'PCNSL': 'ID',
    'OoD': 'OoD',
    'Benign': 'OoD'}


def add_subtype_label(df):
    img_paths = df['img_path']
    # get the subtype from the image path, using the keyword in the kw_label_dict
    df['subtype'] = [[KW_LABEL_DICT[kw] for kw in KW_LABEL_DICT.keys() if kw in img_path][0]
                     for img_path in img_paths]
    df['Cancer Type'] = [[SUBTYPE_COARSE_DICT[kw] for kw in SUBTYPE_COARSE_DICT.keys() if kw in subtype][0]
                         for subtype in df['subtype']]
    return df


def filter_OOD(df, csv='valid_controls_annotation.xlsx'):
    df_anno = pd.read_excel(csv)
    df_normal = df_anno.loc[df_anno['Description'].str.contains(
        'normal', case=False)]
    normal_UUIDs = df_normal['UUID']
    df_filtered = df.loc[
        (df['subtype'] != 'Benign') |
        (df['img_path'].str.contains('Wien-Autopsies')) |
        (df['patient_id'].isin(normal_UUIDs))
    ]
    print(df_filtered.groupby('patient_id').first()
          [['subtype']].value_counts())
    return df_filtered


def plot_roc_CV(y_true_list, y_score_list, positive_class, interp_fpr=True, roc_steps=1000, figsize=(5, 5), zoom=False):
    a = [metrics.roc_curve(y_true=y_true, y_score=y_score)
         for y_true, y_score in zip(y_true_list, y_score_list)]
    fprs = [x[0] for x in a]
    tprs = [x[1] for x in a]
    aucs = np.array([metrics.roc_auc_score(y_true, y_score)
                    for y_true, y_score in zip(y_true_list, y_score_list)])
    fpr_fixed = np.linspace(0, 1, roc_steps)
    auc_mean = np.mean(aucs)
    auc_CI = st.norm.interval(0.95, loc=np.mean(aucs),
                              scale=np.std(aucs)/len(a))
    # if interp_fpr:
    for i in range(len(fprs)):
        tprs[i] = np.interp(fpr_fixed, fprs[i], tprs[i])
        fprs[i] = fpr_fixed
    tprs = np.stack(tprs, axis=0)
    tpr_mean = np.mean(tprs, axis=0)
    tpr_std = np.std(tprs, axis=0)
    tpr_err = tpr_std / np.sqrt(len(a))

    tpr_CI = np.array([st.norm.interval(0.95, loc=mu, scale=sigma)
                      for mu, sigma in zip(tpr_mean, tpr_err)])
    tpr_CI_l = tpr_CI[:, 0]
    tpr_CI_h = tpr_CI[:, 1]
    tpr_CI_l[np.isnan(tpr_CI_l)] = tpr_mean[np.isnan(tpr_CI_l)]
    tpr_CI_h[np.isnan(tpr_CI_h)] = tpr_mean[np.isnan(tpr_CI_h)]

    fig = plt.figure(figsize=figsize)
    lw = 2
    # Plot the original ROC curve
    # plt.style.use('classic')
    plt.style.use(args.plt_style)

    # plt.plot(fpr_fixed, tpr_mean, color='blue', lw=lw,
    #          alpha=0.8, label=f'AUROC: {auc_mean:.3f} (min: {np.min(aucs):.3f}, max: {np.max(aucs):.3f})')
    # plt.plot(fpr_fixed, tpr_mean, color='blue', lw=lw,
    #          alpha=0.8, label=f'AUROC: {auc_mean:.3f} (min: {np.min(aucs):.3f}, max: {np.max(aucs):.3f})')
    plt.plot(fpr_fixed, tpr_mean, color='blue', lw=lw,
             alpha=0.8, label=f'AUROC: {auc_mean:.3f}')
    # Plot confidence interval
    plt.fill_between(fpr_fixed, tpr_CI_l, tpr_CI_h, color='blue', lw=1, alpha=0.1,
                     label=f'confidence interval: [{auc_CI[0]:.3f}, {auc_CI[1]:.3f}]')
    # plot the original ROC curves in dashed lines
    # for i in range(len(fprs)):
    #     plt.plot(fprs[i], tprs[i], color='gray',
    #              lw=1, alpha=1, linestyle=':')
    if not zoom:
        plt.legend(loc='lower right', fontsize='small')

    ##
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    if zoom:
        plt.xlim([0.07, ZOOM_RANGE+0.07])
        plt.ylim([1-ZOOM_RANGE, 1])
    else:
        plt.xlim([-0.0, 1.0])
        plt.ylim([-0.0, 1.0])
    plt.gca().set_aspect('equal', 'box')
    if not zoom:
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
    # plt.grid(True)
    # plt.title(f'ROC for GBM vs PCNSL ({positive_class} is positive class, AUC: {roc_auc:0.02f})')
    # plt.legend(loc="lower right")

    print(f"ROC curve (area = {auc_mean:0.03f})")
    print(f"95% CI = {auc_CI[0]:0.2f} - {auc_CI[1]:0.2f}")

    return fig, auc_mean, auc_CI


def read_pred(result_df, thres=args.threshold,  conf_filter_quantile=0.05, conf_col=args.conf_col, aggregate=args.aggregate,
              mode: Literal['LP', 'HP'] = args.filter_mode, method: Literal['mean', 'quantile'] = 'mean'):
    result_df['pred_uncertainty'] = (
        result_df['y_score_aleatoric'] + result_df['y_score_epistemic'])/2
    if aggregate:
        # group by ID_COL, and keep only the top 5% of the confidence within each ID
        conf_filter = result_df.groupby(
            by=ID_COL).quantile(conf_filter_quantile, numeric_only=True)[conf_col]
        # filter out the rows with confidence lower than the threshold
        dfs = []
        for id in conf_filter.index:
            df = result_df.loc[
                (result_df[ID_COL] == str(id))]
            df = df.loc[(df[conf_col] >= conf_filter[id])] if mode == 'HP' else df.loc[(
                df[conf_col] <= conf_filter[id])]
            # df_debug = result_df.loc[
            #     (result_df[ID_COL] == id)]

            dfs.append(df)
        df_filter = pd.concat(dfs)

        if method == 'mean':
            df_slide = df_filter.groupby(
                by=ID_COL).mean(numeric_only=True)
        elif method == 'quantile':
            df_slide = df_filter.groupby(
                by=ID_COL).quantile(thres, numeric_only=True)
        y = df_slide[LABEL_COL].to_numpy().astype(np.int32)
        probs = df_slide[SCORE_COL].to_numpy()

        print(df_slide[LABEL_COL].value_counts())
        df_out = df_slide
    else:
        y = result_df[LABEL_COL].to_numpy().astype(np.int32)
        probs = result_df[SCORE_COL].to_numpy()
        df_out = result_df
    print("AUROC: ", metrics.roc_auc_score(y, probs))
    return y, probs, df_out



def compute_test_plot(
        result_dfs, thres=args.threshold, save_name=args.save_name,
        subtype_filter=None, plot=True,
):

    y_list = []
    prob_list = []
    # result_dfs = [add_subtype_label(df).reset_index(
    #     drop=True) for df in result_dfs]

    result_dfs = [filter_OOD(add_subtype_label(df)).reset_index(
        drop=True) for df in result_dfs]
    if subtype_filter is not None:
        result_dfs = [df.loc[df['subtype'].isin(subtype_filter)]
                      for df in result_dfs]
    df = pd.concat(result_dfs).reset_index(drop=True)
    # df = add_subtype_label(df).reset_index(drop=True)

    for result_df in result_dfs:
        y, prob, df_out = read_pred(result_df, thres=thres, conf_col=args.conf_col, conf_filter_quantile=args.conf_threshold,
                                    aggregate=args.aggregate, mode=args.filter_mode, method=args.method)
        # print the number of samples in each class
        df_low_conf = df_out.loc[df_out['y_score_epistemic'] < 0.5]
        print(df_low_conf['label'].value_counts())
        print(df_low_conf['label'].value_counts()/df_low_conf.shape[0])

        y_list.append(np.array(y))
        prob_list.append(np.array(prob))

    cohort = 'OOD'
    title = 'OOD'


    save_name_one = save_name + f'_{cohort}.pdf'
    if plot:
        fig, roc_auc, confidence_95 = plot_roc_CV(
            y_list, prob_list, f'{title}')
        fig.savefig(save_name_one)
        plt.close()
    FIGSIZE = (4, 5)

    df_tile = df.copy()
    try:
        df_tile['label'] = [CLASS_DICT[x] for x in df_tile['label']]
    except:
        df_tile['label'] = [SUBTYPE_LABEL_DICT[x] for x in df_tile['subtype']]

    df_tile['confidence'] = df_tile.pop('prob1')
    # replace inf with nan
    df_tile = df_tile.replace([np.inf, -np.inf], np.nan)
    df_tile['dataset'] = 'Vienna\nFFPE'
    if subtype_filter is None:  # only plot the tile-level violin plot when all subtypes are included

        plt.figure(figsize=FIGSIZE)
        sns.set_palette('tab10')
        with sns.axes_style("darkgrid"):
            ax = sns.violinplot(data=df_tile, x='Cancer Type', y='confidence', hue='Cancer Type',
                                order=['GBM', 'PCNSL', 'OoD'],
                                hue_order=['GBM', 'PCNSL', 'OoD'])
            plt.xlabel(None)

            # move legend to the to top
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=False, shadow=False)

            plt.savefig(save_name_one.replace('.pdf', '_violin_tile.pdf'))
    if subtype_filter is not None:
        # if subtype filter is given, print the stats for the filtered subtypes
        x1 = df_tile.loc[df_tile['label'] == 'ID']['confidence'].to_numpy()
        x2 = df_tile.loc[df_tile['label'] == 'OoD']['confidence'].to_numpy()
        stats = scipy.stats.ranksums(x1, x2)
        print("tile-level stats (Wilcoxon rank-sum test):")
        print(stats)
    else:
        # if the subtype filter is not given, print the stats for all subtypes (Kruskal-Wallis Test)
        x_group = [df_tile.loc[df_tile['subtype'] == subtype]['confidence'].to_numpy(
        ) for subtype in df_tile['subtype'].unique()]
        stats = scipy.stats.kruskal(*x_group)
        print("tile-level stats (Kruskal-Wallis Test):")
        print(stats)

    # df['pid_num'], unique_strings = pd.factorize(df['patient_id'])
    df_slide = df.groupby('patient_id').mean(numeric_only=True)
    df2 = df[['subtype', 'patient_id', 'Cancer Type']
             ].groupby('patient_id').first()
    df_slide = df_slide.merge(df2, left_index=True, right_index=True)
    # df_slide = df_slide.merge(df2, on='patient_id')
    df_slide['dataset'] = 'Vienna\nFFPE'

    df_slide['label'] = [CLASS_DICT[x] for x in df_slide['label']]
    df_slide['confidence'] = df_slide.pop('prob1')
    df['dataset'] = 'Vienna\nFFPE'
    save_name_csv = save_name_one.replace('.pdf', '.csv')
    df_slide.to_csv(save_name_csv)

    # if ID prediction and labels are available, also print AUC for ID classification
    if 'ID_label' in df_slide.columns and 'ID_prob' in df_slide.columns and subtype_filter is None:
        df_slide_ID = df_slide.loc[df_slide['label'] == 'ID']
        ID_AUROC = metrics.roc_auc_score(df_slide_ID['ID_label'],
                                         df_slide_ID['ID_prob'])
        print("=========================================")
        print(f"AUC for ID classification: {ID_AUROC:.3f}")
        print("=========================================")

    if subtype_filter is not None:
        # if subtype filter is given, print the stats for the filtered subtypes
        x1 = df_slide.loc[df_slide['label'] == 'ID']['confidence'].to_numpy()
        x2 = df_slide.loc[df_slide['label'] == 'OoD']['confidence'].to_numpy()
        stats = scipy.stats.ranksums(x1, x2)
        print("slide-level stats (Wilcoxon rank-sum test):")
        print(stats)
    else:
        # if the subtype filter is not given, print the stats for all subtypes (Kruskal-Wallis Test)
        x_group = [df_slide.loc[df_slide['subtype'] == subtype]['confidence'].to_numpy(
        ) for subtype in df_slide['subtype'].unique()]
        stats = scipy.stats.kruskal(*x_group)
        print("slide-level stats (Kruskal-Wallis Test):")
        print(stats)

    if subtype_filter is None:  # only plot the tile-level violin plot when all subtypes are included
        plt.close()
        plt.figure(figsize=FIGSIZE)
        with sns.axes_style("darkgrid"):

            ax = sns.violinplot(data=df_slide, x='Cancer Type', y='confidence', hue='Cancer Type',
                                order=['GBM', 'PCNSL', 'OoD'],
                                hue_order=['GBM', 'PCNSL', 'OoD'])
            # move legend to the to top
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=False, shadow=False)

            plt.xlabel(None)

            plt.savefig(save_name_one.replace('.pdf', '_violin_slide.pdf'))

    a = 0


if __name__ == "__main__":
    subtype_filters = [
        ['GBM', 'PCNSL', 'OoD','Benign'],
    ]

    result_dfs = [pd.read_csv(file) for file in args.files]
    for subtype_filter in subtype_filters:
        print("=========================================")
        print(f"subtype_filter: {subtype_filter}")
        print("=========================================")
        subtype_filter_str = "_".join(
            subtype_filter) if subtype_filter is not None else 'AllSubtypes'

        save_folder = args.save_folder
        os.makedirs(save_folder)
        save_name = os.path.join(
            save_folder, f'postnet_OOD_{subtype_filter_str}_10folds_CertainOnly')

        save_name = os.path.join(
            save_folder, f'plotAllFolds_{subtype_filter_str}_10folds_CertainOnly')
        if save_folder != '':
            os.makedirs(save_folder, exist_ok=True)
        compute_test_plot(result_dfs, thres=args.threshold,
                          save_name=save_name, subtype_filter=subtype_filter)

    files_list = copy.deepcopy(args.files)
    for files in files_list:
        files = [files]
        out_prefix = os.path.basename(files[0]).split('.')[0]
        result_dfs = [pd.read_csv(file) for file in files]
        for subtype_filter in subtype_filters:
            print("=========================================")
            print(f"subtype_filter: {subtype_filter}")
            print("=========================================")
            subtype_filter_str = "_".join(
                subtype_filter) if subtype_filter is not None else 'AllSubtypes'
            save_folder = os.path.join(os.path.dirname(
                files[0]))
            save_name = os.path.join(
                save_folder, f'postnet_OOD_{subtype_filter_str}_10folds_CertainOnly')

            save_name = os.path.join(
                save_folder, f'plot_{subtype_filter_str}_{out_prefix}_10folds_CertainOnly')
            if save_folder != '':
                os.makedirs(save_folder, exist_ok=True)
            compute_test_plot(result_dfs, thres=args.threshold,
                              save_name=save_name, subtype_filter=subtype_filter)
