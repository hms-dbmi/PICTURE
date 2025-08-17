import numpy as np
import  argparse
import matplotlib.pyplot as plt
import glob
from os.path import join, basename, dirname
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, f1_score
from tqdm import tqdm
import os
import seaborn as sns
from typing import Literal
from matplotlib.colors import LinearSegmentedColormap
import umap

ID_COL = 'patient_id'  # column name for patient ID
SCORE_COL = 'prob1'  # column name for class probability
LABEL_COL = 'label'

# KW_LABEL_DICT = {
#     'GBM':'GBM',
#     'PCNSL': 'PCNSL',
#     'OOD':'OoD',
#     'Benign': 'Benign',
#     'Wien-Autopsies': 'Benign'}
KW_LABEL_DICT = {
    'GBM':'GBM',
    'PCNSL': 'PCNSL',
    'OOD':'OoD',
    'Benign': 'OoD',
    'Wien-Autopsies': 'OoD'}

def pargse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inpath_sstr', type=str,
                        # default='hydra_logs_CV/wBenign/fold*/train/runs/*_best')
                        default='hydra_logs_CV/wMoreBenign/fold*/train/runs/*best')
    parser.add_argument('--fold', type=int, default=5,help='fold number')
    parser.add_argument('--vis_method', type=str, default='umap',help='tsne or umap',choices=['tsne','umap'])
    parser.add_argument('--redo_tsne', action='store_true',default=True)
    parser.add_argument('--outpath', type=str, default='tsne_moreBenign_AllTiles')
    # parser.add_argument('--q_filter', type=float, default=0.05,help='quantile for filtering confident samples')
    # parser.add_argument('--filter_mode', type=str, default='LP',help='LP or HP')
    # parser.add_argument('--q_filter', type=float, default=0.05,help='quantile for filtering confident samples')
    # parser.add_argument('--filter_mode', type=str, default='LP',help='LP or HP')
    # parser.add_argument('--conf_col', type=str, default='pred_uncertainty',help='column name for confidence',choices=['pred_uncertainty','y_score_epistemic','y_score_aleatoric'])
    parser.add_argument('--perplexity', type=int, default=50,help='perplexity for tsne')
    parser.add_argument('--early_exaggeration', type=int, default=12,help='early_exaggeration for tsne')
    parser.add_argument('--min_dist', type=float, default=0.5,help='min_dist for umap')
    parser.add_argument('--add_slide_pred', type=str,choices=['pred','prob','none'], default='prob',help='add slide level prediction or probability' )
    parser.add_argument('--balanced_sample', action='store_true',default=True)


    args = parser.parse_args()
    return args

def add_subtype_label(df):
    # kw_label_dict = {
    #     'GBM':'GBM',
    #     'PCNSL': 'PCNSL',
    #     'OOD':'OOD',
    #     'Benign': 'Benign'}
    img_paths = df['img_path']
    # get the subtype from the image path, using the keyword in the kw_label_dict
    df['subtype'] = [[KW_LABEL_DICT[kw] for kw in KW_LABEL_DICT.keys() if kw in img_path][0] for img_path in img_paths]
    return df



def add_slide_pred(df, thres=0.5, method: Literal['mean', 'quantile'] = 'mean',col:Literal['pred','prob'] = 'prob'):
        # group by ID_COL, and keep only the top 5% of the confidence within each ID

    if method == 'mean':
        df_slide = df.groupby(
            by=ID_COL).mean(numeric_only=True)
    elif method == 'quantile':
        df_slide = df.groupby(
            by=ID_COL).quantile(thres, numeric_only=True)
    df_slide['OOD_pred'] = (df_slide['prob1'] > 0.5).astype(int)
    df_slide['ID_pred'] = (df_slide['ID_prob'] > 0.5).astype(int)
    if col == 'prob':
        df_slide['OOD_pred'] = df_slide['prob1']
        df_slide['ID_pred'] = 1 - df_slide['ID_prob']
    cols = ['OOD_pred','ID_pred']
    df = df.merge(df_slide[cols], left_on=ID_COL, right_index=True)

    return df




# def export_tsne_figure(y_true, log_entropy, outpath,
def export_tsne_figure(df, label_col, uncertainty_col, outpath,
                       vis_method: Literal['tsne', 'umap'] = 'tsne',
                       embeddings=None,embeddings_tsne=None, seed=42,
                       add_slide_pred:Literal['pred','prob','none']='prob',
                       perplexity=30, early_exaggeration=12,balanced_sample=False,
                       min_dist=0.1,
                       marker_size=4, marker_alpha=0.5,line_width=1):
    y_true = df[label_col]
    uncertainty = df[uncertainty_col]

    if vis_method == 'tsne':
        method_str = f'{vis_method}_Seed{seed}Perplex{perplexity}Exaggerate{early_exaggeration}'
    else:
        method_str = f'{vis_method}_Seed{seed}Perplex{perplexity}MinDist{min_dist}'

    REPORT_OUTPATH = os.path.join(outpath,f'{method_str}_add{add_slide_pred}')
    os.makedirs(REPORT_OUTPATH,exist_ok=True)
    df.to_csv(os.path.join(REPORT_OUTPATH,'results.csv'))
    # Visualize the t-sne
    assert embeddings is not None or embeddings_tsne is not None, "Either embeddings or embeddings_tsne must be provided"
    if embeddings_tsne is None:
        # scale the data
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
        #
        if vis_method == 'tsne':
            transform = TSNE(n_components=2, init="pca", 
                        random_state=seed, n_jobs=4, learning_rate="auto", 
                        perplexity=perplexity, early_exaggeration=early_exaggeration,verbose=1)
        else:
            transform = umap.UMAP(n_components=2, n_neighbors=perplexity, min_dist=min_dist, 
                             random_state=seed,verbose=True)
        if balanced_sample:
            ## randomly sample each class to balance the sample
            # get the number of samples for each class
            n_sample = y_true.value_counts().min()
            df_idx = pd.DataFrame({
                'idx':np.arange(len(y_true)),
                'subtype':y_true,
            })
            df_idx = df_idx.groupby('subtype').sample(n=n_sample)
            ## subsample the data, label and log_entropy
            if vis_method == 'tsne':
                # if using tsne, subsample the data and plot the subsampled data
                embeddings = embeddings[df_idx['idx'],:]
                y_true = y_true.iloc[df_idx['idx']]
                uncertainty = uncertainty.iloc[df_idx['idx']]
                # fit the tsne using subsampled data
                embeddings_tsne = transform.fit_transform(embeddings)
            else:
                # if using umap, fit umap with subsampled data, but plot the full data
                transform.fit(embeddings[df_idx['idx'],:])
                embeddings_tsne = transform.transform(embeddings)



        else:
            embeddings_tsne = transform.fit_transform(embeddings)
        # save embeddings
        np.save(os.path.join(REPORT_OUTPATH,f"{vis_method}_embeddings.npy"), embeddings_tsne)



    # Create a color map for the two classes using the viridis colormap
    # colors = {
    #     0: (0.267004, 0.004874, 0.329415, 1.0),
    #     1: (0.993248, 0.906157, 0.143936, 1.0),
    #     }
    colors = { #refered to https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
        'GBM': '#D81B60',
        'PCNSL': '#1E88E5',
        'OoD':'#FFC107',
        'Benign': '#004D40'}
    
    # create custom colormaps
    # nodes = [0.0, 0.4, 0.8, 1.0]
    # create a colormap from light red to dark red
    nodes = [0.0, 1.0]
    colors_red = ['pink', 'darkred']
    cmap_red = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors_red)))
    # create a colormap from light blue to dark blue
    colors_blue = ['lightblue', 'darkblue']
    cmap_blue = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors_blue)))
    # create a colormap from light green to dark green
    colors_green = ['lightgreen', 'darkgreen']
    color_yellow = ['xkcd:yellow', 'xkcd:yellowish brown']
    color_yellow = ['bisque', 'darkorange']
    cmap_green = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors_green)))
    cmap_yellow = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, color_yellow)))

    colormaps = {
        'GBM': cmap_red,
        'PCNSL': cmap_blue,
        'OoD':cmap_yellow}
    # labels = KW_LABEL_DICT.values().tolist()
    class_markers =['^' if t else 'o' for t in y_true]
    # class_labels = np.array([labels[t] for t in y_true])
    class_labels = y_true
    class_colors = np.array([colors[t] for t in y_true])


    SUBTYPE_NUM_DICT = {
        'GBM':0,
        'PCNSL':0,
        'OoD':1,
        'Benign': 1}
    # Compute F1 score for each threshold value
    y_true_id_ood = np.array([SUBTYPE_NUM_DICT[y] for y in y_true])
    thresholds = sorted(uncertainty)
    thresholds = thresholds[2:-2]
    thresholds = np.linspace(np.quantile(uncertainty,q=0.01), np.quantile(uncertainty,q=0.99), 100)
    f1_scores = []
    for threshold in tqdm(thresholds):
        y_pred = (uncertainty >= threshold).astype(int)
        # tn, fp, fn, tp = confusion_matrix(y_true_id_ood, y_pred).ravel()
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # f1_score = 2 * (precision * recall) / (precision + recall)
        f1 = f1_score(y_true_id_ood, y_pred)
        f1_scores.append(f1)

    # Find the threshold that maximizes the F1 score
    max_f1_score = max(f1_scores)
    optimal_threshold = thresholds[f1_scores.index(max_f1_score)]

    # Set up the figure with two subplots
    fig1, ax1 = plt.subplots(figsize=(13,10))
    fig2, ax2 = plt.subplots(figsize=(20,10))
    fig3, ax3 = plt.subplots(figsize=(15,10))
    print('plotting....')

    # Plot the t-SNE embeddings with class labels
    class_order = [ 'OoD', 'GBM','PCNSL']
    marker_size_dict = {
        'GBM': 2,
        'PCNSL': 2,
        'OoD': 1,
    }
    for label in class_order:
        idxs = np.where(class_labels == label)
        ax1.scatter(embeddings_tsne[idxs, 0], embeddings_tsne[idxs, 1], c=class_colors[idxs], label=label, alpha=marker_alpha, s=marker_size_dict[label], marker='o',linewidths=line_width, cmap="RdYlGn")

    # Set plot title and labels
    ax1.set_title("Latent Space Embedding")
    ax1.set_xlabel("t-SNE Component 1")
    ax1.set_ylabel("t-SNE Component 2")

    # Add legend
    ax1.legend(loc="upper left")
    fig1.savefig(os.path.join(REPORT_OUTPATH,f"{vis_method}_latent_space_{seed}.jpg"), dpi=600)

    # fig1.savefig(os.path.join(REPORT_OUTPATH,f"latent_space_{seed}.jpg"), dpi=600)
    # Plot the t-SNE embeddings with each class highlighted
    
    for hightlight_label in np.unique(class_labels):
        plt.close(fig1)
        fig1, ax1 = plt.subplots(figsize=(13,10))
        for label in np.unique(class_labels):
            idxs = np.where(class_labels == label)
            marker_size_single = 4 if label == hightlight_label else 1
            marker_alpha_single = 1 if label == hightlight_label else 0.25
            marker = 'o' if label == hightlight_label else '.'
            line_width_single = 1 if label == hightlight_label else 1
            
            ax1.scatter(embeddings_tsne[idxs, 0], embeddings_tsne[idxs, 1], c=class_colors[idxs], label=label, alpha=marker_alpha_single, s=marker_size_single, marker=marker,linewidths=line_width_single, cmap="RdYlGn")
        fig1.savefig(os.path.join(REPORT_OUTPATH,f"{vis_method}_latent_space_{hightlight_label}_{seed}.jpg"), dpi=600)
        plt.close(fig1)


    # Plot the t-SNE embeddings with predicted uncertainty
    from sklearn.preprocessing import MinMaxScaler

    # create scaler object
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = StandardScaler()

    # assume data is stored in a numpy array called 'data'
    # create a mask that identifies elements above and below the threshold
    mask = uncertainty >= optimal_threshold

    # create a copy of the data array to work with
    scaled_data = np.copy(uncertainty)

    # apply different scaling factors to the elements above and below the threshold
    if len(scaled_data[~mask])>0:
        scaled_data[~mask] = scaler.fit_transform(scaled_data[~mask].reshape(-1, 1)).ravel() * 0.5
    if len(scaled_data[mask])>0:
        scaled_data[mask] = scaler.fit_transform(scaled_data[mask].reshape(-1, 1)).ravel() * 0.5 + 0.5
    df['scaled_uncertainty'] = scaled_data
    # the resulting 'scaled_data' array will have values between 0 and 1,
    # with values under the threshold mapped to values between 0 and 0.5,
    # and values above the threshold mapped to values between 0.5 and 1

    # Plot the log entropy values
    df['X'] = embeddings_tsne[:, 0]
    df['Y'] = embeddings_tsne[:, 1]

    
    # im = ax2.scatter(df['X'], df['Y'], c=df['ranked_uncertainty'], cmap='viridis', s=marker_size, linewidths=line_width,alpha=marker_alpha)
    ims = []
    for label in class_order:
        idxs = np.where(class_labels == label)
        df_sub = df.iloc[idxs]
        colormap = colormaps[label]
        im = ax2.scatter(df_sub['X'], df_sub['Y'], c=df_sub['scaled_uncertainty'], cmap=colormap, s=marker_size_dict[label], vmin=0, vmax=1,  marker='o',
                         linewidths=line_width,alpha=marker_alpha)
        ims.append(im)
    #Add colorbars
    for im in ims:
        cbar = plt.colorbar(im, ax=ax2)

        # cbar = plt.colorbar(im, ax=ax2)


        # ax2.scatter(embeddings_tsne[idxs, 0], embeddings_tsne[idxs, 1], c=ranked_uncertainty[idxs], label=label, alpha=marker_alpha, s=marker_size, marker='.',linewidths=line_width, cmap="viridis")

    ax2.set_title("Predicted uncertainty")
    ax2.set_xlabel("t-SNE Component 1")
    ax2.set_ylabel("t-SNE Component 2")

    # Add color bar
    # cbar = plt.colorbar(im, ax=ax2)
    # cbar.set_label("Predicted uncertainty")

    # Save the figures
    # make fig2 a tight layout
    fig2.tight_layout()
    fig2.savefig(os.path.join(REPORT_OUTPATH,f"{vis_method}_predicted_entropy_{seed}.jpg"), dpi=600)

    # class_markers =['^' if t else 'o' for t in y_true]


    # # ax3.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],marker = class_markers, c=scaled_data, cmap='viridis', s=4)
    # im1 = ax3.scatter(embeddings_tsne[y_true=='GBM', 0], embeddings_tsne[y_true=='GBM', 1],marker = 'o',
    #                   c=scaled_data[y_true=='GBM'], cmap='viridis', linewidths=line_width, s=marker_size, alpha=marker_alpha, label='GBM')
    # im2 = ax3.scatter(embeddings_tsne[y_true=='PCNSL', 0], embeddings_tsne[y_true=='PCNSL', 1],marker = 'o', 
    #                   c=scaled_data[y_true=='PCNSL'], cmap='viridis',  linewidths=line_width, s=marker_size,  alpha=marker_alpha, label='PCNSL')
    # im3 = ax3.scatter(embeddings_tsne[y_true=='OoD', 0], embeddings_tsne[y_true=='OoD', 1],marker = '+', 
    #                   c=scaled_data[y_true=='OoD'], cmap='viridis',  linewidths=line_width, s=marker_size,  alpha=marker_alpha, label='OoD')
    # im4 = ax3.scatter(embeddings_tsne[y_true=='Benign', 0], embeddings_tsne[y_true=='Benign', 1],marker = 'x', 
    #                   c=scaled_data[y_true=='Benign'], cmap='viridis', linewidths=line_width, s=marker_size, alpha=marker_alpha, label='Benign')

    # # Add legend
    # ax3.legend(loc="upper left")
    # ax3.set_title("Predicted uncertainty")
    # ax3.set_xlabel("t-SNE Component 1")
    # ax3.set_ylabel("t-SNE Component 2")

    # # Add color bar
    # cbar = plt.colorbar(im, ax=ax3)
    # cbar.set_label("Predicted uncertainty")

    # # Save the figures
    # fig3.savefig(os.path.join(REPORT_OUTPATH,f"{vis_method}_predicted_entropy_with_markers{seed}.jpg"), dpi=600)
    return

def aggregate_slide(df):
    # get average confidence for each slide
    # get string label for each slide
    df_slide_label = df[['subtype','patient_id']].groupby('patient_id').first()
    df_slide = df.groupby('patient_id').mean(numeric_only=True)
    df_slide = df_slide.merge(df_slide_label, left_index=True, right_index=True)
    return df_slide

def main(params):
    find_folders = glob.glob(params.inpath_sstr)
    for inpath in find_folders:
        # find data from the input path\
        fold = inpath.split('/')[-4]
        if int(fold.split('fold')[-1]) != params.fold:
            continue
        print(f'processing {fold}')
        data_csvs = glob.glob(join(inpath, 'data','epoch*','postnet*.csv'))
        if len(data_csvs) == 0:
            continue
        data_csv = data_csvs[0]
        tsne_csv = glob.glob(join(inpath, 'data','epoch*','tsne*.csv'))[0]
        outpath = join(params.outpath,fold)
        os.makedirs(outpath,exist_ok=True)
        # data_npy = glob.glob(join(params.inpath, 'data','epoch*','postnet*.npy'))[0]
        # data = np.load(data_npy)
        df_data = pd.read_csv(data_csv)
        df_tsne = pd.read_csv(tsne_csv)
        #merge data
        df = df_data.merge(df_tsne[['x','y','log_entropy']], left_index=True, right_index=True)
        df = add_subtype_label(df)
        print("============\nBefore filtering\n============")
        for subtype in pd.unique(df['subtype']):
            print(f'\n{subtype}: ')
            print(df.loc[df['subtype']==subtype].describe())
        # filter confident samples
        # df = filter_confident(df, conf_filter_quantile=params.q_filter, conf_col=params.conf_col, mode=params.filter_mode)
        #print stats of columns
        print("============\nAfter filtering\n============")
        for subtype in pd.unique(df['subtype']):
            print(f'\n{subtype}: ')
            print(df.loc[df['subtype']==subtype].describe())

        if params.add_slide_pred != 'none':
            df = add_slide_pred(df, thres=0.5, method = 'mean',col=params.add_slide_pred)

        # print number of tiles for each subtype
        # print(df['subtype'].value_counts())
        for subtype in pd.unique(df['subtype']):
            print(f'{subtype}: ')
            print(pd.unique(df.loc[df['subtype']==subtype]['patient_id']).shape[0])

        if args.redo_tsne:
            tsne_npy =  glob.glob(join(inpath, 'data','epoch*','postnet*.npy'))[0]
            data = np.load(tsne_npy)
            idx = df.index.to_numpy()
            embeddings = data[idx,:]
            if params.add_slide_pred != 'none':
                embeddings = np.concatenate([embeddings,df['OOD_pred'].to_numpy().reshape(-1,1),df['ID_pred'].to_numpy().reshape(-1,1)],axis=1)
            # add 
            embeddings_tsne = None
        else:
            embeddings_tsne = df[['x','y']].to_numpy()
            embeddings = None
        export_tsne_figure(df,'subtype', 'y_score_epistemic', 
                           vis_method=params.vis_method,
                           add_slide_pred=params.add_slide_pred,
                           min_dist=params.min_dist,
                           outpath=outpath,perplexity=params.perplexity, early_exaggeration=params.early_exaggeration,
                           balanced_sample=args.balanced_sample,
                           embeddings=embeddings,embeddings_tsne=embeddings_tsne, seed=42)
        




if __name__ == '__main__':
    args = pargse_args()
    # print(args)
    args_dict = vars(args)
    print('====================')
    print('      Arguments:')
    print('====================')
    for k in args_dict.keys():
        print(f'{k}:\t{args_dict[k]}')
    main(args)




