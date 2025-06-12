import argparse
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def export_tsne_figure(y_true, embeddings, log_entropy, seed=42):
    # Visualize the t-sne
    tsne = TSNE(n_components=2, init="pca", random_state=seed, n_jobs=6, learning_rate="auto", perplexity=7)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Create a color map for the two classes using the viridis colormap
    colors = {0: 'red', 1: 'green'}
    labels = {0: 'In-Distribution', 1: 'Out-of-Distribution'}
    class_labels = np.array([labels[t] for t in y_true])
    class_colors = np.array([colors[t] for t in y_true])

    # Compute F1 score for each threshold value
    thresholds = sorted(log_entropy)
    f1_scores = []
    for threshold in thresholds:
        y_pred = (log_entropy >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1_score)

    # Find the threshold that maximizes the F1 score
    max_f1_score = max(f1_scores)
    optimal_threshold = thresholds[f1_scores.index(max_f1_score)]

    # Set up the figure with two subplots
    fig1, ax1 = plt.subplots(figsize=(13,10))
    fig2, ax2 = plt.subplots(figsize=(15,10))

    # Plot the t-SNE embeddings with class labels
    for label in np.unique(class_labels):
        idxs = np.where(class_labels == label)
        ax1.scatter(embeddings_tsne[idxs, 0], embeddings_tsne[idxs, 1], c=class_colors[idxs], label=label, alpha=0.5, s=4, cmap="RdYlGn")

    # Set plot title and labels
    ax1.set_title("Latent Space Embedding")
    ax1.set_xlabel("t-SNE Component 1")
    ax1.set_ylabel("t-SNE Component 2")

    # Add legend
    ax1.legend(loc="upper left")

    # create scaler object
    scaler = MinMaxScaler(feature_range=(0, 1))

    # create a mask that identifies elements above and below the threshold
    mask = log_entropy >= optimal_threshold

    # create a copy of the data array to work with
    scaled_data = np.copy(log_entropy)

    # apply different scaling factors to the elements above and below the threshold
    scaled_data[~mask] = scaler.fit_transform(scaled_data[~mask].reshape(-1, 1)).ravel() * 0.5
    scaled_data[mask] = scaler.fit_transform(scaled_data[mask].reshape(-1, 1)).ravel() * 0.5 + 0.5

    # the resulting 'scaled_data' array will have values between 0 and 1,
    # with values under the threshold mapped to values between 0 and 0.5,
    # and values above the threshold mapped to values between 0.5 and
    # Plot the log entropy values
    im = ax2.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=scaled_data, cmap='RdYlGn', s=4)
    ax2.set_title("Predicted Entropy")
    ax2.set_xlabel("t-SNE Component 1")
    ax2.set_ylabel("t-SNE Component 2")

    # Add color bar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Log Entropy')

    # Save the figures
    fig1.savefig(f"report/figures/latent_space_{seed}.pdf", dpi=600)
    fig2.savefig(f"report/figures/predicted_entropy_{seed}.pdf", dpi=600)

# main 

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--csv", type=str, default="data/postnet_0.csv", help="Path to the true labels")

args = parser.parse_args()

# write main class

import argparse

def main(args):
    # Read the input files
    y_true = pd.read_csv(args.csv)['label']
    embeddings = np.load(args.embeddings_file)
    log_entropy = pd.read_csv(args.csv)['prob1']

    # Call the export_tsne_figure function
    export_tsne_figure(y_true, embeddings, log_entropy, seed=args.seed)

    # Print message 
    print(f"t-SNE figure exported to report/figures/latent_space_{args.seed}.pdf")
    print(f"Predicted entropy figure exported to report/figures/predicted_entropy_{args.seed}.pdf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export t-SNE figure.')
    parser.add_argument('--csv', type=str, help='Path to the csv file containing the true labels.', default="data/postnet_0.csv")
    parser.add_argument('--embeddings_file', type=str, help='Path to embeddings npy file.', default="data/postnet_0.npy")
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator.')
    args = parser.parse_args()
    main(args)
