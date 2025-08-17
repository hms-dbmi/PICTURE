import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import random
from torchvision.transforms.functional import to_tensor


def plot_to_image(figure, dpi=300):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi)
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image).transpose(2, 0, 1)


def create_roc_curve(labels: np.array, probs: np.array) -> np.array:
    roc_auc = roc_auc_score(labels, probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    img_array = plot_to_image(fig)
    fig.clf()
    return img_array


def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_expert_util(expert_names, utilizations, title="Expert Utilization"):
    """
    Creates a bar chart for expert utilization.

    Parameters:
    expert_names (list of str): Names of the experts.
    utilizations (np.array): Utilization rates for each expert.
    title (str): Title of the plot.

    Returns:
    torch.Tensor: An image tensor of the plot.
    """
    fig, ax = plt.subplots()
    y_pos = np.arange(len(expert_names))
    ax.barh(y_pos, utilizations, align="center", color="skyblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(expert_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Utilization Rate")
    ax.set_title(title)
    fig.canvas.draw()
    img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close(fig)
    return to_tensor(img)


def compute_expert_variance(weights):
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()
    primary_experts = np.argmax(weights, axis=1)
    expert_counts = np.bincount(primary_experts, minlength=weights.shape[1])
    variance = np.var(expert_counts)
    return variance
