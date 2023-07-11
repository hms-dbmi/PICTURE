import pandas as pd
import numpy as np
import os
import random
from glob import glob
import h5py
from ctran import ctranspath
import pandas as pd
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import os.path
import tables
import json
import timm
import openslide




def extract_features (net,loader,device,extractor='ssl'):
    all_feats = []
    all_labels = []
    all_paths = []
    with torch.no_grad():
        for batch in loader:
            input = batch[0].to(device)
            if extractor == 'ssl':
                features = net(input)
            else:
                features = net.forward_features(input)

            features = features.cpu().numpy()
            
            all_feats.append(features)
            all_labels.extend(batch[1])
            all_paths.extend(batch[2])

            
    return np.vstack(all_feats),all_labels,all_paths

class FolderData (Dataset):
    def __init__ (self,folder_dir,google=False):
        self.folder_dir = folder_dir
        self.slide_id = folder_dir.split('/')[-1].split('_')[0]
        self.google = google
        self.val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.image_list = glob(f'{folder_dir}/*')
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        raw_img = Image.open(self.image_list[idx]).convert('RGB')
        
        img_tensor = self.val_transform(raw_img)
        
        label = self.image_list[idx].split('/')[-1]
        
        if self.google:
            return img_tensor, self.slide_id+"_"+label,self.image_list[idx]
        else:
            return img_tensor,self.slide_id,self.image_list[idx]


web_lym_dataset = FolderData(folder_dir='',google=True)
web_lym_loader = torch.utils.data.DataLoader(web_lym_dataset, batch_size=1, shuffle=False)



web_gbm_dataset = FolderData(folder_dir='',google=True)
web_gbm_loader = torch.utils.data.DataLoader(web_gbm_dataset, batch_size=1, shuffle=False)

web_outputs_lym = extract_features(model,web_lym_loader,device,extractor=backbone)
web_outputs_gbm = extract_features(model,web_gbm_loader,device,extractor=backbone)


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x
    

    

def auc_bootstrap(y_true, y_pred, n_bootstraps, seed=None):
    # With help from https://stackoverflow.com/a/19132400/5666087
    bootstrapped_aucs = np.empty(n_bootstraps)
    prng = np.random.RandomState(seed)
    for i in range(n_bootstraps):
        indices = prng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        bootstrapped_aucs[i] = metrics.roc_auc_score(
            y_true[indices], y_pred[indices])
        print(f"{round((i + 1) / n_bootstraps * 100, 2)} % completed bootstrapping", end="\r")
    print()
    bootstrapped_aucs.sort()
    return bootstrapped_aucs

def plot_roc(y_true, y_score, positive_class, n_bootstraps=10000, seed=None):
    fpr, tpr, _ = metrics.roc_curve(y_true=y_true, y_score=y_score)
    
    aucs = auc_bootstrap(y_true, y_score, n_bootstraps=n_bootstraps, seed=seed)
    roc_auc = aucs.mean()
    confidence_95 = aucs[int(0.025 * aucs.shape[0])], aucs[int(0.975 * aucs.shape[0])]

    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='black', lw=lw)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC for GBM vs PCNSL ({positive_class} is positive class)')
    
    print(f"ROC curve (area = {roc_auc:0.02f}")
    print(f"95% CI = {confidence_95[0]:0.2f} - {confidence_95[1]:0.2f}")
    print(n_bootstraps, "bootstraps")
    
    return fig, roc_auc, confidence_95




if __name__ == '__main__':

    backbone = 'ssl'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if backbone == 'swin':
        linear_model = LinearClassifier(input_dim=1024)
    elif backbone == 'ssl':
        linear_model = LinearClassifier(input_dim=768)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.01)

    linear_model.to(device)
    train_all_loss = []
    for epoch in range(1000):
        logps = linear_model(web_train_feats)
        loss = criterion(logps, web_train_labels)
        train_all_loss.append(loss.item())
        loss.backward()
        
        probs = torch.nn.functional.softmax(logps, dim=1)

        # Running count of correctly identified classes
        ps = torch.exp(logps)
        _, predictions = ps.topk(1, dim=1)  # top predictions

        equals = predictions == web_train_labels.view(*predictions.shape)

        all_hits = equals.view(equals.shape[0]).tolist()  # get all T/F indices
        all_corrects = web_train_labels[all_hits]
        print(np.count_nonzero(all_hits)/len(all_hits))
        optimizer.step()
        optimizer.zero_grad()
