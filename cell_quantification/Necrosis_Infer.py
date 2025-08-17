import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import joblib
import json
from ctran import ctranspath

def load_model(device):
    """
    Loads and prepares the ctranspath model and the linear classifier.

    Parameters:
    - device: The device to run the models on ('cpu' or 'cuda').

    Returns:
    - tuple: A tuple containing the ctranspath model and the linear classifier model.
    """
    # Load ctranspath model
    model = ctranspath()
    model.head = nn.Identity()
    model_state = torch.load('./pretrained/ctranspath.pth', map_location=device)
    model.load_state_dict(model_state['model'], strict=True)
    model.to(device)
    model.eval()

    # Define and load the linear classifier
    linear_model = LinearClassifier(input_dim=768, output_dim=2)
    model_path = 'necrosis_annotation_binary.pth'
    state_dict = torch.load(model_path, map_location=device)
    linear_model.load_state_dict(state_dict)
    linear_model.to(device)
    linear_model.eval()

    return model, linear_model

class LinearClassifier(nn.Module):
    """
    A simple linear classifier.
    """
    def __init__(self, input_dim=768, output_dim=2):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def predict_necrosis_score(model, linear_model, file_list, device, transform):
    """
    Predicts the necrosis score for images in a given list.

    Parameters:
    - model: The ctranspath model for feature extraction.
    - linear_model: The linear classifier model.
    - file_list: List of image file paths.
    - device: The device to run the models on ('cpu' or 'cuda').
    - transform: The torchvision transforms to apply to the images.

    Returns:
    - dict: A dictionary of image paths and their corresponding necrosis scores.
    """
    results = {}
    for idx, ele in enumerate(file_list):
        img_path = ele[0]
        raw_img = Image.open(img_path).convert('RGB')
        img_tensor = transform(raw_img)
        if idx % 1000 == 0:
            print(idx)
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(device)
            features = model(img_tensor)
            logps = linear_model(features)
            probs = F.softmax(logps, dim=1)
            necrosis_score = probs[0][1].item()
            results[img_path] = necrosis_score
    return results

def main(anno_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, linear_model = load_model(device)
    print('Model Loaded')

    # Image preprocessing transformations
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load file maps for L and G
    L_map = 'placeholder_L_map_path'
    L_lst = joblib.load(L_map)
    G_map = 'placeholder_G_map_path'
    G_lst = joblib.load(G_map)

    # Predict necrosis scores
    L_results = predict_necrosis_score(model, linear_model, L_lst, device, val_transform)
    G_results = predict_necrosis_score(model, linear_model, G_lst, device, val_transform)

    # Save results to JSON
    results_dir = 'placeholder_results_directory'
    with open(f'{results_dir}/G_sel_MORE.json', 'w') as f:
        json.dump(G_results, f, indent=4)
    # Similar saving can be done for L_results if needed

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Script for necrosis score prediction.')
    parser.add_argument('--anno_path', required=True, help='Path to the annotation file.')

    args = parser.parse_args()
    main(args.anno_path)
    print('Feature extraction completed.')
