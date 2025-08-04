# Necrosis Score Prediction and Nuclei Segmentation

This repository contains tools for predicting necrosis scores in pathological images and segmenting cell nuclei. It leverages deep learning models, including a provided PyTorch model for necrosis score prediction, to analyze and process histopathological images effectively.

## Contents

- [Necrosis_Infer.py](Necrosis_Infer.py) - Script for predicting necrosis scores using a pre-trained model.
- [Nuc_Seg.py](Nuc_Seg.py) - Script for performing nuclei segmentation on histopathological images.
- [necrosis_annotation_binary.pth](necrosis_annotation_binary.pth) - Pre-trained PyTorch weights for necrosis score prediction.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before running the scripts, ensure you have the following software installed:

- Python 3.6 or higher
- PyTorch 1.7 or higher
- torchvision
- PIL (Pillow)
- NumPy
- Matplotlib (optional, for visualization)
- Other dependencies listed in `requirements.txt`


#### Usages

python Necrosis_Infer.py --anno_path path/to/your/annotation/file
python Nuc_Seg.py
