# Pathology Imaging Characterization with Uncertainty-aware Rapid Evaluation (PICTURE)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>


## Uncertainty-Aware Deep Learning Differentiates Glioblastoma from its Pathology Mimics – A Multi-Center Study
#### Abstract

Accurate pathological diagnosis is crucial in guiding optimal and personalized treatments for patients with central nervous system (CNS) cancers. Distinguishing several common types of CNS tumors, such as glioblastoma and primary central nervous system lymphoma (PCNSL), can be particularly challenging due to their overlapping histopathology features. Because these cancers require different treatments, fast and accurate pathological evaluation will improve patients’ clinical outcomes. we collected digital images of 4,207 pathology slides from multiple hospitals worldwide and developed the Pathology Image Characterization Tool with Uncertainty-aware Rapid Evaluation (PICTURE) system. 


![PICTURE-Figure1-01](https://github.com/hms-dbmi/PICTURE/assets/31292151/1391afd3-47dc-4129-8e87-b8d90d381cd4)

#### Installation

```console
conda create -n PICTURE python=3.10 -f environment.yml -y
conda activate PICTURE
pip install --upgrade pip 
```

#### Suggested System Requirements (Linux-based high performance computing (HPC) platform at Harvard Medical School)
Linux: Ubuntu 20.04 LTS
CUDA: 12.1
Nvidia GPU. (All experiemnts were conducted using Nvidia A100. However, the inference should be able to use any CUDA supported GPU.)

#### Publicly Available Datasets
1. TCGA provides publicly available tissue slides for PCNSL (TCGA-DLBC) and Gliomblastoma (TCGA-GBM). [Note: One could include IDH-wildtype from TCGA-LGG, according to 2021 WHO guidelines.]
https://portal.gdc.cancer.gov/projects/TCGA-DLBC
https://portal.gdc.cancer.gov/projects/TCGA-GBM

2. the Medical University in Vienna provides an online portal, where researchers are welcome to download both PCNSL, Gliomblastoma and other CNS tumors (out-of-distribution):
https://www.ebrains.eu/tools/human-brain-atlas

#### Trained Model Weights.
Simply use our trained model for differentiating Glioblastoma from others (e.g., PCNSL, OOD). 
```console
python main_exp.py
```

#### Preprocessing (Tiling)
```console
python preprocessing/WSI_tile_extraction.py
```

#### Cell Quantification
See the ReadMe in cell_quantification

#### Heatmap Visualization 

```console
python Heatmap_Vis/generate.py --region' $x_s $y_s $x_e $y_e '--label '$label' --column '$col' --slide-path '$s_path' --model-path '$m_path
```
#### Uncerntaity 

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from the command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

## Reproducibility of uncertainty results

<!-- You can download the weights on [huggingface](https://huggingface.co/raphaelattias/yulab-uncertainty-posterior/blob/main/epoch_031.ckpt). -->
The weights are stored in :
```bash
best_ckpts/
```

In order to perform the hyper parameter sweep which we used to obtain the final model:
```bash
wandb sweep sweep_yamls/sweepCV_vienna_CTransFeature_fold[FOLD].yaml
```
This will return the bash command in order to run the sweep, for example:
```bash
wandb agent sylin/uncertainty_vienna_CTransFeature_wMoreBenign_fold[FOLD]/nqabs50g
```

In order to directly train using the best hyperparameters we found:
```bash
python src/train.py experiment=best_uncertainty_vienna_fold[FOLD].yaml
```

Slide-level AUC using confident tiles can be estimated using:
```bash
python AUC_analysis.py --files "path/to/fold1_prediction.csv" "path/to/fold2_prediction.csv" ... "path/to/fold10_prediction.csv"
```

UMAP visualization can be obtained with:
```bash
python script_visualize.py --fold [FOLD]
```

In order to reproduce the results and validate the model, please run:

```bash 
python script_CTrans_feature.py --checkpoint_path="path/to/checkpoint.ckpt"
```





