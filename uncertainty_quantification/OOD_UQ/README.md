
<div align="center">

# Yu Lab's Pathology Uncertainty
An easy-to-use and scalable framework for digital pathology detection and prediction. 

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>


</div>

## Description

This framework provides the necessary tools to train and test deep learning models for pathology detection and prediction. It is built on top of [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://pytorchlightning.ai/), and uses [Hydra](https://hydra.cc/) for configuration management. 
In this library, we provide a set of pre-defined models, data loaders, and metrics for pathology detection and prediction. You will be able to reproduce the results of relevant papers with a few lines of code. 

## How to run

Install dependencies
```bash
    conda create -n yulab -f environment.yml
    conda activate yulab
```

Train the model with the default configuration
```bash
python src/train.py
```
Train on IvyGap dataset with select hyperparameters
```bash
python src/train.py datamodule=ivygap_subsample model.optimizer.lr=0.0001 datamodule.batch_size=16 model.optimizer.weight_decay=0.04
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from the command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

## Reproducibility of papers results

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
## Features

### Models
The features used in this out-of-distribution detection experiment using [CTransPath](https://github.com/Xiyue-Wang/TransPath). You may also incorporate other feature encoders using the same pipeline.

### Datasets
The cancer histopathology slides used in this experiment are from the openly available [Digital Brain Tumor Atlas](https://www.nature.com/articles/s41597-022-01157-0) from the Medical University of Vienna. 33 in-house normal tissues slides from the same institute were also included in this experiment.


