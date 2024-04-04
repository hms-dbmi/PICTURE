# Pathology Imaging Characterization with Uncertainty-aware Rapid Evaluation (PICTURE)
## Uncertainty-Aware Deep Learning Differentiates Glioblastoma from its Pathology Mimics – A Multi-Center Study
#### Abstract

Accurate pathological diagnosis is crucial in guiding optimal and personalized treatments for patients with central nervous system (CNS) cancers. Distinguishing several common types of CNS tumors, such as glioblastoma and primary central nervous system lymphoma (PCNSL), can be particularly challenging due to their overlapping histopathology features. Because these cancers require different treatments, fast and accurate pathological evaluation will improve patients’ clinical outcomes. we collected digital images of 4,207 pathology slides from multiple hospitals worldwide and developed the Pathology Image Characterization Tool with Uncertainty-aware Rapid Evaluation (PICTURE) system. 


![PICTURE-Figure1-01](https://github.com/hms-dbmi/PICTURE/assets/31292151/1391afd3-47dc-4129-8e87-b8d90d381cd4)

#### Installation

```console
conda create -n PICTURE python=3.10 -y
conda activate PICTURE
pip install --upgrade pip 
pip install -r requirements.txt
```

#### Toy Data
1. TCGA provides publicly available tissue slides for PCNSL (TCGA-DLBC) and Gliomblastoma (TCGA-GBM). [Note: One could include IDH-wildtype from TCGA-LGG, according to 2021 WHO guidelines.]
https://portal.gdc.cancer.gov/projects/TCGA-DLBC
https://portal.gdc.cancer.gov/projects/TCGA-GBM

2. the Medical University in Vienna provides an online portal, where researchers are welcome to download both PCNSL, Gliomblastoma and other CNS tumors (out-of-distribution):
https://www.ebrains.eu/tools/human-brain-atlas


