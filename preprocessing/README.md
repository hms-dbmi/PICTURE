# Processing Utils

Follow this guide to use our processing scripts.

**Basics:** Create a new conda environment and install the requirements

For installing Openslide:

<pre class="c-mrkdwn__pre" data-stringify-type="pre"><div class="p-rich_text_block--no-overflow"># installation by conda
conda install conda-forge::openslide
conda install conda-forge::openslide-python

# if above failed, just try
conda install conda-forge/label/cf202003::openslide</div></pre>

### Supported WSI Formats

Currently the pipeline fully supports ` .svs, .ndpi, .mrxs` (and partially support `.tiff` format)

##### NOTE on tiff format:

**If you are using WSI in tiff format (for example, tiff files converted from the Phillips .isyntax format). you need to add the following arguments when running `create_tiles.py` and `create_features.py:`**

```
--img_format=tiff --tiff_mag [magnification_for_tiff_file]
```

## Create Tiles

Example slurm script:

```bash
sbatch slurm_tiles_batch.sh
```

## Create Features

If you intend to use the `UNI` model, you will need to request access to the model parameters first ([here](https://huggingface.co/MahmoodLab/UNI)). Then add your `hf_token` from [here](https://huggingface.co/settings/tokens) to the `slurm_features.sh` script.

### CTransPath and Chief

Unfortunately, the implementation of ctranspath and chief requires a different `timm` library version than the rest of our code base. Therefore you will have to first download the `timm-0.5.4.tar` file [from here](https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view?usp=sharing) into the root of the repo. Then run `install_oldtimm.sh` with your activated environment. This will install the `timm-0.5.4` version as `oldtimm`.

Unfortunately, the implementation of ctranspath and chief requires a different `timm` library version than the rest of our code base. Therefore you will have to first download the `timm-0.5.4.tar` file [from here](https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view?usp=sharing) into the root of the repo. Then run `install_oldtimm.sh` with your activated environment. This will install the `timm-0.5.4` version as `oldtimm`.

If you are not on O2, you will also need to change the file paths to the model checkpoints in `slurm_features.sh`.

```bash
# example
export CHIEF_PTH="/n/data2/hms/dbmi/kyu/lab/che099/models/chief.pth"
export CTRANS_PTH="/n/data2/hms/dbmi/kyu/lab/che099/models/ctranspath.pth"
```

### Gigapath (WSI-level)

Install the dependencies here: [https://huggingface.co/prov-gigapath/prov-gigapath]()

### Other models

The other models are downloaded on the fly in case you want to use them. Currently we support the following models. We may add more support over time.

- chief
- ctrans
- phikon
- swin224
- resnet50
- lunit
- uni (requires `hf_token`)
- prov-gigapath (requires `hf_token`)
- cigar
- virchov (requires `hf_token`)

Example slurm script:

```bash
sbatch slurm_features_batch.sh
```
