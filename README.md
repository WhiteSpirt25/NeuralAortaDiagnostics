# ML models for aorta diagnostics

This is masters thesis containing different experiments and models code.
Written as testing pipeline with easily changable seettings and models.

[ImagePreparations](ImagePreparations.ipynb) - contains code for preparing dataset from CT files  
[MeasuringAortaDiameter](MeasuringAortaDiameter.ipynb) - contains code (with examples) of measuring sizes of aorta

models - contains model sprecific functions to be used in pipeline. This includes models itself and treir preprocessing functions.  
[config.json](config.json) - configuration file containing model, training and modes saving parameters.

[model_testing](model_testing.py) - main file with training and testing pipeline.

## Usage

### 1. (required only for TransUnet) Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...

### 2. Environment

Please prepare an environment with python=3.9, and then use the command "pip install -r requirements.txt" for the dependencies.

### 3. Prepare data

Please run ["ImagePreparations.ipynb"](ImagePreparations.ipynb) with setting the path for dataset containing .nii.gz files.  
Output folder shall have following structure, without the validation folder:

```
folder
└───images
│   │ [nii.gz filename]_1.jpg
│   │ [nii.gz filename]_2.jpg
│   │ ...
└───masks
│   │   [nii.gz filename]_1.pickle
│   │   [nii.gz filename]_2.pickle
│   │   ...
│
└───validation
    └───images
    │   │   [nii.gz filename]_1.jpg
    │   │   [nii.gz filename]_2.jpg
    │   │   ...
    └───masks
        │   [nii.gz filename]_1.pickle
        │   [nii.gz filename]_2.pickle
        │   ...
```

Validation dataset can be created using code in same notebook. Code selects files with names from list and  moves to designated directory.

### 4. Train

- (required only for TransUnet) Clone [TransUnet repo](https://github.com/Beckschen/TransUNet) in models folder.

- Check [config.json](config.json) and adjust settings as needed.
- Run training-testing script.

```bash
python model_testing.py --model_name TransUnet
```