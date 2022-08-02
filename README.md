# ML models for aorta diagnostics

This is masters thesis containing different experiments and models code.

ImagePreparations - contains code for preparing dataset from CT files  
MeasuringAortaDiameter - contains code (with examples) of measuring sizes of aorta

learning_scripts - contains learning files for machine learning models  
TransUnetCode - folder for code using TransUnet model REQUIRES CLONE OF TransUnet repo inside

## Usage

### 1. Download Google pre-trained ViT models (required only for TransUnet)
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...

### 2. Environment

Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.

### 3. Prepare data

Please run ["ImagePreparations.ipynb"](ImagePreparations.ipynb) with setting the path for dataset containing .nii.gz files.  
Output folder shall have following structure, without the validation folder:

```
folder
│───images
│   | [nii.gz filename]_1.jpg
│   │ [nii.gz filename]_2.jpg
|   | ...
│   masks
│   | [nii.gz filename]_1.pickle
│   │ [nii.gz filename]_2.pickle
|   | ...
│
└───validation
│   │───images
│   |   | [nii.gz filename]_1.jpg
│   │   | [nii.gz filename]_2.jpg
|   |   | ...
│   │   masks
|   │   | [nii.gz filename]_1.pickle
|   │   │ [nii.gz filename]_2.pickle
|   |   | ...
```

Validation dataset can be created using code in same notebook. Code selects files with names from list and  moves to designated directory.

### 4. Train

- Run the train script on dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
python .\TransUnet.py --dataset_dir [dataset path] --pretrained_path [Vit path] --vit_name [vit name]
```