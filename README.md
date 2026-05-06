# APAU-Net: Adaptive Prior-Aware U-Net for Handwritten Text-Line Segmentation

This repository contains a single Jupyter notebook implementation of **APAU-Net**, a two-stage prior-aware U-Net framework for handwritten text-line segmentation in historical manuscript images.

The notebook file is:

```text
udiads-unet-2-stage.ipynb
```

The notebook includes:

- Stage 1 Gaussian prior generation and training;
- Stage 2 prior-guided text-line refinement;
- full-page tiled inference;
- Pixel IoU, Line IU, Detection Rate (DR), Recognition Accuracy (RA), and F-measure (FM) evaluation;
- checkpoint saving, metric logging, and preview/prediction export.

## Requirements

The notebook was developed in a Kaggle/Jupyter environment using Python and PyTorch.

Install the required packages with:

```bash
pip install torch numpy opencv-python Pillow tqdm jupyter
```

Main dependencies:

```text
torch
numpy
opencv-python
Pillow
tqdm
jupyter
```

## Dataset Structure

The notebook expects manuscript images and binary text-line masks to be stored in separate folders with matching filenames.

Example structure:

```text
UDIADS/
├── images/
│   ├── train/
│   └── val/
└── masks/
    ├── train/
    └── val/
```

Example matching image/mask pair:

```text
UDIADS/images/train/page_001.png
UDIADS/masks/train/page_001.png
```

Masks must be binary images: non-zero pixels represent text-line foreground, and zero pixels represent background.

## Configuration

Before running the notebook, update the path variables in the configuration cells.

Stage 1 paths:

```python
TRAIN_IMG_DIR = "/path/to/UDIADS/images/train"
TRAIN_MSK_DIR = "/path/to/UDIADS/masks/train"
VAL_IMG_DIR   = "/path/to/UDIADS/images/val"
VAL_MSK_DIR   = "/path/to/UDIADS/masks/val"
```

Stage 2 paths:

```python
IMG_TRAIN = "/path/to/UDIADS/images/train"
IMG_VAL = "/path/to/UDIADS/images/val"
TL_MSK_TRAIN = "/path/to/UDIADS/masks/train"
TL_MSK_VAL = "/path/to/UDIADS/masks/val"
PRIOR_DIR_TRAIN = "/path/to/priors/train"
PRIOR_DIR_VAL = "/path/to/priors/val"
```

## Running the Notebook

Open the notebook:

```bash
jupyter notebook udiads-unet-2-stage.ipynb
```

Then run the cells in order.

Recommended workflow:

1. Run the import and configuration cells.
2. Train Stage 1 to learn anisotropic Gaussian prior prediction.
3. Generate or save Stage 1 prior maps for the training and validation sets.
4. Train Stage 2 using RGB images concatenated with the prior maps.
5. Run tiled full-page inference and metric evaluation.

## Stage 1: Gaussian Prior Prediction

Stage 1 trains a U-Net to predict anisotropic Gaussian prior maps from downscaled grayscale manuscript pages.

Main settings:

```python
EPOCHS = 500
BATCH_SIZE = 1
BASE_CHANNELS = 32
LR_INIT = 1e-4
MAX_SIDE = 768
RX_SCALE = 1.8
RY_SCALE = 1.2
ROI_SIGMA = 3.0
```

Stage 1 outputs include:

```text
best_maskprob_gauss.pt
last_maskprob_gauss.pt
metrics_maskprob_gauss.csv
preview_maskprob_gauss/
```

## Stage 2: Text-Line Refinement

Stage 2 trains a residual U-Net using full-resolution RGB patches concatenated with the Stage 1 prior map.

Main settings:

```python
EPOCHS = 500
BATCH_SIZE = 1
BASE = 64
LR_INIT = 1e-4
PATCH = 512
VAL_STRIDE = 256
ALPHA_MIN = 0.15
ALPHA_MAX = 1.50
BETA_GATE = 1.5
```

Stage 2 outputs include:

```text
best_lineiu.pt
last_lineiu.pt
metrics_lineiu.csv
preview/
preds/
```

## Inference Example

The notebook contains inference cells for applying the trained Stage 2 model using tiled prediction.

A typical inference workflow is:

1. Load the trained Stage 1 and Stage 2 checkpoints.
2. Load a manuscript image.
3. Generate the Stage 1 prior map.
4. Concatenate the RGB image and the prior map.
5. Apply Stage 2 using tiled inference.
6. Threshold the final probability map at 0.5.
7. Save the predicted binary text-line mask.

The notebook includes utilities for exporting predictions and zipping the output folder.

## Evaluation Metrics

The notebook reports the following metrics:

- Pixel IoU;
- Line IU;
- Detection Rate (DR);
- Recognition Accuracy (RA);
- F-measure (FM).

Line-level metrics are computed using connected components and a matching threshold of 0.75.

## Reproducibility Notes

The notebook uses a fixed random seed:

```python
SEED = 42
```

Training settings used in the reported experiments include:

- AdamW optimizer;
- initial learning rate of `1e-4`;
- batch size of `1`;
- no data augmentation;
- Stage 1 full-page processing at reduced resolution;
- Stage 2 patch-based refinement using `512 x 512` patches.

Stage 1 processes the full manuscript page at reduced resolution to preserve page-level text-line topology while keeping memory usage feasible on an 11 GB GPU.
