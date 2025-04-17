# UNet for Semantic Segmentation on Pascal VOC 2012 (Mini Version)

## Overview
This project implements a **UNet** model for **semantic segmentation** using the **Pascal VOC 2012 (Mini Version)** dataset. The goal is to train a model that learns to identify and segment various object classes in natural images, such as people, animals, and vehicles.

## Background
**What is UNet**
**UNet** is a widely used architecture for semantic segmentation, originally developed for biomedical image segmentation. Its hallmark design involves:
    **Encoder (contracting path)**: captures high-level features through convolution and max-pooling.
    **Decoder (expanding path)**: uses transposed convolutions and skip connections to restore spatial resolution and produce dense predictions.

This structure allows the model to combine semantic context with spatial precision, making it ideal for pixel-wise tasks.

## Dataset: Pascal VOC 2012 (mini version)
**Description**
The **Pascal VOC 2012 **dataset is a benchmark in the computer vision community. The mini version is a smaller subset used for faster experimentation while retaining the original structure and class annotations.
    **Image types**: natural images (animals, humans, indoor/outdoor objects)
    **Segmentation masks**: pixel-wise class labels for each object
    **Total classes**: 20 foreground object classes + 1 background

**Directory Structure (VOC Format)**
```
VOCdevkit/
└── VOC2012/
    ├── JPEGImages/         # RGB images
    ├── SegmentationClass/  # Ground truth segmentation masks (color images)
    ├── ImageSets/
    │   └── Segmentation/
    │       ├── train.txt
    │       ├── val.txt
    └── Annotations/        # (not used in this project)
```

## Project Objectives
**Implement UNet** for multi-class semantic segmentation.
**Preprocess** VOC data (resizing, label encoding).
**Train** UNet on the mini Pascal VOC 2012 dataset.
**Visualize** predictions and assess segmentation quality.

## Project Structure
```
├── data/                         # Pascal VOC 2012 Mini dataset (VOC format)
├── models/
│   └── unet.py                   # UNet architecture implementation
├── train.py                      # Training and validation loop
├── utils.py                      # Data transforms, label maps, metrics
├── predict.py                    # Inference script to test on new images
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation

```

## Installation
You can clone this project on local machine

## Training the Model
```
python train.py --epochs 50 --batch_size 8 --lr 0.001
```
## Evaluation Metrics
**Pixel Accuracy**
**Mean IoU (Intersection over Union)**
**Class-wise IoU**
**Dice Coefficient**

## Sample Results
Here are some qualitative results showing how the UNet performs on the Pascal VOC Mini validation set:

## References