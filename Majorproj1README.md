# Image Classification: ANN vs CNN

This major project compares a traditional Artificial Neural Network (ANN) with a Convolutional Neural Network (CNN) for natural-scene image classification.

## Dataset

Recommended dataset: **Intel Image Classification** from Kaggle.

The dataset contains six categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

Download the dataset from Kaggle and extract it locally.

## Dataset Structure

The code supports this common structure:

```text
intel-image-classification/
├── seg_train/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
└── seg_test/
    └── seg_test/
        ├── buildings/
        ├── forest/
        ├── glacier/
        ├── mountain/
        ├── sea/
        └── street/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py --data_dir "path/to/intel-image-classification" --epochs 10
```

Example for Windows:

```bash
python main.py --data_dir "D:/datasets/intel-image-classification" --epochs 10
```

## What the Project Does

1. Loads images using `torchvision.datasets.ImageFolder`.
2. Resizes images to `150 x 150`.
3. Converts images to tensors.
4. Normalizes pixel values.
5. Trains a baseline ANN using dense layers.
6. Trains a CNN using:
   - Convolutional layers
   - ReLU activation
   - Max-pooling
   - Dropout
7. Compares:
   - Training time
   - Validation accuracy
   - Trainable parameters
8. Saves charts, model weights, and comparison results in `outputs/`.

## Main Files

```text
main.py
requirements.txt
README.md
image_classification_ann_vs_cnn.ipynb
.gitignore
```

## Output Files

```text
outputs/
├── ann_accuracy.png
├── ann_loss.png
├── cnn_accuracy.png
├── cnn_loss.png
├── ann_vs_cnn_comparison.csv
├── ann_weights.pth
├── cnn_weights.pth
└── class_names.json
```

## Why CNN is Suitable for Images

An ANN converts the complete image into a long vector and treats every pixel independently. This creates many parameters and removes the spatial relationship between nearby pixels.

A CNN uses filters that scan local regions of an image. It learns useful visual patterns such as edges, textures, shapes, and object parts. Pooling reduces the spatial size, while dropout helps reduce overfitting.

Therefore, CNNs are generally more suitable for image data because they preserve local spatial information and use parameters more efficiently.

## CUDA and TensorRT Bonus

If a CUDA-enabled GPU is available, PyTorch automatically uses it. GPU acceleration can reduce training and inference time.

For deployment, optimized inference tools such as TensorRT may further improve CNN inference speed through graph optimization, kernel selection, and lower-precision computation.
