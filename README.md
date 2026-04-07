# Thyroid Segmentation using U-Net

This project focuses on **automatic segmentation of the thyroid gland and thyroid nodules** in ultrasound images using a deep learning approach based on the **U-Net architecture**.

The model is trained and evaluated on the **DDTI (Digital Database of Thyroid Images)** dataset, which contains annotated ultrasound images for thyroid analysis.

---

## Overview

Thyroid nodule detection and segmentation are essential for early diagnosis and treatment planning. Manual annotation is time-consuming and subjective, while deep learning methods provide efficient and consistent results.

U-Net is a widely used architecture for medical image segmentation due to its ability to:

* Capture both **contextual and spatial information**
* Preserve fine details through **skip connections**
* Perform accurate **pixel-wise classification**

---

## Dataset

This project uses the **DDTI dataset**, which includes:

* Ultrasound images of the thyroid
* Corresponding ground truth masks

---

## Model Architecture

The model is based on the **U-Net architecture**, consisting of:

* **Encoder (Contracting Path)**
  Extracts high-level features using convolutional layers

* **Decoder (Expanding Path)**
  Reconstructs spatial resolution for precise segmentation

* **Skip Connections**
  Combine low-level and high-level features to improve localization

---

## Features

* Preprocessing of ultrasound images and masks
* Training pipeline for U-Net model
* Evaluation using segmentation metrics
* Visualization of predictions
* Support for CPU/GPU execution

---

## Evaluation Metrics

The model performance is evaluated using:

* Dice Coefficient
* Accuracy

These metrics are commonly used in medical image segmentation tasks.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/merimakopic/Thyroid-Segmentation-using-U-net.git
cd Thyroid-Segmentation-using-U-net
```

---

### Generate masks

```bash
python generate_masks.py
```

### Training

```bash
python train.py
```

### Testing

```bash
python test.py
```

---

## Project Structure

```
├── data/                # Dataset (images and masks)
├── model/               # U-Net model implementation
├── outputs/             # Predictions and results
├── train.py             # Training script
├── predict.py           # Inference script
├── utils.py             # Helper functions
├── requirements.txt     # Dependencies
└── README.md
```

---

## Results

The model produces segmentation masks highlighting:

* Thyroid gland regions
* Thyroid nodules

Results can be visualized and compared with ground truth masks.

---

## License

This project is for research and educational