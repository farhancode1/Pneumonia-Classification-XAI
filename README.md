# Pneumonia Classification with Explainable AI

Deep learning based pneumonia classification from chest X-ray images using a custom CNN, VGG16, and ResNet50, with **Grad-CAM** and **LIME** for model explainability.

## Overview

This project investigates automated pneumonia detection from chest X-rays while addressing an important limitation of deep learning in healthcare: **interpretability**. Three image-classification models are evaluated and their predictions are explained using two complementary XAI methods.

- **Baseline CNN**
- **VGG16** with transfer learning
- **ResNet50** with transfer learning
- **Grad-CAM** for gradient-based visual explanations
- **LIME** for model-agnostic local explanations

## Dataset

The experiments use the **Chest X-Ray Pneumonia** dataset available on Kaggle, containing two classes:

- `NORMAL`
- `PNEUMONIA`

The provided train, validation, and test splits are used. Images are resized to **224 × 224 pixels**, converted to tensors, and normalized before training.

> The dataset is not included in this repository. Download it separately from Kaggle and place it in your local data directory.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Baseline CNN | **0.8173** | **0.7816** | 0.9821 | **0.8705** |
| VGG16 (Transfer Learning) | 0.7420 | 0.7078 | **1.0000** | 0.8289 |
| ResNet50 (Transfer Learning) | 0.7821 | 0.7471 | 0.9846 | 0.8496 |

The **Baseline CNN achieved the highest F1-score (0.8705)** and was therefore selected as the primary model for explainability analysis.

Its confusion matrix recorded:

- True-positive pneumonia cases: **383**
- False-negative pneumonia cases: **7**

This high recall is particularly useful in a screening setting, where missing a pneumonia case can be more consequential than generating an additional false positive.

## Explainable AI

### Grad-CAM

Grad-CAM highlights image regions that contribute strongly to a model prediction by using gradient-weighted activation maps from convolutional layers.

In this project it is applied to the Baseline CNN, VGG16, and ResNet50 to visualize the spatial regions influencing pneumonia predictions.

### LIME

LIME explains an individual prediction by perturbing local image regions (superpixels) and learning a simple surrogate model around that prediction.

Because LIME is model-agnostic, it provides a useful complementary explanation to Grad-CAM.

## Grad-CAM vs LIME

| Method | Main idea | Strength |
|---|---|---|
| Grad-CAM | Uses gradients and convolutional feature maps | Smooth spatial localization |
| LIME | Perturbs superpixels and observes prediction changes | Model-agnostic, local explanation |

Using both methods provides two different views of model behavior and helps make the classification process more transparent.

## Installation

Clone the repository:

```bash
git clone https://github.com/farhancode1/Pneumonia-Classification-XAI.git
cd Pneumonia-Classification-XAI
```

Create and activate a virtual environment if desired, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

The original experiments were conducted in **Python using Google Colab**.

Open the project notebook, configure the path to the downloaded dataset, and run the cells in order to:

1. Load and preprocess the chest X-ray dataset.
2. Train/evaluate the Baseline CNN.
3. Train/evaluate VGG16 and ResNet50 transfer-learning models.
4. Compare accuracy, precision, recall, F1-score, and confusion matrices.
5. Generate Grad-CAM explanations.
6. Generate LIME explanations.

Original Colab notebook:

https://colab.research.google.com/drive/1J-oCdqcfJzkDjc-6vnBermkmMCJVGRgh?usp=sharing

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

F1-score is emphasized because the dataset is class-imbalanced and the metric balances precision and recall.

## Ethical Considerations and Limitations

Medical AI systems can produce false negatives and false positives, and performance may vary across patient groups or imaging conditions. Explainability can improve transparency, but it does not remove model uncertainty or replace clinical judgment.

This repository is intended for **research and educational purposes only** and is **not a medical diagnostic tool**.

## Tech Stack

- Python
- PyTorch / Torchvision
- NumPy
- Matplotlib
- scikit-learn
- LIME
- Pillow
- OpenCV

## Author

**Mohd Farhan**

## License

No license has been added yet. If you plan to reuse or distribute this project publicly, add an appropriate open-source license.
