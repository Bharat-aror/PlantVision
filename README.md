# PlantVision

A plant disease classification project built on the PlantVillage dataset, comparing multiple modelling approaches, from a lightweight MLP through to a tuned stacking ensemble, a fine-tuned CNN with GradCAM interpretability, and unsupervised clustering, alongside a benchmark against Azure Custom Vision.

## Overview

PlantVision classifies leaf images across 7 plant disease categories (Tomato Healthy, Tomato Late Blight, Corn Common Rust, Corn Healthy, Apple Scab, Potato Early Blight, Grape Black Rot) using features extracted from a MobileNetV2 backbone pretrained on ImageNet. The project is structured as four tasks, each exploring a different modelling angle on the same feature space.

## Approach

**Data preparation**
- Images loaded from the PlantVillage dataset (via KaggleHub), resized to 128x128, and normalised
- Class imbalance identified and addressed using `class_weight='balanced'` rather than oversampling, to avoid introducing synthetic noise
- Stratified train/validation/test split (70/15/15)
- Augmentation applied during training (flips, rotation, zoom, brightness, shifts)

**Task A — MLP on extracted features**
- MobileNetV2 used as a frozen feature extractor (GlobalAveragePooling2D output)
- A dense MLP classifier trained on the extracted features
- Interpreted using SHAP (`KernelExplainer`) to identify influential features

**Task B — Optuna-tuned stacking ensemble**
- Random Forest hyperparameters tuned with Optuna (15 trials, maximising validation accuracy)
- Stacking ensemble combining Random Forest, Extra Trees, and Histogram Gradient Boosting as base learners, with Logistic Regression as the meta-learner
- Interpreted using SHAP (`TreeExplainer`) on the Random Forest component

**Task C — End-to-end fine-tuned CNN**
- MobileNetV2 base combined with dense classification layers, trained end-to-end
- GradCAM implemented to visualise which regions of each leaf image drove the model's predictions
- A dropout rate sweep (0.1 / 0.25 / 0.4) run to evaluate regularisation trade-offs
- Benchmarked against Azure Custom Vision as an external comparison point

**Task D — Unsupervised clustering**
- PCA used for dimensionality reduction on extracted features
- K-Means, Gaussian Mixture Models, Agglomerative Clustering, and HDBSCAN compared
- Clustering quality evaluated using Adjusted Rand Index, Normalised Mutual Information, and Silhouette Score against the true disease labels

## Tech Stack

- Python, TensorFlow / Keras (MobileNetV2, CNN)
- scikit-learn (Stacking Ensemble, clustering, PCA)
- Optuna (hyperparameter optimisation)
- SHAP (model interpretability)
- HDBSCAN
- OpenCV (GradCAM overlay)
- Azure Custom Vision (external benchmark)

## Dataset

[PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset), filtered to 7 classes with up to 1,500 images per class.

## Results

The CNN and stacking ensemble approaches achieved the strongest performance, with the best configuration reaching 99.43% test accuracy. Full per-task accuracy comparisons, confusion matrices, and SHAP plots are generated within the notebook.

## Notes

- API keys (Kaggle, Azure Custom Vision) are referenced via placeholder variables in the notebook and must be supplied separately as environment variables, not committed to the repository
- The notebook is structured to run sequentially in Google Colab, with intermediate artefacts (processed arrays, trained models, results) saved to disk between sections

## Author

Bharat Arora
[LinkedIn](http://www.linkedin.com/in/bharat-arora-2a13772b9)
