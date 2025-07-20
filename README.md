# Caltech-101 Image Classification with DenseNet-121 and Hyperparameter Tuning

This project implements an image classification pipeline on the Caltech-101 dataset using transfer learning with **DenseNet-121**. The goal is to identify the best model configuration through **comprehensive hyperparameter tuning** and **data augmentation**, achieving state-of-the-art performance on a challenging multi-class dataset.

---

## 📂 Dataset

- **Dataset:** [Caltech-101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)
- **Categories:** 101 object classes
- **Image Size:** Resized to 224×224 pixels
- **Preprocessing:**
  - Normalization
  - Train-validation split
  - Data augmentation (rotation, flipping, zoom)

---

## 🧠 Model Overview

- **Base Model:** DenseNet-121 (pretrained on ImageNet)
- **Custom Head:**
  - Global Average Pooling
  - Dense Layer with ReLU or ELU
  - Dropout
  - Final Dense Layer (softmax activation)
- **Training Strategy:**
  - Fine-tuning all layers (not just the head)
  - Early stopping and model checkpointing

---

## 🔍 Hyperparameter Grid Search

Conducted a full grid search over the following parameters:

| Parameter        | Values                            |
|------------------|------------------------------------|
| Learning Rate     | `0.01`, `0.001`, `0.0001`          |
| Dropout Rate      | `0.2`, `0.3`                       |
| Dense Units       | `512`, `128`                      |
| Activation        | `relu`, `elu`                     |
| Optimizer         | `adam`, `adagrad`, `rmsprop`      |
| Epochs            | `50` (initial), `200` (best model)|

Each configuration was saved along with its training metrics and evaluation plots.

---

## ✅ Best Model

- **Configuration:** `adagrad_lr0.01_drop0.3_units512_relu`
- **Validation Accuracy:** 95.10%
- **Validation Loss:** 0.1915
- **Training Epochs:** 200
- **Model Saved At:** `models/adagrad_lr0.01_drop0.3_units512_relu.h5`

---

## 📁 Main Code Files

### A. `generate_configs.py` – 🔧 Configuration Generator
- Automatically creates JSON configuration files for all combinations of:
  - Learning rate, dropout rate, dense layer size, optimizer, activation function, and number of epochs.
- Outputs saved in the `experiment_configs/` directory.
- **Purpose:** Enables reproducible, modular experimentation.

> 📂 Example output: `experiment_configs/adagrad_lr0.01_drop0.3_units512_relu.json`

---

### B. `generate_experiment_runner.py` – 🧪 Notebook Generator
- Dynamically generates a full Jupyter notebook (`experiment_runner.ipynb`) that:
  - Loads all experiment configs
  - Preprocesses data
  - Builds and trains models using each configuration
  - Applies early stopping and checkpoint saving
  - Saves training plots and validation metrics
- Avoids re-running experiments if model checkpoint already exists.

> ✅ Supports headless batch-style training in a reproducible notebook format.

---

### C. `experiment_runner.ipynb` – 📓 Auto-Generated Training Notebook
- Created by `generate_experiment_runner.py`
- Executes all experiments defined in the `experiment_configs/` folder
- Saves:
  - Trained models to `/models`
  - Training plots to `/plots`
  - Final metrics (val accuracy, loss, epochs) to `results/experiment_results.csv`

---

### D. `colab_experiment_script.ipynb` – 🧪 Google Colab Notebook
- Lightweight Colab version for evaluating or rerunning specific experiments
- Mounts Google Drive and sets working directory
- Can be used to:
  - Train a single model using one config
  - Visualize training curves
  - Save and sync results to Google Drive

---

## 📊 Outputs

- Training and validation loss/accuracy plots
- CSV log of all runs and hyperparameter combinations
- Saved best-performing model
- Colab evaluation notebook to:
  - Load the best model
  - Evaluate on validation set
  - Visualize predictions

---

## 📄 Inspiration

This work is inspired by the paper:

> H. Jiang et al., *"AutoTune: A Robust Self-Tuning Algorithm for Image Classification"*, arXiv preprint arXiv:2005.02165, 2020.  
> [[PDF]](https://arxiv.org/pdf/2005.02165.pdf)

---

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy, Matplotlib, Pandas
- tqdm, scikit-learn

