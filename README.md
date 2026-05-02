# Practical Deep Learning

Hands-on implementation of the [fast.ai](https://www.fast.ai/) curriculum. This repository documents my journey through deep learning, featuring custom training loops, data augmentation strategies, and experiments with various architectures (CNNs, Transformers, and LSTMs) to optimize performance across different datasets.

## 🚀 Projects Overview

The repository is organized into lessons, each focusing on specific deep learning concepts and practical implementations.

### [Lesson 1: NutrientScan](./Lesson_1)
**Focus:** Computer Vision, Transfer Learning, and Classification.
- **Project:** `Excercise_1_NutrientScan.py`
- **Description:** A binary classifier designed to distinguish between "Good" and "Bad" nutrient samples.
- **Key Techniques:**
    - Transfer learning using **ResNet18**.
    - Fastai's `DataBlock` API for efficient data pipeline construction.
    - Model persistence (exporting/loading `.pkl` files) for inference.
    - Automated GPU detection and setup.

### [Lesson 2: Can Classifier](./Lesson_2)
**Focus:** Web Scraping, Data Augmentation, and Model Interpretation.
- **Project:** `lesson_2_canClassifier.py`
- **Description:** A multi-class classifier to categorize different types of cans (Glass, Plastic, Aluminium, Paper).
- **Key Techniques:**
    - Automated image dataset creation using DuckDuckGo search (`ddgs`).
    - Data cleaning with `verify_images`.
    - Advanced data augmentation (Random Resized Crop).
    - Model interpretation using **Confusion Matrices** and **Top Losses** analysis.

### [Lesson 3: MNIST from Scratch](./Lesson_3)
**Focus:** Deep Learning Foundations, Tensors, and Optimization.
- **Project:** `lesson_3_Modeltrainin.py`
- **Description:** Implementing a digit classifier for the MNIST dataset focusing on the fundamental mechanics of neural networks.
- **Key Techniques:**
    - Tensor manipulation and normalization.
    - Calculation of mean images ("Ideal 3" vs "Ideal 7").
    - Loss functions from scratch (L1, L2, and Custom Sigmoid Binary Cross Entropy).
    - Building a custom training loop with SGD (Stochastic Gradient Descent).
    - Understanding broadcasting and matrix multiplication in PyTorch.

## 🛠️ Requirements & Installation

This project uses the `fastai` library and `PyTorch`.

```bash
# Install fastai
pip install fastai

# Install other dependencies
pip install duckduckgo_search fastcore
```

## 💻 Usage

Each lesson folder contains its own dataset (zipped) and Python script.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MH165/Practical-Deep-Learning.git
   cd Practical-Deep-Learning
   ```

2. **Run a lesson:**
   ```bash
   # Example for Lesson 1
   python Lesson_1/Excercise_1_NutrientScan.py
   ```

## 📊 Results & Observations
- **Lesson 1:** Achieved high accuracy on binary classification with ResNet18 fine-tuning.
- **Lesson 2:** Explored the impact of data quality and the power of interpreting model failures through top losses.
- **Lesson 3:** Deepened understanding of backpropagation and loss gradients by implementing them without high-level abstractions.

---
*Maintained by [MH165](https://github.com/MH165)*
