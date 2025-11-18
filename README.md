# AMLS Assignment 2024-25

This repository contains the implementation for the Applied Machine Learning Systems (AMLS) assignment, which involves two distinct machine learning tasks using medical image datasets from MedMNIST.

## Project Overview

This project implements two classification tasks:
- **Task A**: Binary classification of breast tumor ultrasound images (Benign vs. Malignant) using the BreastMNIST dataset
- **Task B**: Multi-class classification of blood cell images into 8 different types using the BloodMNIST dataset

## Project Structure

```
AMLS_Coursework/
├── A/                      # Task A implementation files
├── B/                      # Task B implementation files
├── Datasets/               # Dataset folder (empty for submission)
│   ├── BreastMNIST/        # Will contain BreastMNIST dataset during assessment
│   └── BloodMNIST/         # Will contain BloodMNIST dataset during assessment
├── main.py                 # Main script to run the project
└── README.md              # This file
```

## File Descriptions

### `main.py`
The main entry point for the project. Running this script will execute both Task A and Task B, including data loading, model training, validation, and testing. The script should be executed from the terminal using:
```bash
python main.py
```

### `A/`
Contains all code files related to Task A (Binary Classification):
- Data preprocessing and loading scripts for BreastMNIST
- Model implementation and training code
- Evaluation and testing scripts
- Pre-trained models (if applicable)

### `B/`
Contains all code files related to Task B (Multi-class Classification):
- Data preprocessing and loading scripts for BloodMNIST
- Model implementation and training code
- Evaluation and testing scripts
- Pre-trained models (if applicable)

### `Datasets/`
This folder should remain empty for submission. During assessment, the datasets will be placed in this folder with the following structure:
- `Datasets/BreastMNIST/` - Contains training, validation, and test splits for BreastMNIST
- `Datasets/BloodMNIST/` - Contains training, validation, and test splits for BloodMNIST

## Required Packages

The following Python packages are required to run this project:

### Core Libraries
- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `scipy` - Scientific computing

### Machine Learning
- `scikit-learn` - Machine learning algorithms (SVM, Random Forest, etc.)
- `torch` or `tensorflow` - Deep learning frameworks (if using neural networks)

### Image Processing
- `PIL` (Pillow) - Image processing
- `opencv-python` (optional) - Additional image processing capabilities

### Visualization
- `matplotlib` - Plotting and visualization
- `seaborn` (optional) - Statistical data visualization

### Dataset
- `medmnist` - MedMNIST dataset loader (if using the official package)

### Installation

Install the required packages using pip:

## Dataset Information

### BreastMNIST (Task A)
- **Type**: Binary classification
- **Classes**: Benign (non-cancerous) vs. Malignant (cancerous)
- **Image Size**: 28×28 pixels (grayscale)
- **Splits**:
  - Training: 546 images
  - Validation: 78 images
  - Testing: 156 images
- **Source**: [MedMNIST](https://medmnist.com/)

### BloodMNIST (Task B)
- **Type**: Multi-class classification
- **Classes**: 8 different blood cell types
- **Image Size**: 28×28 pixels
- **Splits**:
  - Training: 11,959 images
  - Validation: 1,715 images
  - Testing: 3,421 images
- **Source**: [MedMNIST](https://medmnist.com/)

## Usage

1. Ensure all required packages are installed (see Required Packages section above).

2. Place the datasets in the `Datasets/` folder:
   - `Datasets/BreastMNIST/` - BreastMNIST dataset files
   - `Datasets/BloodMNIST/` - BloodMNIST dataset files

3. Run the main script:
   ```bash
   python main.py
   ```

The script will:
- Load and preprocess the datasets
- Train models for both Task A and Task B
- Evaluate models on validation sets
- Test models on test sets
- Display or save results (training, validation, and test accuracies/errors)

## Notes

- The `Datasets/` folder should be empty when submitting the code. During assessment, datasets will be placed in this folder by the assessors.
- Pre-trained models can be saved in the respective task folders (`A/` or `B/`) if needed.
- The code is designed to read datasets directly from the `Datasets/` folder structure as it will appear during assessment.

## References

Yang, J., Shi, R., Wei, D. et al. *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Sci Data 10, 41 (2023).
