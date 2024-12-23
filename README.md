# Environmental Sound Classification Using CNN and LinearSVC

This project explores advanced methods for environmental sound classification using convolutional neural networks (CNNs) and transfer learning techniques. The proposed framework leverages weakly labeled web audio data for sound event detection and classification, demonstrating efficiency with variable-length audio samples and achieving state-of-the-art performance on the ESC-50 dataset. 

To further enhance model robustness and generalization, data augmentation techniques such as pitch shifting, time stretching, and noise addition were applied, simulating real-world variations in environmental sound data. A Linear Support Vector Machine (LinearSVC) is employed as the classification method to effectively distinguish between environmental sound categories under diverse conditions.

This repository contains the implementation, feature extraction modules, and all necessary tools for reproducing the results.

## Features
- **Feature Extraction:** Implements a CNN-based feature extractor for audio signals using mel spectrograms.
- **Data Augmentation:** Includes techniques like pitch shifting, time stretching, and noise addition to improve generalization.
- **Classification:** Uses a LinearSVC classifier to achieve high accuracy on the ESC-50 dataset.
- **Cross-Validation:** Supports stratified k-fold cross-validation to evaluate model performance.

## Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the ESC-50 dataset and place the audio files in the `ESC-50-master/audio` directory and metadata in `ESC-50-master/meta/esc50.csv`.

## Usage

1. Run the main pipeline:
   ```bash
   python main_pipeline.py
   ```
   During execution, you will be prompted to enable data augmentation and cross-validation.

2. Results, including classification reports and metrics, will be displayed in the terminal.

## Code Overview

- **`main_pipeline.py`:** Main script to run the entire pipeline including feature extraction, data augmentation, and classification.
- **`data_augmentation.py`:** Implements audio transformations for augmenting the dataset.
- **`cross_validation.py`:** Handles k-fold cross-validation for robust evaluation.
- **`feat_extractor.py`:** Contains methods for extracting audio features using a pre-trained CNN.
- **`extractor.py`:** Defines the CNN architecture and feature extraction layer.
- **`network_architecture.py`:** Implements the CNN architecture used for environmental sound classification.

## Network Architecture Description

The `network_architecture.py` file defines a convolutional neural network (CNN) for extracting features from audio data. Here are the key details:

- **Class Definition:** The `weak_mxh64_1024` class inherits from `nn.Module` and defines a CNN with 19 layers.
- **Layers:**
  - Sequential layers include convolutional layers (`Conv2d`), batch normalization (`BatchNorm2d`), and activation functions (`ReLU`).
  - Pooling layers (`MaxPool2d`) progressively reduce the spatial dimensions of feature maps.
  - A final convolutional layer reduces the features to the number of output classes, with a sigmoid activation for class probabilities.
- **Global Pooling:** The model uses a global pooling function (`glplfn`) to aggregate the feature maps into a single feature vector per class.
- **Forward Pass:** The forward method applies the defined layers sequentially to the input tensor and outputs class probabilities after global pooling.

This architecture supports robust feature extraction for variable-length audio samples, enabling state-of-the-art performance on the ESC-50 dataset.

## Results
The model achieved significant performance improvements on the ESC-50 dataset, with state-of-the-art accuracy when using data augmentation and cross-validation.

## Acknowledgments
This project is inspired by research in audio event detection and classification using weakly labeled datasets. Special thanks to the creators of the ESC-50 dataset and the PyTorch community.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
