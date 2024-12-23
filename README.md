# SMUL-Enviornment-Sound-Classification
This project explores advanced environmental sound classification using CNNs and transfer learning, leveraging weakly labeled web audio for sound event detection. Data augmentation techniques like pitch shifting, time stretching, and noise additio. A LinearSVC is used for classification, achieving state-of-the-art results on the ESC-50 dataset.
# Environmental Sound Classification Using CNN and LinearSVC

This project explores advanced methods for environmental sound classification using convolutional neural networks (CNNs) and transfer learning techniques. The proposed framework leverages weakly labeled web audio data for sound event detection and classification, demonstrating efficiency with variable-length audio samples and achieving state-of-the-art performance on the ESC-50 dataset. 

To further enhance model robustness and generalization, data augmentation techniques such as pitch shifting, time stretching, and noise addition were applied, simulating real-world variations in environmental sound data. A Linear Support Vector Machine (LinearSVC) is employed as the classification method to distinguish between environmental sound categories under diverse conditions effectively.

This repository contains the implementation, feature extraction modules, and all necessary tools for reproducing the results.

## Features
- **Feature Extraction:** Implements a CNN-based feature extractor for audio signals using mel spectrograms.
- **Data Augmentation:** Includes techniques like pitch shifting, time stretching, and noise addition to improve generalization.
- **Classification:** Uses a LinearSVC classifier to achieve high accuracy on the ESC-50 dataset.
- **Cross-Validation:** Supports stratified k-fold cross-validation to evaluate model performance.

## Setup

1. Clone this repository:
   ```bash
   git clone (https://github.com/AngiusHamilton1/SMUL-Enviornment-Sound-Classification.git)
   cd weak_feature_extractor
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the ESC-50 dataset and place the audio files in the `ESC-50-master/audio` directory and metadata in `ESC-50-master/meta/esc50.csv`.

## Usage

1. Run the main pipeline:
   ```bash
   python main.py
   ```
   During execution, you will be prompted to enable data augmentation and cross-validation.

2. Results, including classification reports and metrics, will be displayed in the terminal.

## Code Overview

- **`main_pipeline.py`:** Main script to run the entire pipeline including feature extraction, data augmentation, and classification.
- **`data_augmentation.py`:** Implements audio transformations for augmenting the dataset.
- **`cross_validation.py`:** Handles k-fold cross-validation for robust evaluation.
- **`feat_extractor.py`:** Contains methods for extracting audio features using a pre-trained CNN.
- **`extractor.py`:** Defines the CNN architecture and feature extraction layer.

## Results
The model achieved significant performance improvements on the ESC-50 dataset, with state-of-the-art accuracy when using data augmentation and cross-validation.

## Acknowledgments
This project is inspired by research in audio event detection and classification using weakly labeled datasets. Special thanks to the creators of the ESC-50 dataset and the PyTorch community.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
