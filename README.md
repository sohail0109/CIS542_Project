# CIS542_Project
# Deepfake Detector

This script allows you to detect whether an image is a deepfake or not using a pre-trained model.

## Prerequisites

- Python
- TensorFlow (install using `pip install tensorflow`)
- Pillow (install using `pip install pillow`)

## Usage

1. Running the script:

   ```bash
   python detector.py --image_path "/pathToPicture" --model_path "/pathTomodel"
   ```
   eg

   ```bash
   python detector.py --image_path "test_images/real.JPG" --model_path ""
   ```
