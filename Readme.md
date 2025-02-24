# Face Tracking Model

This repository contains a Jupyter Notebook used to train a face-tracking model and a `predict.py` script to perform real-time face tracking using a webcam.

## Model Training
The face tracking model was trained on **Google Colab** using a **T4 GPU** for accelerated computation. The model is based on **VGG16** as a feature extractor and consists of two output branches:

1. **Classification Output** - Determines whether a face is present in the frame.
2. **Regression Output** - Predicts the bounding box coordinates of the detected face.

### Dataset Collection and Augmentation
- A total of **30 images** were collected and labeled using the **LabelMe** library.
- Data augmentation was performed using the **Albumentations** library.
- The applied augmentations included:
  - **Rotation**
  - **Scaling**
  - **Horizontal flipping**
  - **Brightness and contrast adjustments**
  - **Gaussian noise addition**
  
After training, the model weights were exported and saved as `facetracker.h5`.

## Model Inference Using `predict.py`
The `predict.py` script loads the trained model and uses a webcam feed to detect and track faces in real time.

### How It Works:
- Captures video from the webcam.
- Processes the frame by resizing and normalizing it.
- Feeds the frame into the trained model for prediction.
- If a face is detected, it draws a bounding box around the detected face and labels it.
- The live feed continues until the user presses 'q' to exit.

### Dependencies:
Make sure you have the following installed before running `predict.py`:
```bash
pip install tensorflow opencv-python numpy
```

### Running the Script:
Run the following command to start the face tracker:
```bash
python predict.py
```

## Files in This Repository:
- `face detection.ipynb` - Jupyter Notebook containing the training pipeline.
- `facetracker.h5` - Trained model weights.
- `predict.py` - Script to perform real-time face tracking using a webcam.


