# American Sign Language (ASL) Recognition System

This repository contains a set of Python scripts that facilitate the creation, annotation, training, and real-time recognition of American Sign Language gestures using machine learning and computer vision techniques.

## Overview

The system consists of four main components:

1. **ASL Dataset Collection**: Script to collect images for creating an ASL dataset using a webcam.
2. **ASL Landmark Annotation**: Script to annotate these images with hand landmarks using MediaPipe.
3. **ASL Model Builder**: Script for training a convolutional neural network (CNN) on the processed dataset.
4. **Real-time ASL Recognition**: Script that uses a pre-trained model to recognize ASL signs in real-time via a webcam.

### Prerequisites

- Python 3.8+
- OpenCV
- PyTorch
- torchvision
- MediaPipe
- NumPy
- Matplotlib
- scikit-learn

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/your-github-username/asl-recognition.git
cd asl-recognition
```

Install the required Python packages:

```bash
pip install opencv-python mediapipe torch torchvision numpy matplotlib scikit-learn
```

## Usage

Each script in the repository has a specific role in the ASL recognition pipeline:

### 1. ASL Dataset Builder (`asl_dataset_builder.py`)

This script captures images from a webcam and saves them into folders corresponding to different ASL gestures.

**Usage:**

```bash
python asl_dataset_builder.py
```

### 2. ASL Landmark Annotation (`asl_landmark_annotation.py`)

This script reads the saved images and uses MediaPipe to annotate them with hand landmarks, saving the output in a pickle format.

**Usage:**

```bash
python asl_landmark_annotation.py
```

### 3. ASL Model Builder (`asl_model_builder.py`)

This script processes the annotated data to train a CNN for ASL gesture recognition. It evaluates the model's performance and saves it for further use.

**Usage:**

```bash
python asl_model_builder.py
```

### 4. Real-Time ASL Recognition (`asl_recognition_system.py`)

This script uses the trained model to recognize ASL gestures in real-time from webcam feed.

**Usage:**

```bash
python asl_recognition_system.py
```

## Contact

**Name:** Manushi
