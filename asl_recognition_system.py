# Hand Gesture Recognition System
#
# Author: Manushi
# Date: 20 April 2024
# File: asl_recognition_system.py
#
# This script integrates a convolutional neural network (CNN) with real-time hand tracking using MediaPipe
# to recognize and classify hand gestures. It captures video from a webcam, processes each frame to detect
# hand landmarks, converts these landmarks into a format suitable for the CNN, and classifies the gesture.
# Recognized gestures are displayed on the screen. The system handles unknown gestures by displaying 'Unknown Gesture'
# when the confidence is below a certain threshold.

import cv2
import mediapipe as mp
import numpy as np
import pickle
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn


class CNNModel(nn.Module):
    """
    A convolutional neural network model for classifying hand gestures.

    This CNN architecture is configured with three convolutional layers each followed by a ReLU activation and max pooling.
    The extracted features are flattened and passed through a dense layer to classify into predefined gesture classes.
    """

    def __init__(self, num_classes):
        """
        Initializes the CNN model with the necessary layers.

        Args:
            num_classes (int): The number of distinct classes for output layer.
        """
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        """
        Defines the computation performed at every call of the CNN model.

        Args:
            x (Tensor): The input data, a batch of images.

        Returns:
            Tensor: Output tensor after passing through the layers of the CNN model.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_image_from_landmarks(landmarks, img_size=(64, 64)):
    """
    Generates a 2D image from hand landmarks for gesture recognition.

    This function processes hand landmarks detected by a hand tracking system, plotting them on a blank image
    of a specified size. Each landmark is drawn as a white circle on a black background.

    Args:
        landmarks (list of dicts): List of landmark positions with 'x' and 'y' as keys.
        img_size (tuple): The size of the output image (width, height).

    Returns:
        np.array: An image array with landmarks plotted.
    """
    img = np.zeros(img_size + (3,), dtype=np.uint8)
    if landmarks:
        for hand_landmarks in landmarks:
            for lm in hand_landmarks:
                x, y = int(lm['x'] * img_size[1]), int(lm['y'] * img_size[0])
                cv2.circle(img, (x, y), 2, (255, 255, 255), -1)
    return img


def load_model(model_path):
    """
    Loads a pre-trained CNN model from a file along with its class names.

    This function reads a saved model state and its corresponding class names from a pickle file.
    It initializes a CNNModel instance with the number of classes based on the loaded data, sets the model
    to evaluation mode, and returns the model along with the class labels.

    Args:
        model_path (str): The path to the pickle file containing the saved model and class names.

    Returns:
        tuple: A tuple containing the loaded model and a dictionary mapping class indices to names.
    """
    model_dict = pickle.load(open(model_path, 'rb'))
    model = CNNModel(num_classes=len(model_dict['class_names']))
    model.load_state_dict(model_dict['model_state_dict'])
    model.eval()
    return model, model_dict['class_names']


def setup_camera():
    """
    Initializes the webcam and MediaPipe hand tracking.

    This function sets up the video capture from the default camera and configures MediaPipe Hands
    for real-time hand tracking. It returns both the camera capture object and the configured MediaPipe Hands object.

    Returns:
        tuple: A tuple containing the camera capture object and the MediaPipe Hands object.
    """
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return cap, mp_hands


def process_frame(frame, model, labels_dict, transform, mp_hands, confidence_threshold):
    """
    Processes a single video frame for hand detection and gesture recognition.

    Converts the frame to RGB, processes it using MediaPipe to detect hands and extracts landmarks.
    If hands are detected, it processes these landmarks to create an input tensor for the CNN.
    It predicts the gesture class using the CNN model, and annotates the frame with the prediction result.

    Args:
        frame (numpy.ndarray): The current video frame.
        model (torch.nn.Module): The pre-trained CNN model for gesture recognition.
        labels_dict (dict): Dictionary mapping class indices to gesture names.
        transform (torchvision.transforms.Compose): Transformation to apply to the input frame.
        mp_hands (mp.solutions.hands.Hands): MediaPipe Hands object for hand detection.
        confidence_threshold (float): Threshold above which a prediction is considered confident.

    Effects:
        Modifies the input frame to include the gesture recognition result as text.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame_rgb)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            landmarks = [[{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark] for hand_landmarks in
                         results.multi_hand_landmarks]

            img_from_landmarks = create_image_from_landmarks(landmarks)
            img_tensor = transform(img_from_landmarks).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                max_prob, predicted = probabilities.max(1)

                if max_prob.item() < confidence_threshold:
                    predicted_label = 'Unknown Gesture'
                else:
                    predicted_label = labels_dict[predicted.item()]

                cv2.putText(frame, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "No Hand Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Hand Tracking', frame)


def main():
    """
    Main function to execute the gesture recognition system.

    This function loads the model, sets up the camera, and enters a loop to continuously process
    each frame from the camera. It uses MediaPipe for hand detection and a CNN model for gesture recognition.
    The recognized gestures are displayed on the screen, and the loop can be exited by pressing the 'ESC' key.

    Effects:
        Continuously captures frames from the webcam, processes them for gesture recognition, and displays the results.
        The loop runs until the 'ESC' key is pressed or the webcam is disconnected.
    """
    model, class_names = load_model('./cnn_model.p')
    labels_dict = {i: label for i, label in enumerate(class_names)}
    cap, mp_hands = setup_camera()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    confidence_threshold = 0.7  # Threshold for classifying a gesture as recognized

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            process_frame(frame, model, labels_dict, transform, mp_hands, confidence_threshold)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        mp_hands.close()


if __name__ == '__main__':
    main()