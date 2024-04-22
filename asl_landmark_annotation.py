# ASL Landmark Annotation Script
#
# Author: Manushi
# Date: 20 April 2024
# File: asl_landmark_annotation.py
#
# This script processes images of American Sign Language (ASL) gestures to detect and annotate hand landmarks using
# Google's MediaPipe library. It reads images from a predefined directory, applies hand detection, and annotates the
# images with landmarks. The landmarks are then saved in a pickle format for further analysis and model training.
# The script is designed to aid in the development of ASL recognition systems by providing detailed landmark annotations
# for training robust gesture recognition models.

import cv2
import mediapipe as mp
import os
import pickle


class HandLandmarkAnnotator:
    """
    Handles the detection and annotation of hand landmarks in images using MediaPipe.

    Attributes:
        data_dir (str): Directory containing gesture images.
        mp_hands (object): MediaPipe Hands object for hand landmark detection.
        all_data (list): Stores all hand landmark data and metadata for images.
    """

    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        self.all_data = []

    def process_images(self):
        """
        Processes each image in the data directory, annotates it with landmarks, and collects the data.
        """
        for label in os.listdir(self.data_dir):
            label_path = os.path.join(self.data_dir, label)
            for img_file in os.listdir(label_path):
                self._process_single_image(label, label_path, img_file)

        self._save_all_data()

    def _process_single_image(self, label, label_path, img_file):
        """
        Processes a single image, detects landmarks, draws them on the image, and saves the data.

        Args:
            label (str): The gesture label.
            label_path (str): Path to the label directory.
            img_file (str): The image file name.
        """
        img_path = os.path.join(label_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(img_rgb)

        img_drawn = img.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self._draw_landmarks(img_drawn, hand_landmarks)

        cv2.imshow('Hand Landmarks', img_drawn)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            return

        landmarks = self._extract_landmarks(results)
        self.all_data.append({'label': label, 'image_path': img_path, 'landmarks': landmarks})

    def _draw_landmarks(self, image, landmarks):
        """
        Draws hand landmarks on the image.

        Args:
            image (ndarray): The image on which to draw.
            landmarks (object): Landmarks to draw on the image.
        """
        mp.solutions.drawing_utils.draw_landmarks(
            image, landmarks, mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())

    def _extract_landmarks(self, results):
        """
        Extracts hand landmarks from MediaPipe results.

        Args:
            results (object): MediaPipe hand landmark detection results.

        Returns:
            list: List of dictionaries with x, y, z coordinates of landmarks.
        """
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
                landmarks.append(hand_data)
        return landmarks

    def _save_all_data(self):
        """
        Saves all collected data to a pickle file for later use.
        """
        pickle_path = os.path.join(self.data_dir, 'all_data.pickle')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.all_data, f)

    def close_resources(self):
        """
        Releases resources used by MediaPipe and OpenCV.
        """
        cv2.destroyAllWindows()
        self.mp_hands.close()


def main():
    """
    Main function to execute the ASL landmark annotation process.

    Initializes the HandLandmarkAnnotator, processes images, and releases resources.
    """
    annotator = HandLandmarkAnnotator()
    annotator.process_images()
    annotator.close_resources()


if __name__ == '__main__':
    main()
