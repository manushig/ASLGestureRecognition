# ASL Dataset Collection Script
#
# Author: Manushi
# Date: 20 April 2024
# File: asl_dataset_builder.py
#
# This script facilitates the creation of an American Sign Language (ASL) gesture recognition dataset. It utilizes a
# webcam to capture and save a specified number of images for each predefined ASL gesture label such as 'hello',
# 'thanks', 'yes', 'no', and 'iloveyou'. The script is designed to operate interactively, prompting the user to begin
# capturing images for each label with a simple keypress. This setup assists in building a diverse dataset for
# training machine learning models, specifically focusing on improving the accessibility of communication tools for
# the deaf and hard of hearing community. The script handles the setup of necessary directories for each label and
# ensures that images are stored efficiently, making it easier to manage and access the dataset for future machine
# learning tasks.

import cv2
import uuid
import os

class ASLDatasetBuilder:
    """
    ASL Dataset Builder

    This class facilitates the collection of American Sign Language (ASL) gesture images using a webcam.
    It allows the user to capture and save a specified number of images for predefined ASL labels.

    Attributes:
        data_dir (str): The directory where the dataset will be stored.
        labels (list of str): The list of gesture labels for which images are collected.
        dataset_size (int): The number of images to collect for each label.
    """

    def __init__(self, data_dir='./data', labels=None, dataset_size=20):
        """
        Initializes the ASL Dataset Builder with data directory, labels, and dataset size.

        Args:
            data_dir (str): The directory to store captured images.
            labels (list of str): Labels for which images are to be collected.
            dataset_size (int): Number of images to collect for each label.
        """
        if labels is None:
            labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
        self.data_dir = data_dir
        self.labels = labels
        self.dataset_size = dataset_size
        self._prepare_directories()

    def _prepare_directories(self):
        """
        Prepares the directories for storing images. Creates a main directory and subdirectories for each label.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        for label in self.labels:
            label_path = os.path.join(self.data_dir, label)
            if not os.path.exists(label_path):
                os.makedirs(label_path)

    def collect_images(self):
        """
        Starts the webcam and collects images for each label.

        The method opens a webcam feed, displays the current frame, and captures images upon pressing 'q'.
        Each captured image is saved to the corresponding label directory.
        """
        cap = cv2.VideoCapture(0)
        try:
            for label in self.labels:
                print(f'Collecting images for {label}. Press "q" to start.')
                while True:
                    ret, frame = cap.read()
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                for imgNum in range(self.dataset_size):
                    print(f'Collecting image {imgNum} for {label}')
                    ret, frame = cap.read()
                    imgName = os.path.join(self.data_dir, label, f'{uuid.uuid4()}.jpg')
                    cv2.imwrite(imgName, frame)
                    cv2.imshow('frame', frame)
                    cv2.waitKey(500)
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """
    Main function to execute the ASL dataset collection process.

    This function initializes the dataset builder and starts the image collection process for predefined labels.
    """
    dataset_builder = ASLDatasetBuilder()
    dataset_builder.collect_images()

if __name__ == '__main__':
    main()

