# Hand Gesture Recognition Using CNN
#
# Author: Manushi
# Date: 20 April 2024
# File: asl_model_builder.py
#
# This script employs a convolutional neural network (CNN) to classify hand gesture images. It utilizes the torchvision
# library to manage image transformations and the PyTorch framework for constructing and training the CNN. The dataset
# consists of pre-processed hand gesture images from various individuals, converted from landmarks to 2D images. This
# script demonstrates the loading of this dataset, the dynamic adjustment of the CNN architecture for hand gesture
# recognition, and the evaluation of the model's performance using metrics such as accuracy, precision, recall, and
# F1-score. Additionally, it showcases the use of a learning rate scheduler to improve training efficiency and early
# stopping to prevent overfitting. The script also provides functionality for saving the trained model and plotting
# confusion matrices to visually analyze the model's performance.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchinfo import summary


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
        landmarks_scaled = [(int(lm['x'] * img_size[1]), int(lm['y'] * img_size[0])) for hand in landmarks for lm in
                            hand]
        for x, y in landmarks_scaled:
            cv2.circle(img, (x, y), 2, (255, 255, 255), -1)
    return img


class HandGestureDataset(Dataset):
    """
    Custom dataset class for hand gesture recognition using PyTorch.

    This class handles loading images and their corresponding labels for training and testing a neural network.
    The dataset is expected to consist of images converted from hand landmarks to 2D images with labels for each gesture.
    """
    def __init__(self, data, labels, transform=None):
        """
        Initializes the dataset with data, labels, and optional transformations.

        Args:
            data (list): List of images.
            labels (list): Corresponding labels for the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetches the image and label at the specified index, applying transformations if specified.

        Args:
            idx (int): Index of the image and label to return.

        Returns:
            tuple: Tuple containing the transformed image and its label.
        """
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


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


class GestureRecognizer:
    """
    Manages the training and evaluation process for a gesture recognition convolutional neural network.

    This class encapsulates the neural network model along with methods for training, evaluating,
    and plotting the performance metrics such as the confusion matrix. It is designed to work with
    PyTorch models and data loaders.
    """
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, scheduler, device='cpu'):
        """
        Initializes the GestureRecognizer with all components necessary for training and evaluation.

        Args:
            model (nn.Module): The neural network model to be trained and evaluated.
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the testing/validation dataset.
            criterion (loss function): The loss function used for training the model.
            optimizer (Optimizer): The optimizer used for updating model weights.
            scheduler (lr_scheduler): Learning rate scheduler to adjust the learning rate during training.
            device (str): Device to which tensors will be sent ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_model(self, num_epochs=50, early_stopping_threshold=9):
        """
        Trains the neural network model using the provided training dataset.

        Args:
            num_epochs (int): Maximum number of epochs to train the model.
            early_stopping_threshold (int): The number of consecutive epochs without improvement in validation loss to stop training.

        During each epoch, the model is trained on the training dataset, and the validation loss is computed
        using the test dataset. Early stopping is employed to prevent overfitting.
        """
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            self.model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Validation loss
            validation_loss = self._validate_model()

            # Learning rate adjustment
            self.scheduler.step(validation_loss)

            # Early stopping check
            if validation_loss < best_loss:
                best_loss = validation_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_threshold:
                    print(f'Early stopping triggered after {epoch + 1} epochs!\n')
                    break

            print(f'Epoch {epoch + 1}: Training Loss: {loss.item():.4f}, Validation Loss: {validation_loss:.4f}')

    def _validate_model(self):
        """
        Validates the model on the test dataset to compute and return the average loss.

        Returns:
            float: The average validation loss across all batches of the test dataset.
        """
        self.model.eval()
        validation_loss = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                validation_loss += loss.item()
        return validation_loss / len(self.test_loader)

    def evaluate_model(self, class_names):
        """
        Evaluates the model's performance on the test dataset and prints accuracy, precision, recall, F1 score,
        and plots a confusion matrix.

        Args:
            class_names (list of str): List of class names corresponding to labels.
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='macro')
        recall = recall_score(all_targets, all_predictions, average='macro')
        f1 = f1_score(all_targets, all_predictions, average='macro')
        cm = confusion_matrix(all_targets, all_predictions)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        self._plot_confusion_matrix(cm, class_names)

        print("\nClassification Report:\n",
              classification_report(all_targets, all_predictions, target_names=class_names))

    def _plot_confusion_matrix(self, cm, class_names):
        """
        Plots a confusion matrix using seaborn heatmap.

        Args:
            cm (array): Confusion matrix data as a 2D array.
            class_names (list of str): List of class names for the axis labels.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()

    def print_model_summary(self, input_size=(1, 3, 64, 64)):
        """
        Prints a summary of the CNN model showing all layers, their types, output shapes, and the number of parameters.

        This method utilizes the torchinfo library to generate a detailed summary of the neural network,
        helping in understanding the network's architecture and parameter distribution.

        Args:
            input_size (tuple): The expected input size of the model, format (batch_size, channels, height, width).
        """
        print(summary(self.model, input_size=input_size))

    def save_model(self, model_path, class_names):
        """
        Saves the trained model state to a specified path using Python's pickle module.

        This method serializes the model state dictionary, which includes weights and biases of all model layers,
        facilitating model persistence for later use or transfer learning.

        Args:
            model_path (str): The file path where the model state dictionary will be saved.
            class_names (list of str): List of class names corresponding to the output neurons of the model.
        """
        classes = [label for label in class_names]

        with open(model_path, 'wb') as f:
            # Saving only the model state dictionary
            pickle.dump({'model_state_dict': self.model.state_dict()}, f)

        with open(model_path, 'wb') as f:
            pickle.dump({'model_state_dict': self.model.state_dict(), 'class_names': classes}, f)

        print("Model saved successfully.")


def main():
    """
    Main function to execute the gesture recognition training and evaluation.

    This function loads the dataset, creates training and testing DataLoader objects, initializes the model and
    training components, and orchestrates the model training and evaluation. It also saves the trained model and
    prints a summary of the model's architecture.
    """
    # Data loading and preprocessing
    with open('./all_data.pickle', 'rb') as f:
        all_data = pickle.load(f)

    processed_data = []
    labels = []

    for item in all_data:
        if item['landmarks']:
            img_tensor = create_image_from_landmarks(item['landmarks'])
            processed_data.append(img_tensor)
            labels.append(item['label'])

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    processed_data = np.array(processed_data)
    labels = np.array(labels)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    x_train, x_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2, random_state=42)
    train_dataset = HandGestureDataset(x_train, y_train, transform)
    test_dataset = HandGestureDataset(x_test, y_test, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model setup
    class_names = le.classes_
    model = CNNModel(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

    recognizer = GestureRecognizer(model, train_loader, test_loader, criterion, optimizer, scheduler)
    recognizer.train_model()
    recognizer.evaluate_model(class_names)
    recognizer.print_model_summary(input_size=(1, 3, 64, 64))

    model_save_path = './cnn_model.p'
    recognizer.save_model(model_save_path, class_names)


if __name__ == '__main__':
    main()
