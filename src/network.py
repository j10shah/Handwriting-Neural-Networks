# Name: Jay Shah
# Date: 03/31/2026
# Purpose: Task 1 - Define the CNN architecture for MNIST digit recognition,
#          including convolutional, pooling, dropout, and fully connected layers.
# Run: Imported as a module in training and evaluation scripts

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNetwork(nn.Module): # Define a custom CNN architecture for MNIST classification, consisting of two convolutional layers with ReLU activations and max pooling, followed by a dropout layer and two fully connected layers for classification, with a log softmax activation at the end to output class probabilities
    def __init__(self): # Initialize the network architecture by defining the layers and their configurations, including convolutional layers for feature extraction, a dropout layer for regularization, and fully connected layers for classification, to be used in training and evaluating the model on the MNIST dataset
        super(MyNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x): # Define the forward pass through the network by applying the convolutional layers with ReLU activations and max pooling, followed by a dropout layer, and then passing the flattened output through the fully connected layers to produce the final class probabilities for MNIST classification
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))

        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)