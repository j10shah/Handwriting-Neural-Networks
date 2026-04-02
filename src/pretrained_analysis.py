# Name: Jay Shah
# Date: 03/31/2026
# Purpose: Extension - Analyze a pretrained ResNet18 model by visualizing
#          its first convolutional layer filters and their effects on images.
# Run: python src/pretrained_analysis.py

import sys
import os
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import cv2
import numpy as np


def load_model(): # Load the pretrained ResNet18 model from torchvision, set it to evaluation mode, and return the model for further analysis of its convolutional filters and their effects on images
    model = models.resnet18(pretrained=True)
    model.eval()
    return model


def print_model(model): # Print the structure of the loaded ResNet18 model to understand its architecture and identify the layers, especially the first convolutional layer, for analysis and visualization of its filters and their effects on images
    print("\n--- ResNet18 Model Structure ---\n")
    print(model)


def get_conv1_weights(model): # Extract the weights of the first convolutional layer (conv1) from the loaded ResNet18 model, print its shape and the weights of the first filter for analysis, and return the weights for visualization of the filters and their effects on images
    weights = model.conv1.weight  # shape: (64, 3, 7, 7)
    print("\nConv1 weight shape:", weights.shape)
    return weights


def show_filters(weights): # Visualize the first convolutional layer filters from the ResNet18 model by normalizing the filter weights and displaying them as images in a grid layout, saving the visualization to a file for report inclusion and analysis of the learned features in the pretrained model
    os.makedirs("images", exist_ok=True)

    fig, axes = plt.subplots(8, 8, figsize=(10, 10))

    for i in range(64): # Loop through the 64 filters in conv1 and visualize each filter by normalizing its weights and displaying it as an image in an 8x8 grid layout, which allows for analysis of the learned features in the pretrained ResNet18 model
        filt = weights[i].detach().numpy()

        filt = filt.mean(axis=0)

        ax = axes[i // 8][i % 8]
        ax.imshow(filt, cmap='viridis')
        ax.set_title(f"F{i}", fontsize=6)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("images/resnet_filters.png")
    plt.show()


def apply_filters(weights): # Apply the first convolutional layer filters from the ResNet18 model to a sample image (a handwritten digit '5' created using OpenCV) and visualize the results by displaying the filtered images in a grid layout, which allows for analysis of how the pretrained filters affect the input image and what features they extract
    os.makedirs("images", exist_ok=True)

    image = np.zeros((28, 28), dtype=np.float32)
    cv2.putText(image, '5', (5, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (1,), 2)

    results = []

    for i in range(8):  # show first 8 filters
        kernel = weights[i].mean(dim=0).detach().numpy()
        filtered = cv2.filter2D(image, -1, kernel)
        results.append(filtered)

    fig, axes = plt.subplots(2, 4, figsize=(8, 4))

    for i in range(8): # Loop through the first 8 filters and their results to display them in a grid layout, showing how each filter from the pretrained ResNet18 model affects the sample image of the digit '5' and what features it extracts, which can provide insights into the learned representations in the model
        ax = axes[i // 4][i % 4]
        ax.imshow(results[i], cmap='gray')
        ax.set_title(f"F{i}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("images/resnet_filter_results.png")
    plt.show()


def main(argv): # Main function to load the pretrained ResNet18 model, print its structure, extract and visualize the first convolutional layer filters, and apply those filters to a sample image to analyze their effects, which provides insights into the learned features in the pretrained model and their impact on input images
    model = load_model()
    print_model(model)

    weights = get_conv1_weights(model)

    show_filters(weights)
    apply_filters(weights)


if __name__ == "__main__":
    main(sys.argv)