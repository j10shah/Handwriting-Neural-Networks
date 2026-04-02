# Name: Jay Shah
# Date: 03/31/2026
# Purpose: Task 2 - Apply the first convolutional layer filters to an input image
#          and visualize the effect of each filter on the image.
# Run: python src/filter_effects.py

import sys
import torch
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

from torchvision import datasets, transforms
from network import MyNetwork


def show_filter_effects(): # Apply conv1 filters to a sample image and visualize results
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST( # Load MNIST training dataset to get a sample image for filtering
        './data',
        train=True,
        download=True,
        transform=transform
    )

    image, label = train_dataset[0]
    image_np = image.squeeze().numpy().astype(np.float32)

    model = MyNetwork() # Load the trained CNN model to access the conv1 filters
    model.load_state_dict(torch.load("models/mnist_cnn.pth"))
    model.eval()

    filtered_images = []

    with torch.no_grad(): # Get the conv1 weights and apply each filter to the sample image using OpenCV's filter2D function
        weights = model.conv1.weight

        for i in range(10): # Loop through the 10 filters in conv1 and apply them to the image
            kernel = weights[i, 0].detach().numpy().astype(np.float32)

            filtered = cv2.filter2D(image_np, -1, kernel)

            filtered_images.append(filtered)

    plot_results(weights, filtered_images)


def plot_results(weights, filtered_images): # Visualize the conv1 filters and their effects on the sample image in a grid layout
    os.makedirs("images", exist_ok=True)

    fig, axes = plt.subplots(5, 4, figsize=(10, 10))

    for i in range(5): # Loop through the first 5 filters and their results to display them in a grid
        w1 = weights[2 * i, 0].detach().numpy() # Get the weights of the first filter in the pair (even index)
        f1 = filtered_images[2 * i]

        w1_norm = (w1 - w1.min()) / (w1.max() - w1.min() + 1e-8) # Normalize the filter weights for better visualization

        axes[i, 0].imshow(w1_norm, cmap='viridis') # Show the filter weights as an image
        axes[i, 0].set_title(f"Filter {2*i}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(f1, cmap='gray')
        axes[i, 1].set_title("Result")
        axes[i, 1].axis('off')

        w2 = weights[2 * i + 1, 0].detach().numpy()
        f2 = filtered_images[2 * i + 1]

        w2_norm = (w2 - w2.min()) / (w2.max() - w2.min() + 1e-8) # Normalize the second filter weights

        axes[i, 2].imshow(w2_norm, cmap='viridis')
        axes[i, 2].set_title(f"Filter {2*i+1}")
        axes[i, 2].axis('off')

        axes[i, 3].imshow(f2, cmap='gray')
        axes[i, 3].set_title("Result")
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig("images/conv1_filterResults.png")
    plt.show()


def main(argv): # Main function to run the filter effects visualization
    show_filter_effects()


if __name__ == "__main__":
    main(sys.argv)