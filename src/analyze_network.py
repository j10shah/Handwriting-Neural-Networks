# Name: Jay Shah
# Date: 03/31/2026
# Purpose: Task 2 - Analyze the trained CNN by extracting and visualizing
#          the first convolutional layer filters.
# Run: python src/analyze_network.py

import sys
import torch
import matplotlib.pyplot as plt
import os

from network import MyNetwork


def analyze_network(): # Analyze the trained CNN model and visualize conv1 filters
    model = MyNetwork() # Load the trained model
    model.load_state_dict(torch.load("models/mnist_cnn.pth"))
    model.eval() # Set model to evaluation mode

    print("\n--- Model Structure ---\n")
    print(model)

    weights = model.conv1.weight.data # Get conv1 weights (shape: [10, 1, 5, 5])

    print("\n--- conv1 Weights Info ---\n")
    print("Shape:", weights.shape)   # should be [10, 1, 5, 5]

    print("\nFirst filter weights:\n", weights[0, 0])

    visualize_filters(weights)


def visualize_filters(weights): # Visualize the first convolutional layer filters
    os.makedirs("images", exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(8, 6))

    for i in range(10): # Loop through the 10 filters
        row = i // 4
        col = i % 4

        filter_img = weights[i, 0].numpy()

        axes[row, col].imshow(filter_img, cmap='viridis') # Show filter as image
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        axes[row, col].set_title(f"Filter {i}")

    axes[2, 2].axis('off')
    axes[2, 3].axis('off')

    plt.tight_layout()
    plt.savefig("images/conv1_filters.png")
    plt.show()


def main(argv): # Main function to run the analysis
    analyze_network()


if __name__ == "__main__":
    main(sys.argv)