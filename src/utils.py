# Name: Jay Shah
# Date: 03/31/2026
# Purpose: Utility functions for MNIST project, including visualization of
#          the first six test images using matplotlib for report inclusion.
# Run: Imported and used by visualization scripts

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def show_first_six_digits(): # Load the MNIST test dataset, extract the first six images and their labels, and visualize them in a 2x3 grid using matplotlib, saving the visualization to a file for report inclusion and analysis of the dataset
    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.MNIST( # Load MNIST test dataset to get the first six images for visualization and analysis of the dataset
        './data',
        train=False,
        download=True,
        transform=transform
    )

    fig, axes = plt.subplots(2, 3, figsize=(6, 4))

    for i in range(6): # Loop through the first six test images, extract the image and label, and display them in a 2x3 grid layout with the label as the title for each image, which allows for visual analysis of the dataset and understanding of the types of images and labels present in the MNIST test set
        image, label = test_dataset[i]

        row = i // 3
        col = i % 3

        axes[row, col].imshow(image.squeeze(), cmap='gray')
        axes[row, col].set_title(f"Label: {label}")
        axes[row, col].axis('off')

    plt.tight_layout()

    plt.savefig("images/first_six_digits.png")

    plt.show()