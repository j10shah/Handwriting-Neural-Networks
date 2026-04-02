# Name: Jay Shah
# Date: 03/31/2026
# Purpose: Task 1 - Test the trained CNN on user-provided handwritten digit images
#          by preprocessing, resizing, and classifying them.
# Run: python src/custom_input.py

import sys
import torch
import cv2
import os
import matplotlib.pyplot as plt

from network import MyNetwork


def preprocess_image(path): # Preprocess the input image to match MNIST format
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # Read image in grayscale

    img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1] # Binarize image

    img = cv2.resize(img, (28, 28)) # Resize to 28x28 pixels
    img = 255 - img

    img = img / 255.0

    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0).unsqueeze(0)

    return img


def test_custom_digits(): # Test the trained CNN on custom handwritten digit images
    model = MyNetwork()
    model.load_state_dict(torch.load("models/mnist_cnn.pth"))
    model.eval()

    images = []
    predictions = []

    print("\n--- Handwritten Digit Predictions ---\n")

    for i in range(10): # Loop through digit_0.png to digit_9.png
        path = f"images/handwritten/digit_{i}.png"

        img_tensor = preprocess_image(path)

        output = model(img_tensor) # Get model output
        probs = output.detach().numpy()[0]
        pred = probs.argmax()

        print(f"Digit {i}")
        print("Outputs:", ["{:.2f}".format(p) for p in probs])
        print(f"Predicted: {pred}\n")

        img_display = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # Read original image for display
        images.append(img_display)
        predictions.append(pred)

    plot_results(images, predictions)


def plot_results(images, predictions): # Plot the custom digit images with their predicted labels
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    for i in range(10): # Loop through the 10 images and predictions
        row = i // 5
        col = i % 5

        axes[row, col].imshow(images[i], cmap='gray')
        axes[row, col].set_title(f"Pred: {predictions[i]}")
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig("images/custom_predictions.png")
    plt.show()


def main(argv): # Main function to run the custom digit testing
    test_custom_digits()


if __name__ == "__main__":
    main(sys.argv)