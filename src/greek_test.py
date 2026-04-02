# Name: Jay Shah
# Date: 03/31/2026
# Purpose: Task 3 - Evaluate the transfer-learned CNN on Greek letter images
#          and display classification results.
# Run: python src/greek_test.py

import sys
import torch
import matplotlib.pyplot as plt
import os
import torchvision

from PIL import Image
from network import MyNetwork


class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def test_greek(): # Test the transfer-learned CNN on Greek letter images, preprocess them, and display the predicted class labels along with the original images in a grid format
    model = MyNetwork()

    model.fc2 = torch.nn.Linear(50, 3)

    model.load_state_dict(torch.load("models/greek_model.pth"))  # if you saved it
    model.eval()

    transform = torchvision.transforms.Compose([ # Define the same transformation used during training to preprocess the Greek letter images for testing
        torchvision.transforms.ToTensor(),
        GreekTransform(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    class_names = ["alpha", "beta", "gamma"]

    images = []
    preds = []

    print("\n--- Greek Test Predictions ---\n")

    for file in sorted(os.listdir("images/greek_test")): # Loop through the test images in the specified directory, preprocess them, and get predictions from the model to display results
        path = os.path.join("images/greek_test", file)

        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        output = model(img_tensor)
        pred = output.argmax().item()

        print(f"{file} → {class_names[pred]}")

        images.append(img)
        preds.append(class_names[pred])

    plot_results(images, preds)


def plot_results(images, preds): # Plot the original Greek letter images with their predicted class labels in a grid format for visualization
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))

    for i in range(len(images)): # Loop through the images and predictions, displaying each image with its corresponding predicted label in a 2x3 grid layout
        ax = axes[i // 3][i % 3]
        ax.imshow(images[i])
        ax.set_title(preds[i])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("images/greek_test_results.png")
    plt.show()


def main(argv): # Main function to run the Greek letter testing process
    test_greek()


if __name__ == "__main__":
    main(sys.argv)