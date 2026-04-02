# Name: Jay Shah
# Date: 03/31/2026
# Purpose: Task 1 - Load the trained CNN model and evaluate it on the first
#          10 MNIST test images, printing outputs and displaying predictions.
# Run: python src/test_model.py

import sys
import torch
import matplotlib.pyplot as plt
import os

from torchvision import datasets, transforms
from network import MyNetwork


def test_network(): # Load the MNIST test dataset, load the trained CNN model, evaluate the model on the first 10 test images by printing the output probabilities, predicted labels, and actual labels for each image, and visualize the predictions in a grid layout for analysis of the model's performance on unseen data
    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.MNIST( # Load MNIST test dataset to evaluate the model on unseen data and analyze its performance by comparing predicted labels with actual labels for the first 10 test images
        './data',
        train=False,
        download=True,
        transform=transform
    )

    model = MyNetwork() # Load the trained CNN model from the saved state dictionary, set it to evaluation mode, and use it to make predictions on the first 10 test images, which allows for analysis of the model's performance and its ability to generalize to new data
    model.load_state_dict(torch.load("models/mnist_cnn.pth")) # Load the trained model weights from the saved file "mnist_cnn.pth" in the "models" directory, which contains the learned parameters of the CNN model after training on the MNIST dataset, allowing for evaluation of the model's performance on the test dataset
    model.eval()  

    print("\n--- Model Outputs on First 10 Test Images ---\n")

    predictions = []
    images = []

    for i in range(10): # Loop through the first 10 test images, get the model's output probabilities for each image, determine the predicted label by taking the argmax of the output, and print the predicted and actual labels for analysis of the model's performance on these test examples
        image, label = test_dataset[i]
        image = image.unsqueeze(0)

        output = model(image)
        probs = output.detach().numpy()[0]

        pred = probs.argmax()

        predictions.append(pred)
        images.append(image.squeeze())

        print(f"Example {i}")
        print("Outputs:", ["{:.2f}".format(p) for p in probs])
        print(f"Predicted: {pred}, Actual: {label}\n")

    plot_predictions(images, predictions)


def plot_predictions(images, predictions): # Visualize the first 10 test images along with their predicted labels in a grid layout, which allows for analysis of the model's performance by visually comparing the predicted labels with the actual images and understanding any misclassifications
    os.makedirs("images", exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))

    for i in range(9): # Loop through the first 9 images and their predictions to display them in a 3x3 grid layout, showing the predicted label for each image, which allows for visual analysis of the model's performance and identification of any patterns in misclassifications
        row = i // 3
        col = i % 3

        axes[row, col].imshow(images[i], cmap='gray')
        axes[row, col].set_title(f"Pred: {predictions[i]}")
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig("images/predictions.png")
    plt.show()


def main(argv): # Main function to run the testing process for the trained CNN model on the MNIST test dataset, which includes loading the model, evaluating it on the first 10 test images, printing the outputs and predictions, and visualizing the results for analysis of the model's performance on unseen data
    test_network()


if __name__ == "__main__":
    main(sys.argv)