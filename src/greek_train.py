# Name: Jay Shah
# Date: 03/31/2026
# Purpose: Task 3 - Perform transfer learning by adapting the MNIST CNN
#          to classify Greek letters (alpha, beta, gamma).
# Run: python src/greek_train.py

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import torchvision

from torchvision import transforms
from network import MyNetwork


class GreekTransform: # Custom transformation class to preprocess Greek letter images by converting to grayscale, resizing, cropping, and inverting pixel values to match the MNIST format for training the CNN
    def __call__(self, x): # Apply a series of transformations to the input image: convert to grayscale, resize to 36x36, center crop to 28x28, and invert pixel values to prepare the Greek letter images for training the CNN model
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def train_greek(): # Train the CNN model on the Greek letter dataset using transfer learning by freezing the convolutional layers and only training the final fully connected layer to classify the three Greek letters, while also tracking and plotting the training loss over epochs
    training_set_path = "images/greek_train"

    greek_train = torch.utils.data.DataLoader( #    Load the Greek letter training dataset from the specified directory, applying the custom transformations to preprocess the images for training the CNN model
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=transforms.Compose([ # Define the transformation pipeline to preprocess the Greek letter images for training, including conversion to tensor, custom Greek transformations, and normalization to match MNIST statistics
                transforms.ToTensor(),
                GreekTransform(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=5,
        shuffle=True
    )

    model = MyNetwork()
    model.load_state_dict(torch.load("models/mnist_cnn.pth"))

    for param in model.parameters(): # Freeze all layers of the pre-trained CNN model to prevent their weights from being updated during training on the Greek letter dataset, allowing only the final fully connected layer to be trained for the new classification task
        param.requires_grad = False

    model.fc2 = nn.Linear(50, 3)

    print("\n--- Modified Network ---\n")
    print(model)

    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01)

    train_losses = []

    epochs = 20  # usually small dataset → need more epochs

    for epoch in range(epochs): # Train the model for a specified number of epochs, calculating and storing the average training loss for each epoch to monitor the training progress and visualize it later with a plot
        model.train()
        total_loss = 0

        for data, target in greek_train: # Loop through the training data batches, perform a forward pass to get predictions, calculate the loss using negative log likelihood, backpropagate the loss, and update the model parameters using the optimizer, while accumulating the total loss for the epoch to compute the average loss at the end of each epoch
            optimizer.zero_grad()

            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(greek_train)
        train_losses.append(avg_loss)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/greek_model.pth")

    os.makedirs("images", exist_ok=True)

    plt.figure()
    plt.plot(range(1, epochs+1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Greek Letter Training Loss")
    plt.savefig("images/greek_training_loss.png")
    plt.show()


def main(argv): # Main function to run the training process for the Greek letter classification task using transfer learning with the pre-trained MNIST CNN model, and to visualize the training loss over epochs
    train_greek()


if __name__ == "__main__":
    main(sys.argv)