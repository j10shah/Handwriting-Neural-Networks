# Name: Jay Shah
# Date: 03/31/2026
# Purpose: Extension - Replace the first convolutional layer with fixed Gabor filters,
#          freeze them, and evaluate performance compared to learned filters.
# Run: python src/gabor_experiment.py

import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from torchvision import datasets, transforms
from network import MyNetwork

import cv2
import numpy as np

def create_gabor_filters(num_filters=10): # Create a set of Gabor filters with different orientations to use as fixed weights in the first convolutional layer
    filters = []

    for i in range(num_filters): # Loop through the number of filters and create a Gabor kernel with a specific orientation for each filter
        theta = np.pi * i / num_filters
        kernel = cv2.getGaborKernel( # Create a Gabor kernel with specified parameters (size, orientation, wavelength, aspect ratio, phase offset)
            (5, 5), sigma=1.0, theta=theta,
            lambd=3.0, gamma=0.5, psi=0
        )
        filters.append(kernel)

    filters = np.array(filters)
    filters = torch.tensor(filters, dtype=torch.float32)
    filters = filters.unsqueeze(1)  # (num_filters, 1, 5, 5)

    return filters

def evaluate(model, loader): # Evaluate the model's performance on a given data loader by calculating average loss and accuracy
    model.eval()
    correct = 0
    total_loss = 0

    with torch.no_grad(): # Loop through the data in the loader, get model predictions, calculate loss, and count correct predictions to compute average loss and accuracy
        for data, target in loader: # Get the model's output for the input data, calculate the negative log likelihood loss, and determine how many predictions are correct to compute overall performance metrics
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(loader.dataset) # Calculate average loss by dividing total loss by the number of examples in the dataset
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy


def train_network(): # Train the model with Gabor filters in the first convolutional layer and evaluate performance over multiple epochs, generating training/testing loss and accuracy plots
    batch_size = 64
    epochs = 5
    learning_rate = 0.01

    transform = transforms.Compose([transforms.ToTensor()])

    train_loader = torch.utils.data.DataLoader( # Load MNIST training dataset with specified batch size
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader( # Load MNIST test dataset
        datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=False
    )

    model = MyNetwork()

    print("\n--- Using Gabor Filters in conv1 (frozen) ---\n")
    gabor_weights = create_gabor_filters(10)

    with torch.no_grad(): # Replace the conv1 weights with the Gabor filters and freeze them by setting requires_grad to False, then create an optimizer for the rest of the model parameters
        model.conv1.weight = torch.nn.Parameter(gabor_weights)
        model.conv1.bias.zero_()

    for param in model.conv1.parameters(): # Freeze the conv1 parameters so they are not updated during training
        param.requires_grad = False
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    train_steps = []

    test_losses = []
    test_steps = []

    train_accs = []
    test_accs = []

    examples_seen = 0
    initial_test_loss, initial_test_acc = evaluate(model, test_loader)

    test_steps.append(0)
    test_losses.append(initial_test_loss)
    for epoch in range(epochs): # Train the model for a specified number of epochs, evaluating and recording training/testing loss and accuracy at regular intervals to visualize performance over time
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader): # Loop through the training data batches, perform forward pass, calculate loss, backpropagate, and update model parameters while keeping the Gabor filters fixed
            optimizer.zero_grad()
            output = model(data)

            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            examples_seen += len(data)

            if batch_idx % 100 == 0: # Every 100 batches, record the training loss and the number of examples seen to track training progress
                train_steps.append(examples_seen)
                train_losses.append(loss.item())

        train_loss, train_acc = evaluate(model, train_loader)
        test_loss, test_acc = evaluate(model, test_loader)

        test_steps.append(examples_seen)
        test_losses.append(test_loss)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.4f}\n")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mnist_cnn.pth") # Save the trained model's state dictionary for future use
    

    plot_metrics(train_steps, train_losses, test_steps, test_losses,
                 train_accs, test_accs)


def plot_metrics(train_steps, train_losses, test_steps, test_losses, # Generate and save plots for training/testing loss and accuracy over time to visualize the model's performance with Gabor filters compared to learned filters
                 train_accs, test_accs):

    os.makedirs("images", exist_ok=True)

    plt.figure()
    plt.plot(train_steps, train_losses, label="Train Loss")
    plt.plot(test_steps, test_losses, label="Test Loss")
    plt.xlabel("Number of Training Examples Seen")
    plt.ylabel("Negative Log Likelihood Loss")
    plt.title("Training vs Testing Loss")
    plt.legend()
    plt.savefig("images/loss_plot.png")
    plt.show()

    plt.figure()
    epochs = range(1, len(train_accs) + 1)

    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Testing Accuracy")
    plt.legend()
    plt.savefig("images/accuracy_plot.png")
    plt.show()
    

def main(argv): # Main function to run the training process with Gabor filters and evaluate performance
    train_network()

    
if __name__ == "__main__":
    main(sys.argv)