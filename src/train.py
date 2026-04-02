# Name: Jay Shah
# Date: 03/31/2026
# Purpose: Task 1 - Train the CNN model on the MNIST dataset, evaluate performance
#          over multiple epochs, and generate training/testing loss and accuracy plots.
# Run: python src/train.py

import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from torchvision import datasets, transforms
from network import MyNetwork


def evaluate(model, loader): # Evaluate the model on the given data loader by calculating the average loss and accuracy, which allows for assessment of the model's performance on both training and testing datasets during the training process
    model.eval()
    correct = 0
    total_loss = 0

    with torch.no_grad(): # Loop through the data in the loader, get the model's output for each batch, calculate the negative log likelihood loss, and count the number of correct predictions to compute overall performance metrics such as average loss and accuracy
        for data, target in loader: # Get the model's output for the input data, calculate the negative log likelihood loss, and determine how many predictions are correct to compute overall performance metrics
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy


def train_network(): #  Train the model on the MNIST dataset, evaluate its performance over multiple epochs, and generate training/testing loss and accuracy plots to analyze the model's learning progress and generalization capabilities
    batch_size = 64
    epochs = 5
    learning_rate = 0.01

    transform = transforms.Compose([transforms.ToTensor()])

    train_loader = torch.utils.data.DataLoader( # Load MNIST training dataset with specified batch size and transformations, which allows for efficient batching and shuffling of the training data during the training process
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader( # Load MNIST test dataset with specified batch size and transformations, which allows for efficient batching of the test data during evaluation of the model's performance on unseen data
        datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=False
    )

    model = MyNetwork()
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
    for epoch in range(epochs): # Train the model for a specified number of epochs, evaluating its performance on both training and testing datasets at the end of each epoch to track the learning progress and generalization capabilities of the model on the MNIST dataset
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader): # Loop through the training data batches, perform a forward pass to get predictions, calculate the loss using negative log likelihood, backpropagate the loss, and update the model parameters using the optimizer, while accumulating the total number of examples seen to track training progress and compute average loss at the end of each epoch
            optimizer.zero_grad()
            output = model(data)

            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            examples_seen += len(data)

            if batch_idx % 100 == 0: # Every 100 batches, record the current training loss and the number of examples seen, which allows for monitoring the training progress and visualizing the loss curve over time
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
    torch.save(model.state_dict(), "models/mnist_cnn.pth")

    plot_metrics(train_steps, train_losses, test_steps, test_losses,
                 train_accs, test_accs)


def plot_metrics(train_steps, train_losses, test_steps, test_losses,
                 train_accs, test_accs): # Generate and save plots for training/testing loss and accuracy over time to visualize the model's performance during training, which allows for analysis of the learning progress and generalization capabilities of the model on the MNIST dataset

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


def main(argv): # Main function to run the training process for the model on the MNIST dataset, which includes loading the data, initializing the model and optimizer, training the model over multiple epochs while evaluating its performance on both training and testing datasets, and generating plots to visualize the learning progress and generalization capabilities of the model
    train_network()


if __name__ == "__main__":
    main(sys.argv)