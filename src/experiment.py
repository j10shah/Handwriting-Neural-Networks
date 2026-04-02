# Name: Jay Shah
# Date: 03/31/2026
# Purpose: Task 5 - Run automated experiments to evaluate the impact of
#          different hyperparameters on CNN performance.
# Run: python src/experiment.py

import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from network import MyNetwork

class CustomNet(nn.Module): # Custom CNN architecture with variable filters and dropout
    def __init__(self, f1=10, f2=20, dropout=0.5): # Initialize the network with specified filters and dropout
        super().__init__()

        self.conv1 = nn.Conv2d(1, f1, 5) # First convolutional layer with f1 filters
        self.conv2 = nn.Conv2d(f1, f2, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(f2 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x): # Define the forward pass through the
        x = self.pool(torch.relu(self.conv1(x))) # Apply first convolution, ReLU, and pooling
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return torch.log_softmax(x, dim=1)

def run_experiment(f1, f2, dropout, batch_size): # Run a single experiment with specified hyperparameters
    transform = transforms.Compose([transforms.ToTensor()])

    train_loader = torch.utils.data.DataLoader( # Load MNIST training dataset with specified batch size
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader( # Load MNIST test dataset
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000
    )

    model = CustomNet(f1, f2, dropout) # Create an instance of the custom network with specified hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    for epoch in range(3): # Train for a fixed number of epochs (3) to evaluate performance
        model.train()
        for data, target in train_loader: # Loop through training data batches
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start_time

    model.eval()
    correct = 0

    with torch.no_grad(): # Evaluate the trained model on the test dataset
        for data, target in test_loader: # Loop through test data batches
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    accuracy = correct / 10000

    return accuracy, train_time # Return the test accuracy and training time for this experiment


def main(argv): # Main function to run multiple experiments with different hyperparameter combinations
    results = []

    filters = [10, 20, 30]
    dropouts = [0.0, 0.25, 0.5]
    batch_sizes = [32, 64, 128]

    for f in filters: # Loop through different filter sizes for the convolutional layers
        for d in dropouts: # Loop through different dropout rates
            for b in batch_sizes: # Loop through different batch sizes for training
                print(f"\nRunning: filters={f}, dropout={d}, batch={b}")

                acc, t = run_experiment(f, f*2, d, b)

                results.append((f, d, b, acc, t))

                print(f"Accuracy: {acc:.4f}, Time: {t:.2f}s")

    print("\n--- FINAL RESULTS ---")
    for r in results: # Loop through the collected results and print them in a readable format
        print(r)


if __name__ == "__main__":
    main(sys.argv)