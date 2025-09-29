# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: mono
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Digit Classification with CNN
#
# This notebook demonstrates digit classification using PyTorch CNNs on the MNIST dataset. We'll train the model to classify handwritten digits (0-9) using a convolutional neural network.

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Load MNIST Dataset
#
# We'll load the MNIST dataset for digit classification:
# 1. **Training dataset**: Contains all 60,000 training images with labels
# 2. **Test dataset**: Contains 10,000 test images for evaluation

# %%
# Define transforms
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ]
)

# Download MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)


print(f"Original training dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create data loaders
batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %% [markdown]
# ## Visualize Sample Data
#
# Let's look at some sample images from our training dataset to understand what the CNN will learn to classify.


# %%
# Visualize sample normal images
def visualize_samples(dataloader, n_samples=8):
    """Visualize sample images from dataloader"""
    fig, axes = plt.subplots(1, n_samples, figsize=(12, 2))
    fig.suptitle("Sample Images")

    data_iter = iter(dataloader)
    batch_X, batch_y = next(data_iter)

    for i in range(n_samples):
        img = batch_X[i].squeeze().cpu().numpy()
        img = (img + 1) / 2  # Denormalize from [-1,1] to [0,1]

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {batch_y[i].item()}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# Show sample normal images (should not contain the anomalous digit)
visualize_samples(train_dataloader)


# %% [markdown]
# ## Define CNN Architecture
#
# We'll create a convolutional neural network with:
# - **Feature Extraction**: Convolutional layers to detect spatial patterns
# - **Classification Head**: Fully connected layers to map features to digit classes (0-9)


# %%
class DigitClassifierCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, 3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Conv2d(64, 128, 3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 7x7 -> 1x1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),  # Reduced from 0.5 to 0.25 (25%)
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Initialize model
num_classes = 10
model = DigitClassifierCNN(num_classes=num_classes).to(device)

# Print model architecture
print("Digit Classifier CNN Architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## Training Setup
#
# Now let's set up the training components: loss function, optimizer, and training loop.

# %%
# Training setup
learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Loss function: {loss_fn}")
print(f"Optimizer: {optimizer}")
print(f"Learning rate: {learning_rate}")

# %% [markdown]
# ## Training Loop
#
# Here's the main training loop for classification. We'll train the CNN to predict digit labels.

# %%
# -------------------------
# Training loop
# -------------------------
num_epochs = 20
train_losses = []

print("Starting training...")
print("=" * 50)

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (batch_X, batch_y) in enumerate(train_dataloader):
        # Move data to device
        batch_X = batch_X.to(device)

        # Move labels to device too
        batch_y = batch_y.to(device)

        # Forward pass: compute outputs and build computation graph
        logits = model(batch_X)

        # Compute scalar loss (classification error)
        loss = loss_fn(logits, batch_y)

        # Zero out old gradients (they accumulate by default)
        optimizer.zero_grad()

        # Backward pass: autograd computes gradients of loss w.r.t. parameters
        loss.backward()

        # Parameter update: optimizer applies update rule using gradients
        optimizer.step()

        # Accumulate loss for epoch average
        epoch_loss += loss.item()
        num_batches += 1

        # Report current batch loss every 50 batches
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Calculate and store average epoch loss
    avg_epoch_loss = epoch_loss / num_batches
    train_losses.append(avg_epoch_loss)

    print(f"Epoch {epoch + 1}/{num_epochs} Complete - Average Loss: {avg_epoch_loss:.4f}")
    print("-" * 40)

print("Training completed!")
print("=" * 50)

# %% [markdown]
# ## Visualize Training Progress
#
# Let's plot the training loss to see how well our CNN learned.

# %%
# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, "b-o", linewidth=2, markersize=8)
plt.title("CNN Training Loss", fontsize=14, fontweight="bold")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Average Loss (CrossEntropy)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(1, num_epochs + 1))

# Add value annotations
for i, loss in enumerate(train_losses):
    plt.annotate(f"{loss:.4f}", (i + 1, loss), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)

plt.tight_layout()
plt.show()

print(f"Final training loss: {train_losses[-1]:.4f}")
print(f"Loss reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")


# %% [markdown]
# ## Evaluate Model Performance
#
# Let's test our trained CNN on the test dataset and compute classification accuracy.


# %%
# Evaluate model on test set
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Get predictions
            logits = model(batch_X)
            predictions = torch.argmax(logits, dim=1)

            # Accumulate statistics
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    accuracy = correct / total
    return accuracy, np.array(all_predictions), np.array(all_targets)


# Evaluate the model
accuracy, predictions, targets = evaluate_model(model, test_dataloader)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Correct predictions: {(predictions == targets).sum()} / {len(targets)}")

# %% [markdown]
# ## Detailed Classification Analysis
#
# Let's dive deeper into the classification performance with confusion matrix and per-class metrics.

# %%
# Make sure we have the evaluation results
if "targets" not in locals() or "predictions" not in locals():
    print("Running evaluation first...")
    accuracy, predictions, targets = evaluate_model(model, test_dataloader)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# 1. Confusion Matrix
cm = confusion_matrix(targets, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix - Digit Classification", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Digit", fontsize=12)
plt.ylabel("True Digit", fontsize=12)
plt.tight_layout()
plt.show()

# 2. Per-class accuracy
per_class_accuracy = np.diag(cm) / cm.sum(axis=1)
print("Per-Class Accuracy:")
for digit, acc in enumerate(per_class_accuracy):
    print(f"  Digit {digit}: {acc:.3f} ({acc * 100:.1f}%)")

# 3. Classification Report
print("\nDetailed Classification Report:")
print(classification_report(targets, predictions, target_names=[f"Digit {i}" for i in range(10)]))


# %% [markdown]
# ## Misclassified Examples
#
# Let's visualize some examples that the model got wrong to understand failure patterns.


# %%
def visualize_misclassified_examples(model, test_loader, num_examples=12):
    """
    Visualize examples that were misclassified by the model.
    Shows the image, true label, predicted label, and confidence.
    """
    model.eval()
    device = next(model.parameters()).device
    misclassified_examples = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Get predictions and probabilities
            logits = model(batch_X)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            # Find misclassified examples
            misclassified_mask = predictions != batch_y

            if misclassified_mask.any():
                misclassified_indices = torch.where(misclassified_mask)[0]

                for idx in misclassified_indices:
                    example = {
                        "image": batch_X[idx].cpu(),
                        "true_label": batch_y[idx].item(),
                        "predicted_label": predictions[idx].item(),
                        "confidence": probabilities[idx][predictions[idx]].item(),
                    }
                    misclassified_examples.append(example)

                    if len(misclassified_examples) >= num_examples:
                        break

            if len(misclassified_examples) >= num_examples:
                break

    # Create visualization
    rows = (num_examples + 3) // 4  # 4 images per row
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes

    for i, example in enumerate(misclassified_examples):
        if i >= len(axes):
            break

        ax = axes[i]

        # Display image
        img = example["image"].squeeze()
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        # Add title with true vs predicted labels and confidence
        title = f"True: {example['true_label']}, Pred: {example['predicted_label']}\n"
        title += f"Conf: {example['confidence']:.2f}"
        ax.set_title(title, fontsize=10, color="red")

    # Hide unused subplots
    for i in range(len(misclassified_examples), len(axes)):
        axes[i].axis("off")

    plt.suptitle("Misclassified Examples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# Visualize misclassified examples
visualize_misclassified_examples(model, test_dataloader, num_examples=12)
