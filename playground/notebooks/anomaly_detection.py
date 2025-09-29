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
# # Anomaly Detection with Autoencoders
#
# This notebook demonstrates anomaly detection using PyTorch autoencoders on the MNIST dataset. We'll train the model on normal digits (excluding one chosen digit) and test it on the excluded digit as anomalies.

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Configuration
#
# Set the digit to be treated as anomaly. You can change this to experiment with different digits (0-9).

# %%
# Configuration: Choose which digit to treat as anomaly
anomalous_digit = 5  # Change this to any digit 0-9 to experiment

print(f"Anomalous digit set to: {anomalous_digit}")
print(f"Normal digits will be: {[i for i in range(10) if i != anomalous_digit]}")

# %% [markdown]
# ## Load MNIST Dataset
#
# We'll load the MNIST dataset and create two datasets:
# 1. **Normal dataset**: Contains all digits except the configured anomalous digit
# 2. **Test dataset**: Contains all digits (including the anomalous digit)

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


# Create normal dataset (exclude anomalous digit)
def create_normal_dataset(dataset, exclude_digit):
    """Create dataset excluding specified digit"""
    indices = []
    for i, (_, label) in enumerate(dataset):
        if label != exclude_digit:
            indices.append(i)
    return Subset(dataset, indices)


# Create normal training dataset (excluding anomalous digit)
normal_train_dataset = create_normal_dataset(train_dataset, exclude_digit=anomalous_digit)

print(f"Original training dataset size: {len(train_dataset)}")
print(f"Normal training dataset size: {len(normal_train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create data loaders
batch_size = 128
normal_dataloader = DataLoader(normal_train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %% [markdown]
# ## Visualize Sample Data
#
# Let's look at some sample images from our normal dataset to understand what the autoencoder will learn to reconstruct.


# %%
# Visualize sample normal images
def visualize_samples(dataloader, title="Sample Images", n_samples=8):
    """Visualize sample images from dataloader"""
    fig, axes = plt.subplots(1, n_samples, figsize=(12, 2))
    fig.suptitle(title)

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
visualize_samples(normal_dataloader, f"Normal Training Samples (No Digit {anomalous_digit})")


# %% [markdown]
# ## Define Autoencoder Architecture
#
# We'll create a simple autoencoder with:
# - **Encoder**: Compresses 28×28 images to a lower-dimensional representation
# - **Decoder**: Reconstructs the original image from the compressed representation


# %%
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        # Encoder: 28*28 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

        # Decoder: latent_dim -> 28*28
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Tanh(),  # Output in [-1, 1] to match normalization
        )

    def forward(self, x):
        # Encode
        latent = self.encoder(x)

        # Decode
        reconstructed = self.decoder(latent)
        reconstructed = reconstructed.view(-1, 1, 28, 28)  # Reshape to image format

        return reconstructed

    def encode(self, x):
        """Get the latent representation"""
        return self.encoder(x)


# Initialize model
latent_dim = 64
model = Autoencoder(latent_dim=latent_dim).to(device)

# Print model architecture
print("Autoencoder Architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## Training Setup
#
# Now let's set up the training components: loss function, optimizer, and training loop.

# %%
# Training setup
learning_rate = 0.001
loss_fn = nn.MSELoss()  # Mean Squared Error for reconstruction
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Loss function: {loss_fn}")
print(f"Optimizer: {optimizer}")
print(f"Learning rate: {learning_rate}")

# %% [markdown]
# ## Training Loop
#
# Here's the main training loop as you requested. We'll train the autoencoder on normal images only.

# %%
# -------------------------
# Training loop
# -------------------------
num_epochs = 30
train_losses = []

print("Starting training...")
print("=" * 50)

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (batch_X, _batch_y) in enumerate(normal_dataloader):
        # Move data to device
        batch_X = batch_X.to(device)

        # Forward pass: compute outputs and build computation graph
        # Note: For autoencoders, we reconstruct the input, so target is batch_X
        reconstructed = model(batch_X)

        # Compute scalar loss (reconstruction error)
        loss = loss_fn(reconstructed, batch_X)

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
# Let's plot the training loss to see how well our autoencoder learned.

# %%
# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, "b-o", linewidth=2, markersize=8)
plt.title("Autoencoder Training Loss", fontsize=14, fontweight="bold")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Average Loss (MSE)", fontsize=12)
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
# ## Visualize Reconstructions
#
# Let's see how well our trained autoencoder reconstructs normal images.


# %%
def visualize_reconstructions(model, dataloader, title="Reconstructions", n_samples=8):
    """Visualize original vs reconstructed images"""
    model.eval()

    with torch.no_grad():
        data_iter = iter(dataloader)
        batch_X, batch_y = next(data_iter)
        batch_X = batch_X.to(device)

        # Get reconstructions
        reconstructed = model(batch_X)

        # Move back to CPU for visualization
        batch_X = batch_X.cpu()
        reconstructed = reconstructed.cpu()

        # Create subplots
        fig, axes = plt.subplots(2, n_samples, figsize=(15, 4))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        for i in range(n_samples):
            # Original images
            orig_img = batch_X[i].squeeze().numpy()
            orig_img = (orig_img + 1) / 2  # Denormalize

            axes[0, i].imshow(orig_img, cmap="gray")
            axes[0, i].set_title(f"Original\nLabel: {batch_y[i].item()}")
            axes[0, i].axis("off")

            # Reconstructed images
            recon_img = reconstructed[i].squeeze().numpy()
            recon_img = (recon_img + 1) / 2  # Denormalize

            axes[1, i].imshow(recon_img, cmap="gray")
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.show()


# Visualize reconstructions on normal data
visualize_reconstructions(model, normal_dataloader, "Normal Images - Reconstructions")


# %% [markdown]
# ## Anomaly Detection
#
# Now let's test our autoencoder on the full test dataset, including the anomalous digit.


# %%
def compute_reconstruction_errors(model, dataloader):
    """Compute reconstruction errors for all images in dataloader"""
    model.eval()
    errors = []
    labels = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)

            # Get reconstructions
            reconstructed = model(batch_X)

            # Compute reconstruction error for each image
            batch_errors = torch.mean((batch_X - reconstructed) ** 2, dim=(1, 2, 3))

            errors.extend(batch_errors.cpu().numpy())
            labels.extend(batch_y.numpy())

    return np.array(errors), np.array(labels)


# Compute reconstruction errors on test set
print("Computing reconstruction errors on test set...")
test_errors, test_labels = compute_reconstruction_errors(model, test_dataloader)

# Separate normal vs anomaly errors
normal_mask = test_labels != anomalous_digit  # Normal digits
anomaly_mask = test_labels == anomalous_digit  # Anomaly digit

normal_errors = test_errors[normal_mask]
anomaly_errors = test_errors[anomaly_mask]

print(f"Normal images: {len(normal_errors)}")
print(f"Anomaly images (digit {anomalous_digit}): {len(anomaly_errors)}")
print(f"Normal error mean: {normal_errors.mean():.4f} ± {normal_errors.std():.4f}")
print(f"Anomaly error mean: {anomaly_errors.mean():.4f} ± {anomaly_errors.std():.4f}")

# %% [markdown]
# ## Visualize Reconstruction Error Distribution

# %%
# Plot reconstruction error distributions
plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
normal_digits_list = [str(i) for i in range(10) if i != anomalous_digit]
plt.hist(
    normal_errors,
    bins=50,
    alpha=0.7,
    label=f"Normal (digits {','.join(normal_digits_list)})\nn={len(normal_errors)}",
    color="blue",
    density=True,
)
plt.hist(
    anomaly_errors,
    bins=50,
    alpha=0.7,
    label=f"Anomaly (digit {anomalous_digit})\nn={len(anomaly_errors)}",
    color="red",
    density=True,
)
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Density")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.grid(True, alpha=0.3)

# Box plot
plt.subplot(1, 2, 2)
box_data = [normal_errors, anomaly_errors]
normal_digits_str = ",".join([str(i) for i in range(10) if i != anomalous_digit])
box_labels = [f"Normal\n({normal_digits_str})", f"Anomaly\n({anomalous_digit})"]
box_plot = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
box_plot["boxes"][0].set_facecolor("lightblue")
box_plot["boxes"][1].set_facecolor("lightcoral")
plt.ylabel("Reconstruction Error (MSE)")
plt.title("Reconstruction Error Box Plot")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical summary
print("\nStatistical Summary:")
print(f"Normal errors - Mean: {normal_errors.mean():.4f}, Median: {np.median(normal_errors):.4f}")
print(f"Anomaly errors - Mean: {anomaly_errors.mean():.4f}, Median: {np.median(anomaly_errors):.4f}")
print(f"Separation ratio: {anomaly_errors.mean() / normal_errors.mean():.2f}x")

# %% [markdown]
# ## Evaluate Anomaly Detection Performance
#
# Let's evaluate how well our autoencoder can distinguish anomalies using ROC analysis.

# %%
# Create binary labels for ROC analysis (1 = anomaly, 0 = normal)
y_true = (test_labels == anomalous_digit).astype(int)  # 1 for anomalous digit, 0 for others
y_scores = test_errors  # Higher reconstruction error = more likely to be anomaly

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc_score = roc_auc_score(y_true, y_scores)

# Plot ROC curve
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Anomaly Detection")
plt.legend()
plt.grid(True, alpha=0.3)

# Find optimal threshold (closest to top-left corner)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

plt.subplot(1, 2, 2)
plt.plot(thresholds, tpr, label="True Positive Rate", linewidth=2)
plt.plot(thresholds, fpr, label="False Positive Rate", linewidth=2)
plt.axvline(optimal_threshold, color="red", linestyle="--", label=f"Optimal Threshold = {optimal_threshold:.4f}")
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("TPR and FPR vs Threshold")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"AUC Score: {auc_score:.3f}")
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print("At optimal threshold:")
print(f"  True Positive Rate: {tpr[optimal_idx]:.3f}")
print(f"  False Positive Rate: {fpr[optimal_idx]:.3f}")

# Calculate performance metrics at optimal threshold
predictions = (test_errors > optimal_threshold).astype(int)
tp = np.sum((predictions == 1) & (y_true == 1))
tn = np.sum((predictions == 0) & (y_true == 0))
fp = np.sum((predictions == 1) & (y_true == 0))
fn = np.sum((predictions == 0) & (y_true == 1))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\nPerformance Metrics:")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
print(f"  F1-Score: {f1_score:.3f}")

# %% [markdown]
# ## Summary and Next Steps
#
# ### What We've Accomplished:
#
# 1. **Built an Autoencoder**: Created a simple neural network that learns to compress and reconstruct normal images (all digits except the anomalous one)
#
# 2. **Trained on Normal Data**: The model learned to minimize reconstruction error on normal digits only
#
# 3. **Anomaly Detection**: Used reconstruction error as an anomaly score - higher error indicates potential anomaly
#
# 4. **Evaluated Performance**: Analyzed how well the model distinguishes the anomalous digit from normal digits
#
# ### Key Insights:
# - Autoencoders learn to reconstruct what they've seen during training
# - Images that differ significantly from training data result in higher reconstruction errors
# - This makes them effective for detecting anomalies in a self-supervised manner
#
# ### Potential Improvements:
# 1. **Better Architectures**: Try convolutional autoencoders (better for images)
# 2. **Variational Autoencoders (VAEs)**: Add probabilistic components
# 3. **Different Anomaly Classes**: Test with other digits or different types of anomalies
# 4. **Ensemble Methods**: Combine multiple models for better performance
# 5. **Threshold Optimization**: Fine-tune detection thresholds for specific use cases

# %% [markdown]
# ## Quick Reference: Performance Metrics
#
# ### Confusion Matrix Components:
# - **True Positives (TP)**: Correctly identified anomalies
# - **True Negatives (TN)**: Correctly identified normal samples
# - **False Positives (FP)**: Normal samples incorrectly flagged as anomalies
# - **False Negatives (FN)**: Anomalies missed by the detector
#
# ### Key Metrics:
#
# **True Positive Rate (Sensitivity/Recall):**
# $$TPR = \frac{TP}{TP + FN} = \frac{\text{Correctly detected anomalies}}{\text{Total actual anomalies}}$$
#
# **False Positive Rate:**
# $$FPR = \frac{FP}{FP + TN} = \frac{\text{Normal samples flagged as anomalies}}{\text{Total actual normal samples}}$$
#
# **Accuracy:**
# $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Correct predictions}}{\text{Total predictions}}$$
#
# **Precision:**
# $$Precision = \frac{TP}{TP + FP} = \frac{\text{Correctly detected anomalies}}{\text{Total predicted anomalies}}$$
#
# **Recall (same as TPR):**
# $$Recall = \frac{TP}{TP + FN} = \frac{\text{Correctly detected anomalies}}{\text{Total actual anomalies}}$$
#
# **F1-Score (Harmonic mean of Precision and Recall):**
# $$F1 = 2 \cdot \frac{Precision \times Recall}{Precision + Recall} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
#
# ### Interpretation:
# - **High Precision**: Few false alarms (good for systems where false positives are costly)
# - **High Recall**: Few missed anomalies (good for safety-critical systems)
# - **High F1-Score**: Good balance between precision and recall
# - **AUC**: Area Under ROC Curve - measures overall discriminative ability (0.5 = random, 1.0 = perfect)

# %% [markdown]
# ## Experiment with Different Anomalies
#
# Now you can easily experiment with different digits as anomalies! Simply:
#
# 1. **Change the `anomalous_digit` variable** in the configuration cell (cell 3)
# 2. **Re-run all cells** to see how the autoencoder performs with different anomaly types
#
# ### Suggested Experiments:
# - **Digit 0**: Circular shape vs other digits
# - **Digit 1**: Thin vertical line vs other digits
# - **Digit 4**: Complex shape with crossing lines
# - **Digit 7**: Simple angular shape
# - **Digit 9**: Similar to 6 when rotated - interesting case!
#
# ### What to observe:
# - How does the **AUC score** change with different anomalies?
# - Which digits are **easier/harder** for the autoencoder to detect as anomalies?
# - Are there **false positives** - normal digits that look similar to your chosen anomaly?
#
# This flexibility makes it easy to understand how autoencoder-based anomaly detection performs across different types of visual patterns!
