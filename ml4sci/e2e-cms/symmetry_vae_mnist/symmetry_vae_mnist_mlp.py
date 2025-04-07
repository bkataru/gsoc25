# -*- coding: utf-8 -*-
"""
End-to-End VAE for Symmetry Discovery on Rotated MNIST (Digits 1 & 2).

Implements:
1. Rotated MNIST Dataset Preparation (Digits 1 & 2).
2. Convolutional VAE Training with KL Annealing & Early Stopping.
3. Supervised Symmetry (Rotation) Discovery in Latent Space.
4. Unsupervised Symmetry Discovery (Oracle-Preserving Latent Flows).
5. Bonus: Rotation-Invariant Classifier using discovered symmetry (using EMLP structure with PyTorch placeholder).

Improvements v3:
- Added comprehensive visualization functions (loss curves, latent space, reconstructions, transformations).
- Plots are saved to a 'plots' subdirectory.
- Enhanced logging for clarity (model saving, final scores, stopping reasons).
- Confirmed and clarified early stopping for unsupervised phase.
- Optimized visualization for GPU memory constraints (inference on GPU, plotting on CPU).

Improvements v2:
- Stabilize Unsupervised Training:
    - Added tanh activation + learnable scaling to GeneratorNet output.
    - Added Gradient Clipping to flow and generator optimizers.
    - Changed generator regularization loss to encourage norm approx 1.0.
- Added explicit NaN checks in unsupervised loop.

Improvements v1:
- Uses a Convolutional VAE (ConvVAE) instead of MLP-VAE.
- Increased default training epochs.
- Added ReduceLROnPlateau learning rate schedulers.
- Implemented KL annealing for VAE training.
- Added basic early stopping based on test set performance.
- Increased default number of flow layers.
- Adjusted default unsupervised generator loss weight.
- Integrated logging and more type hints.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF # Use functional for tensor rotation
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import math
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from torch import Tensor
import logging as lg
import time # For timestamping plots

# Setup basic logging
lg.basicConfig(level=lg.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')
logging = lg.getLogger("main")

# --- EMLP / JAX Integration ---
EMLP_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp
    # Configure JAX for potential GPU sharing issues with PyTorch
    # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # Note: Careful GPU memory management is needed if running both on one GPU.
    import emlp.nn.pytorch as enn # Use PyTorch EMLP implementation if available
    import emlp.reps
    import emlp.groups
    EMLP_AVAILABLE = True
    logging.info("EMLP library found and imported successfully.")
except ImportError:
    logging.warning("'emlp' or 'jax' library not found. Bonus Task 5 (Rotation Invariant Network) will be skipped.")
    logging.warning("Install using: pip install emlp jax jaxlib")
except Exception as e:
    logging.error(f"Error importing EMLP/JAX: {e}. Bonus Task 5 will be skipped.")

# --- Configuration ---
def parse_args():
    parser = argparse.ArgumentParser(description="VAE Symmetry Discovery on Rotated MNIST")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # Training Epochs
    parser.add_argument('--epochs_vae', type=int, default=50, help='Max Epochs for VAE training')
    parser.add_argument('--epochs_transform_mlp', type=int, default=25, help='Max Epochs for Supervised Transform MLP training')
    parser.add_argument('--epochs_oracle', type=int, default=15, help='Max Epochs for Oracle Classifier training')
    parser.add_argument('--epochs_unsup', type=int, default=30, help='Max Epochs for Unsupervised Symmetry Discovery')
    parser.add_argument('--epochs_invariant_cls', type=int, default=30, help='Max Epochs for Invariant Classifier training')
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--latent_dim', type=int, default=20, help='Dimension of the VAE latent space')
    parser.add_argument('--hidden_dim_mlp', type=int, default=128, help='Hidden dimension for helper MLPs')
    parser.add_argument('--num_rotations', type=int, default=12, help='Number of rotations (360 / num_rotations degrees step)')
    parser.add_argument('--digits', nargs='+', type=int, default=[1, 2], help='Digits to use from MNIST')
    # Loss Weights
    parser.add_argument('--beta_kl', type=float, default=1.0, help='Max weight for KL divergence term in VAE loss (annealed)')
    parser.add_argument('--beta_unsup_oracle', type=float, default=1.0, help='Weight for Oracle Preservation loss in Unsupervised task')
    parser.add_argument('--beta_unsup_gen', type=float, default=1.0, help='Weight for Generator loss/regularization (norm ~ 1.0) in Unsupervised task')
    # Other Params
    parser.add_argument('--num_flow_layers', type=int, default=8, help='Number of layers in the SimpleFlow model')
    parser.add_argument('--eps_unsup', type=float, default=0.1, help='Perturbation size epsilon for unsupervised symmetry check')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Max norm for gradient clipping in unsupervised training')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--scheduler_factor', type=float, default=0.1, help='Factor for ReduceLROnPlateau scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Patience for ReduceLROnPlateau scheduler')
    # Visualization
    parser.add_argument('--plot_subset_size', type=int, default=1000, help='Number of samples for latent space plots (t-SNE/PCA)')
    parser.add_argument('--latent_plot_method', type=str, default='pca', choices=['tsne', 'pca'], help='Method for latent space dimensionality reduction')
    # Execution Control
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save models and plots')
    parser.add_argument('--skip_training', action='store_true', help='Skip training steps if models exist')
    parser.add_argument('--run_bonus', action='store_true', help='Run the bonus task (requires emlp)')

    args = parser.parse_args()
    if not 0 < args.beta_kl:
         raise ValueError("--beta_kl must be positive")
    args.rotation_step = 360.0 / args.num_rotations
    # Create directories
    args.plot_dir = os.path.join(args.save_dir, 'plots')
    args.model_dir = os.path.join(args.save_dir, 'models')
    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    logging.info(f"Running with arguments: {args}")
    return args

# --- Utility Functions ---
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Set random seed to {seed}")

def save_model(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    # Log saving, but avoid excessive logging inside loops (handled by early stopping logic)
    # logging.info(f"Saved model state to {path}")

def load_model(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(path):
        logging.error(f"Model file not found at {path}. Cannot load.")
        raise FileNotFoundError(f"Model file not found at {path}")
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        logging.info(f"Loaded model from {path} to {device}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {path}: {e}")
        raise

# --- Visualization Functions ---

def plot_losses(history: Dict[str, List[float]], title: str, filename: str):
    """ Plots training and validation losses. """
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    if 'train_loss' in history:
        plt.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
    if 'test_loss' in history:
        plt.plot(epochs, history['test_loss'], 'ro-', label='Test Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved loss plot to {filename}")

def plot_accuracies(history: Dict[str, List[float]], title: str, filename: str):
    """ Plots training and validation accuracies. """
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(history.get('train_acc', [])) + 1)
    if 'train_acc' in history:
        plt.plot(epochs, history['train_acc'], 'bo-', label='Train Accuracy')
    if 'test_acc' in history:
        plt.plot(epochs, history['test_acc'], 'ro-', label='Test Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved accuracy plot to {filename}")

def plot_latent_space(latent_vectors: np.ndarray, labels: np.ndarray, angles: np.ndarray, method: str, title_suffix: str, filename_base: str):
    """ Plots 2D latent space using PCA or t-SNE, colored by label and angle. """
    logging.info(f"Generating {method.upper()} latent space plot...")
    if latent_vectors.shape[0] == 0:
        logging.warning("Cannot plot latent space: No latent vectors provided.")
        return

    start_time = time.time()
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    try:
        latent_2d = reducer.fit_transform(latent_vectors)
    except Exception as e:
        logging.error(f"Error during {method.upper()} fitting: {e}. Skipping plot.")
        return

    logging.info(f"{method.upper()} fitting took {time.time() - start_time:.2f} seconds.")

    # Plot colored by digit label
    plt.figure(figsize=(10, 8))
    scatter_label = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis', s=10, alpha=0.7)
    plt.title(f'Latent Space ({method.upper()}) - {title_suffix} (Colored by Digit)')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.legend(handles=scatter_label.legend_elements()[0], labels=np.unique(labels).astype(str))
    plt.grid(True)
    plt.savefig(f"{filename_base}_label.png")
    plt.close()
    logging.info(f"Saved latent space plot (by label) to {filename_base}_label.png")

    # Plot colored by angle
    plt.figure(figsize=(10, 8))
    scatter_angle = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=angles, cmap='hsv', s=10, alpha=0.7)
    plt.title(f'Latent Space ({method.upper()}) - {title_suffix} (Colored by Angle)')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    cbar = plt.colorbar(scatter_angle)
    cbar.set_label('Rotation Angle (degrees)')
    plt.grid(True)
    plt.savefig(f"{filename_base}_angle.png")
    plt.close()
    logging.info(f"Saved latent space plot (by angle) to {filename_base}_angle.png")

def plot_reconstructions(originals: Tensor, reconstructions: Tensor, filename: str, n_images: int = 8):
    """ Plots original images side-by-side with their reconstructions. """
    if originals.shape[0] == 0 or reconstructions.shape[0] == 0:
        logging.warning("Cannot plot reconstructions: No images provided.")
        return
    n_images = min(n_images, originals.size(0))
    originals = originals[:n_images].cpu()
    reconstructions = reconstructions[:n_images].cpu()
    comparison = torch.cat([originals, reconstructions])
    grid = make_grid(comparison, nrow=n_images)
    plt.figure(figsize=(15, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title('Top: Originals, Bottom: Reconstructions')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved reconstruction plot to {filename}")

def plot_latent_transformation(z_a: np.ndarray, z_b: np.ndarray, z_b_pred: Optional[np.ndarray], title: str, filename: str, dim_indices: Tuple[int, int] = (0, 1)):
    """ Plots the transformation in the first 2D of the latent space. """
    if z_a.shape[0] == 0:
        logging.warning(f"Cannot plot transformation '{title}': No data points.")
        return
    if z_a.shape[1] < max(dim_indices) + 1:
        logging.warning(f"Cannot plot transformation '{title}': Latent dim {z_a.shape[1]} too small for indices {dim_indices}.")
        return

    idx1, idx2 = dim_indices
    plt.figure(figsize=(8, 8))
    plt.scatter(z_a[:, idx1], z_a[:, idx2], alpha=0.5, s=20, label='Original (z_a / w)')
    plt.scatter(z_b[:, idx1], z_b[:, idx2], alpha=0.5, s=20, label='Target Transformed (z_b / w\')')
    if z_b_pred is not None:
        plt.scatter(z_b_pred[:, idx1], z_b_pred[:, idx2], alpha=0.5, s=20, label='Predicted Transformed (z_b_pred)')

    # Draw arrows for a subset
    num_arrows = min(50, z_a.shape[0])
    indices = np.random.choice(z_a.shape[0], num_arrows, replace=False)
    for i in indices:
        plt.arrow(z_a[i, idx1], z_a[i, idx2], z_b[i, idx1] - z_a[i, idx1], z_b[i, idx2] - z_a[i, idx2],
                  head_width=0.1, head_length=0.1, fc='gray', ec='gray', alpha=0.3)
        if z_b_pred is not None:
             plt.arrow(z_a[i, idx1], z_a[i, idx2], z_b_pred[i, idx1] - z_a[i, idx1], z_b_pred[i, idx2] - z_a[i, idx2],
                       head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.3, linestyle='--')


    plt.title(f'{title} (Dims {idx1} vs {idx2})')
    plt.xlabel(f'Latent Dim {idx1}')
    plt.ylabel(f'Latent Dim {idx2}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved latent transformation plot to {filename}")

# --- Task 1: Dataset Preparation ---
# (Dataset class and get_dataloaders function remain the same as previous version)
class RotatedMNIST(Dataset):
    """
    MNIST dataset filtered for specific digits and rotated in memory.
    Each item contains: (rotated_image, original_label, rotation_angle, original_index)
    """
    def __init__(self, mnist_dataset: Dataset, digits_to_keep: List[int], num_rotations: int = 12, rotation_step: float = 30.0):
        self.digits_to_keep = digits_to_keep
        self.num_rotations = num_rotations
        self.rotation_step = rotation_step
        self.original_indices_map: Dict[int, int] = {} # Map filtered index to original MNIST index
        self.original_labels: List[int] = []
        self.original_images: List[Tensor] = []

        logging.info(f"Filtering MNIST for digits: {digits_to_keep}...")
        idx_filtered = 0
        for i in tqdm(range(len(mnist_dataset)), desc="Filtering MNIST"):
            img, label = mnist_dataset[i]
            if label in digits_to_keep:
                self.original_indices_map[idx_filtered] = i
                self.original_labels.append(label)
                self.original_images.append(img)
                idx_filtered += 1
        logging.info(f"Found {len(self.original_images)} images for digits {digits_to_keep}.")

        self.num_original_images = len(self.original_images)
        self.total_images = self.num_original_images * self.num_rotations
        logging.info(f"Dataset will contain {self.total_images} rotated samples.")

    def __len__(self) -> int:
        return self.total_images

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, float, int]:
        if not 0 <= idx < self.total_images:
            raise IndexError("Index out of bounds")

        original_idx_in_filtered = idx // self.num_rotations
        rotation_num = idx % self.num_rotations
        angle = rotation_num * self.rotation_step

        original_image = self.original_images[original_idx_in_filtered]
        original_label = self.original_labels[original_idx_in_filtered]
        # original_mnist_idx = self.original_indices_map[original_idx_in_filtered] # Global index if needed

        # Apply rotation using torchvision.transforms.functional
        rotated_image = TF.rotate(original_image, angle)

        # Return filtered index for easier grouping in supervised task
        return rotated_image, original_label, angle, original_idx_in_filtered

def get_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    # Note: No normalization, as VAE uses Sigmoid output and BCE loss assumes [0, 1] range.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download and load original MNIST
    try:
        train_dataset_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        logging.error(f"Failed to download or load MNIST dataset: {e}")
        raise

    # Create rotated datasets for selected digits
    rotated_train_dataset = RotatedMNIST(train_dataset_full, args.digits, args.num_rotations, args.rotation_step)
    rotated_test_dataset = RotatedMNIST(test_dataset_full, args.digits, args.num_rotations, args.rotation_step)

    # Use num_workers > 0 for faster data loading, adjust based on system
    num_workers = 4 if torch.cuda.is_available() else 0 # Avoid multiprocessing issues on some systems without GPU
    persistent_workers = num_workers > 0

    train_loader = DataLoader(rotated_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
    # No shuffle for test loader to ensure consistent evaluation
    test_loader = DataLoader(rotated_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)

    # Also create a loader for the *original* digits (needed for Oracle)
    original_indices_train = [i for i, (_, label) in enumerate(train_dataset_full) if label in args.digits]
    original_indices_test = [i for i, (_, label) in enumerate(test_dataset_full) if label in args.digits]

    original_train_subset = Subset(train_dataset_full, original_indices_train)
    original_test_subset = Subset(test_dataset_full, original_indices_test)

    original_train_loader = DataLoader(original_train_subset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
    original_test_loader = DataLoader(original_test_subset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)

    logging.info("Created DataLoaders for rotated and original datasets.")
    return train_loader, test_loader, original_train_loader, original_test_loader

# --- Task 2: VAE Model (Convolutional) and Training ---
# (ConvVAE, ConvEncoder, ConvDecoder, vae_loss_function remain the same)
class ConvEncoder(nn.Module):
    """ Convolutional Encoder for VAE """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1) # Output: 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # Output: 7x7
        self.fc = nn.Linear(64 * 7 * 7, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class ConvDecoder(nn.Module):
    """ Convolutional Decoder for VAE """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # Output: 14x14
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1) # Output: 28x28

    def forward(self, z: Tensor) -> Tensor:
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 64, 7, 7) # Reshape
        x = F.relu(self.deconv1(x))
        recon_x = torch.sigmoid(self.deconv2(x)) # Sigmoid for [0, 1] output
        return recon_x

class ConvVAE(nn.Module):
    """ Convolutional Variational Autoencoder """
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, current_beta: float) -> Tensor:
    """ VAE loss = Reconstruction Loss + Beta * KL Divergence """
    # Reconstruction Loss (Binary Cross Entropy) - Sum over pixels, mean over batch
    BCE = F.binary_cross_entropy(recon_x.view(x.size(0), -1), x.view(x.size(0), -1), reduction='sum')

    # KL Divergence (analytical form for Gaussian vs Standard Normal)
    # Sum over latent dims per sample
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    # Return average loss per sample in batch
    # Ensure KLD sum is taken before multiplying by beta and averaging
    return (BCE + current_beta * KLD.sum()) / x.size(0)


def train_vae_epoch(model: ConvVAE, train_loader: DataLoader, optimizer: optim.Optimizer, epoch: int, device: torch.device, args: argparse.Namespace, current_beta: float) -> float:
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"VAE Epoch {epoch+1}/{args.epochs_vae} (beta={current_beta:.3f})", leave=False)
    for batch_idx, (data, _, _, _) in enumerate(pbar):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar, current_beta)

        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN or Inf detected in VAE loss (Epoch {epoch+1}, Batch {batch_idx}). Skipping batch.")
            continue

        loss.backward()
        # Optional: Gradient clipping for VAE
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
        train_loss += loss.item()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

    # Avoid division by zero if loader is empty or all batches were skipped
    num_batches = len(train_loader)
    avg_loss = train_loss / num_batches if num_batches > 0 else 0
    # Log average loss for the epoch
    # logging.info(f'VAE Epoch: {epoch+1} Average Train loss: {avg_loss:.4f}') # Logged in main loop
    return avg_loss

def test_vae_epoch(model: ConvVAE, test_loader: DataLoader, epoch: int, device: torch.device, args: argparse.Namespace, current_beta: float) -> float:
    model.eval()
    test_loss = 0
    num_valid_batches = 0
    with torch.no_grad():
        for data, _, _, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar, current_beta)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                 test_loss += loss.item()
                 num_valid_batches += 1

    avg_loss = test_loss / num_valid_batches if num_valid_batches > 0 else float('inf')
    # Log average test loss for the epoch
    # logging.info(f'====> VAE Epoch: {epoch+1} Average Test loss: {avg_loss:.4f}') # Logged in main loop
    return avg_loss


# --- Task 3: Supervised Symmetry Discovery ---
# (TransformMLP, create_latent_pairs, LatentPairDataset remain the same)
class TransformMLP(nn.Module):
    """ MLP to learn the transformation in latent space """
    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Consider adding dropout
        # self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z: Tensor) -> Tensor:
        h = F.relu(self.fc1(z))
        # h = self.dropout(h)
        h = F.relu(self.fc2(h))
        return self.fc3(h)

def create_latent_pairs(vae_model: ConvVAE, dataloader: DataLoader, device: torch.device, args: argparse.Namespace) -> Tuple[Tensor, Tensor]:
    """
    Generates pairs of latent vectors (z_a, z_b) where z_b corresponds
    to the rotation_step degree rotation of the image for z_a.
    Uses the 'mu' vector from the VAE encoder.
    """
    vae_model.eval()
    # Store latents grouped by the original image index
    all_latents: Dict[int, Dict[float, np.ndarray]] = {}

    logging.info("Generating latent vectors (mu) for supervised task...")
    with torch.no_grad():
        for data, _, angles, original_indices in tqdm(dataloader, desc="Encoding Latents"):
            data = data.to(device)
            # Use mu from the encoder output
            mu, _ = vae_model.encoder(data)
            mu_cpu = mu.cpu().numpy()
            angles_cpu = angles.cpu().numpy()
            original_indices_cpu = original_indices.cpu().numpy()

            for i in range(data.size(0)):
                orig_idx = original_indices_cpu[i]
                angle = angles_cpu[i]
                latent = mu_cpu[i]
                if orig_idx not in all_latents:
                    all_latents[orig_idx] = {}
                # Use rounded angle as key to avoid float precision issues
                all_latents[orig_idx][round(angle, 1)] = latent

    latent_pairs_a: List[np.ndarray] = []
    latent_pairs_b: List[np.ndarray] = []
    target_rotation = args.rotation_step

    logging.info(f"Creating latent pairs for {target_rotation} degree rotation...")
    for orig_idx in tqdm(all_latents, desc="Creating Pairs"):
        angles_present = sorted(all_latents[orig_idx].keys())
        for angle_a in angles_present:
            # Calculate target angle B, wrapping around 360
            angle_b_target = round((angle_a + target_rotation) % 360.0, 1)

            # Check if the target rotated angle exists in the dictionary for this original image
            if angle_b_target in all_latents[orig_idx]:
                z_a = all_latents[orig_idx][angle_a]
                z_b = all_latents[orig_idx][angle_b_target]
                latent_pairs_a.append(z_a)
                latent_pairs_b.append(z_b)

    if not latent_pairs_a:
        logging.warning("No latent pairs were created. Check rotation steps and data.")
        # Return empty tensors to avoid errors downstream
        return torch.empty((0, args.latent_dim)), torch.empty((0, args.latent_dim))

    logging.info(f"Created {len(latent_pairs_a)} latent pairs.")
    # Convert lists of numpy arrays to PyTorch tensors
    z_a_tensor = torch.from_numpy(np.array(latent_pairs_a)).float()
    z_b_tensor = torch.from_numpy(np.array(latent_pairs_b)).float()
    return z_a_tensor, z_b_tensor

class LatentPairDataset(Dataset):
    def __init__(self, z_a_tensor: Tensor, z_b_tensor: Tensor):
        assert len(z_a_tensor) == len(z_b_tensor)
        self.z_a = z_a_tensor
        self.z_b = z_b_tensor

    def __len__(self) -> int:
        return len(self.z_a)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.z_a[idx], self.z_b[idx]

def train_transform_mlp_epoch(transform_model: TransformMLP, pair_loader: DataLoader, optimizer: optim.Optimizer, epoch: int, device: torch.device, args: argparse.Namespace) -> float:
    transform_model.train()
    train_loss = 0
    pbar = tqdm(pair_loader, desc=f"Transform MLP Epoch {epoch+1}/{args.epochs_transform_mlp}", leave=False)
    for batch_idx, (z_a, z_b) in enumerate(pbar):
        z_a, z_b = z_a.to(device), z_b.to(device)
        optimizer.zero_grad()
        z_b_pred = transform_model(z_a)
        loss = F.mse_loss(z_b_pred, z_b) # Use MSE for regression
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

    avg_loss = train_loss / len(pair_loader) if len(pair_loader) > 0 else 0
    # logging.info(f'Transform MLP Epoch: {epoch+1} Average Train loss: {avg_loss:.4f}') # Logged in main loop
    return avg_loss

def test_transform_mlp_epoch(transform_model: TransformMLP, pair_loader: DataLoader, epoch: int, device: torch.device, args: argparse.Namespace) -> float:
    transform_model.eval()
    test_loss = 0
    if len(pair_loader) == 0: # Handle case where no test pairs were created
        logging.warning("No test pairs available for Transform MLP evaluation.")
        return float('inf')
    with torch.no_grad():
        for z_a, z_b in pair_loader:
            z_a, z_b = z_a.to(device), z_b.to(device)
            z_b_pred = transform_model(z_a)
            test_loss += F.mse_loss(z_b_pred, z_b).item()

    avg_loss = test_loss / len(pair_loader)
    # logging.info(f'====> Transform MLP Epoch: {epoch+1} Average Test loss: {avg_loss:.4f}') # Logged in main loop
    return avg_loss

# --- Task 4: Unsupervised Symmetry Discovery (Oracle-Preserving Latent Flows) ---

# 4a. Oracle Model (Simple Classifier)
# (OracleClassifier remains the same)
class OracleClassifier(nn.Module):
    """ Simple MLP classifier for original MNIST digits (e.g., 1 and 2) """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim) # Output logits

    def forward(self, x: Tensor) -> Tensor:
        # Classifier expects flattened input
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.fc3(h)
        return logits

def train_oracle_epoch(oracle_model: OracleClassifier, train_loader: DataLoader, optimizer: optim.Optimizer, epoch: int, device: torch.device, args: argparse.Namespace) -> Tuple[float, float]:
    oracle_model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Oracle Epoch {epoch+1}/{args.epochs_oracle}", leave=False)
    # Map target digits (e.g., 1, 2) to class indices (0, 1)
    target_map = {digit: i for i, digit in enumerate(args.digits)}

    for batch_idx, (data, target) in enumerate(pbar):
        # Map labels
        target = torch.tensor([target_map[t.item()] for t in target], dtype=torch.long)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        logits = oracle_model(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = logits.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)

    avg_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
    train_acc = 100. * correct / total if total > 0 else 0
    # logging.info(f'Oracle Epoch: {epoch+1} Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%') # Logged in main loop
    return avg_loss, train_acc

def test_oracle_epoch(oracle_model: OracleClassifier, test_loader: DataLoader, epoch: int, device: torch.device, args: argparse.Namespace) -> Tuple[float, float]:
    oracle_model.eval()
    test_loss = 0
    correct = 0
    total = 0
    target_map = {digit: i for i, digit in enumerate(args.digits)}
    with torch.no_grad():
        for data, target in test_loader:
            target = torch.tensor([target_map[t.item()] for t in target], dtype=torch.long)
            data, target = data.to(device), target.to(device)
            logits = oracle_model(data)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            _, predicted = logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / total if total > 0 else 0
    test_acc = 100. * correct / total if total > 0 else 0
    # logging.info(f'====> Oracle Epoch: {epoch+1} Test Loss: {avg_loss:.4f}, Test Acc: {test_acc:.2f}%') # Logged in main loop
    return avg_loss, test_acc

# 4b. Invertible Flow Model (Simple Affine Coupling Layer)
# (AffineCoupling, SimpleFlow remain the same)
class AffineCoupling(nn.Module):
    """ Simple affine coupling layer for invertible flow """
    def __init__(self, dim: int, hidden_dim: int = 128, mask_type: str = 'even'):
        super().__init__()
        self.dim = dim
        # Determine which dimensions to condition on and which to transform
        if mask_type == 'even': # Transform even indices, condition on odd
            self.condition_indices = torch.arange(1, dim, 2)
            self.transform_indices = torch.arange(0, dim, 2)
        elif mask_type == 'odd': # Transform odd indices, condition on even
            self.condition_indices = torch.arange(0, dim, 2)
            self.transform_indices = torch.arange(1, dim, 2)
        else:
            raise ValueError(f"Invalid mask_type: {mask_type}")

        if len(self.condition_indices) == 0 or len(self.transform_indices) == 0:
             raise ValueError(f"Cannot create AffineCoupling layer for dim={dim} with mask_type='{mask_type}'. "
                              "One set of indices is empty. Use dim >= 2.")

        # Networks to predict scale and translation parameters
        self.scale_net = nn.Sequential(
            nn.Linear(len(self.condition_indices), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.transform_indices))
        )
        self.translate_net = nn.Sequential(
            nn.Linear(len(self.condition_indices), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.transform_indices))
        )

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """ Computes w = f(z) and log |det J_f| """
        if z.shape[1] != self.dim:
            raise ValueError(f"Input tensor has dimension {z.shape[1]}, expected {self.dim}")

        z_cond = z[:, self.condition_indices]
        z_trans = z[:, self.transform_indices]

        # Predict scale (log_s) and translation (t) from conditioning dimensions
        log_s = torch.tanh(self.scale_net(z_cond)) # Use tanh for stability, bounded log_s
        t = self.translate_net(z_cond)

        # Apply affine transformation: w_transformed = s * z_transformed + t
        s = torch.exp(log_s) # Calculate scale s = exp(log_s)
        w_trans = s * z_trans + t

        # Construct the output tensor w
        w = z.clone()
        w[:, self.transform_indices] = w_trans

        # Log determinant is the sum of log scales
        log_det = torch.sum(log_s, dim=1)
        return w, log_det

    def inverse(self, w: Tensor) -> Tuple[Tensor, Tensor]:
        """ Computes z = f^{-1}(w) and log |det J_{f^{-1}}| """
        if w.shape[1] != self.dim:
            raise ValueError(f"Input tensor has dimension {w.shape[1]}, expected {self.dim}")

        w_cond = w[:, self.condition_indices]
        w_trans = w[:, self.transform_indices]

        # Predict scale (log_s) and translation (t) from conditioning dimensions
        log_s = torch.tanh(self.scale_net(w_cond))
        t = self.translate_net(w_cond)

        # Apply inverse affine transformation: z_transformed = (w_transformed - t) / s
        # Add epsilon for numerical stability during division
        s = torch.exp(log_s)
        z_trans = (w_trans - t) / (s + 1e-6) # Use s = exp(log_s)

        # Construct the output tensor z
        z = w.clone()
        z[:, self.transform_indices] = z_trans

        # Log determinant of inverse is sum of -log scales
        log_det_inv = torch.sum(-log_s, dim=1)
        return z, log_det_inv

class SimpleFlow(nn.Module):
    """ A simple invertible flow composed of AffineCoupling layers """
    def __init__(self, dim: int, num_layers: int = 8, hidden_dim: int = 128):
        super().__init__()
        if dim < 2:
            raise ValueError("SimpleFlow requires dimension >= 2 for AffineCoupling layers.")
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Alternate mask types for coupling layers
            mask_type = 'even' if i % 2 == 0 else 'odd'
            self.layers.append(AffineCoupling(dim, hidden_dim, mask_type))
        logging.info(f"Initialized SimpleFlow with {num_layers} AffineCoupling layers for dim={dim}.")

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """ Computes w = f(z) and log |det J_f| """
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        w = z
        for layer in self.layers:
            w, log_det = layer(w) # Apply layer
            log_det_sum += log_det # Accumulate log determinant
        return w, log_det_sum

    def inverse(self, w: Tensor) -> Tuple[Tensor, Tensor]:
        """ Computes z = f^{-1}(w) and log |det J_{f^{-1}}| """
        log_det_sum = torch.zeros(w.shape[0], device=w.device)
        z = w
        # Apply layers in reverse order for inverse
        for layer in reversed(self.layers):
            z, log_det_inv = layer.inverse(z) # Apply inverse of layer
            log_det_sum += log_det_inv # Accumulate log determinant of inverse
        return z, log_det_sum

# 4c. Generator Network (Predicts Lie Algebra Generator L)
# (GeneratorNet remains the same - includes tanh and scaling)
class GeneratorNet(nn.Module):
    """ Predicts the parameters of the Lie Algebra generator(s) """
    def __init__(self, latent_dim: int, num_generators: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_generators = num_generators # For SO(2), we expect 1 generator

        # Simplification: Assume rotation primarily affects the first 2 latent dims.
        # Predict one parameter 'a' for the SO(2) generator in the first 2 dims.
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_generators) # Raw output for tanh
        # Learnable scaling factor for the generator parameter
        self.gen_scale = nn.Parameter(torch.tensor([5.0])) # Initialize scale, e.g., 5.0

    def forward(self, w: Tensor) -> Tensor:
        """ Predicts generator parameters (e.g., 'a' for SO(2)) from flowed latent code w """
        h = F.relu(self.fc1(w))
        # Use tanh to bound the raw output, then scale
        raw_params = self.fc2(h)
        scaled_params = self.gen_scale * torch.tanh(raw_params)
        return scaled_params # Shape: (batch_size, num_generators)

    def get_generator_matrix(self, gen_params: Tensor) -> Tensor:
        """ Constructs the Lie Algebra generator matrix L from predicted params """
        # Assuming num_generators=1 and it represents SO(2) in first 2 dims
        batch_size = gen_params.size(0)
        if self.latent_dim < 2:
             return torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=gen_params.device)

        a = gen_params[:, 0] # Get the scaled 'a' parameter

        # Create the 2x2 SO(2) generator block: [[0, -a], [a, 0]]
        L_2x2 = torch.zeros(batch_size, 2, 2, device=gen_params.device)
        L_2x2[:, 0, 1] = -a
        L_2x2[:, 1, 0] = a

        # Embed into the full latent dimension matrix
        L = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=gen_params.device)
        L[:, :2, :2] = L_2x2 # Place the 2x2 block in the top-left corner
        return L # Shape: (batch_size, latent_dim, latent_dim)

# 4d. Unsupervised Training Loop
# (train_unsupervised_epoch remains the same - includes grad clipping and new gen loss)
def train_unsupervised_epoch(
    vae_model: ConvVAE, oracle_model: OracleClassifier, flow_model: SimpleFlow, gen_model: GeneratorNet,
    train_loader: DataLoader, optimizer_flow: optim.Optimizer, optimizer_gen: optim.Optimizer,
    epoch: int, device: torch.device, args: argparse.Namespace
) -> Tuple[float, float, float, Optional[np.ndarray]]:
    """ Trains one epoch for unsupervised symmetry discovery """
    vae_model.eval() # Freeze VAE
    oracle_model.eval() # Freeze Oracle
    flow_model.train() # Train Flow
    gen_model.train() # Train Generator

    total_loss_epoch = 0.0
    oracle_loss_epoch = 0.0
    generator_loss_epoch = 0.0
    last_batch_gen_params = None
    batches_processed = 0

    pbar = tqdm(train_loader, desc=f"Unsup Epoch {epoch+1}/{args.epochs_unsup}", leave=False)
    for batch_idx, (data, _, _, _) in enumerate(pbar):
        data = data.to(device)
        batch_size = data.size(0)

        optimizer_flow.zero_grad()
        optimizer_gen.zero_grad()

        # --- Forward Pass ---
        # 1. Encode: Get latent code z (sample from VAE posterior)
        with torch.no_grad():
            mu, logvar = vae_model.encoder(data)
            z = vae_model.reparameterize(mu, logvar)

        # 2. Flow: Map z to w using the invertible flow
        w, _ = flow_model(z) # log_det_f not needed for this loss formulation

        # 3. Generate: Predict Lie Algebra generator L from w
        gen_params = gen_model(w) # Now scaled and bounded
        L = gen_model.get_generator_matrix(gen_params) # (batch_size, latent_dim, latent_dim)
        last_batch_gen_params = gen_params.detach() # Keep for inspection

        # Check for NaNs in L before matrix_exp
        if torch.isnan(L).any() or torch.isinf(L).any():
            logging.warning(f"NaN or Inf detected in generator matrix L (Epoch {epoch+1}, Batch {batch_idx}). Skipping batch.")
            continue

        # 4. Transform: Apply transformation in w space: w' = exp(eps * L) * w
        eps_L = args.eps_unsup * L
        try:
            exp_eps_L = torch.matrix_exp(eps_L) # (batch_size, latent_dim, latent_dim)
        except Exception as e:
             logging.error(f"torch.matrix_exp failed (Epoch {epoch+1}, Batch {batch_idx}): {e}. Skipping batch.")
             continue

        if torch.isnan(exp_eps_L).any() or torch.isinf(exp_eps_L).any():
            logging.warning(f"NaN or Inf detected after matrix_exp (Epoch {epoch+1}, Batch {batch_idx}). Skipping batch.")
            continue

        w_reshaped = w.unsqueeze(-1)
        w_prime_reshaped = torch.bmm(exp_eps_L, w_reshaped)
        w_prime = w_prime_reshaped.squeeze(-1) # Back to (batch_size, latent_dim)

        # 5. Inverse Flow: Map w' back to z'
        z_prime, _ = flow_model.inverse(w_prime) # log_det_f_inv not needed

        if torch.isnan(z_prime).any() or torch.isinf(z_prime).any():
            logging.warning(f"NaN or Inf detected after inverse flow (Epoch {epoch+1}, Batch {batch_idx}). Skipping batch.")
            continue

        # 6. Decode: Reconstruct image x' from transformed latent z'
        x_prime_flat = vae_model.decoder(z_prime)
        x_prime = x_prime_flat # Decoder output is already in image shape (B, C, H, W)

        # --- Calculate Losses ---
        # Loss 1: Oracle Preservation Loss
        with torch.no_grad():
           logits_orig = oracle_model(data)
        logits_prime = oracle_model(x_prime) # Gradients flow back from here

        # Check logits before loss calculation
        if torch.isnan(logits_prime).any() or torch.isinf(logits_prime).any():
             logging.warning(f"NaN or Inf detected in logits_prime (Epoch {epoch+1}, Batch {batch_idx}). Skipping batch.")
             continue

        oracle_loss = F.kl_div(
            F.log_softmax(logits_prime, dim=1),
            F.softmax(logits_orig.detach(), dim=1), # Target distribution (detached)
            reduction='batchmean',
            log_target=False # Target is already probabilities
        )

        # Loss 2: Generator Regularization (Encourage norm ~ 1.0)
        gen_norm = torch.linalg.matrix_norm(L, ord='fro') # Shape: (batch_size,)
        # MSE loss towards norm 1.0
        generator_loss = torch.mean((gen_norm - 1.0)**2)

        # Total Loss
        loss = args.beta_unsup_oracle * oracle_loss + args.beta_unsup_gen * generator_loss

        # --- Backward Pass & Optimization ---
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN or Inf detected in final unsupervised loss (Epoch {epoch+1}, Batch {batch_idx}). Skipping batch.")
            continue

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=args.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(gen_model.parameters(), max_norm=args.grad_clip_norm)

        optimizer_flow.step()
        optimizer_gen.step()

        total_loss_epoch += loss.item()
        oracle_loss_epoch += oracle_loss.item()
        generator_loss_epoch += generator_loss.item()
        batches_processed += 1
        pbar.set_postfix(TotalL=loss.item(), OracleL=oracle_loss.item(), GenL=generator_loss.item())

    # Calculate average losses based on successfully processed batches
    avg_total_loss = total_loss_epoch / batches_processed if batches_processed > 0 else float('nan')
    avg_oracle_loss = oracle_loss_epoch / batches_processed if batches_processed > 0 else float('nan')
    avg_gen_loss = generator_loss_epoch / batches_processed if batches_processed > 0 else float('nan')

    # Log average losses for the epoch
    # logging.info(f'Unsup Epoch: {epoch+1} Avg Losses -> Total: {avg_total_loss:.4f}, Oracle: {avg_oracle_loss:.4f}, GenReg: {avg_gen_loss:.4f}') # Logged in main loop

    # Return the average learned generator parameters from the last successful batch for inspection
    final_gen_params = last_batch_gen_params.mean(dim=0).cpu().numpy() if last_batch_gen_params is not None else None
    return avg_total_loss, avg_oracle_loss, avg_gen_loss, final_gen_params


# --- Task 5: Bonus - Rotation Invariant Network (using EMLP) ---
# (EMLP classes and training/testing functions remain the same)
if EMLP_AVAILABLE:
    class DiscoveredSymmetryGroup(emlp.groups.Group):
        """ Custom group defined by a learned Lie Algebra generator matrix. """
        def __init__(self, learned_generator_matrix: jnp.ndarray, dim: int):
            # learned_generator_matrix should be a JAX array (num_generators, dim, dim)
            if learned_generator_matrix.ndim != 3 or learned_generator_matrix.shape[1:] != (dim, dim):
                 raise ValueError(f"Expected generator shape (num_gen, {dim}, {dim}), got {learned_generator_matrix.shape}")

            self.lie_algebra = learned_generator_matrix # Store the basis generator(s)
            self.dim = dim
            super().__init__(self.dim) # Pass dimension to base class

        def exp(self, A: jnp.ndarray) -> jnp.ndarray:
            return emlp.groups.expm(A)

        def log(self, U: jnp.ndarray) -> jnp.ndarray:
            return emlp.groups.logm(U)

        def sample(self) -> jnp.ndarray:
            coeffs = np.random.randn(self.lie_algebra.shape[0]) # Random coefficients
            A = sum(coeffs[i] * self.lie_algebra[i] for i in range(len(coeffs)))
            return self.exp(A)

        def __str__(self) -> str:
            return f"DiscoveredGroup(dim={self.dim}, num_generators={self.lie_algebra.shape[0]})"
        __repr__ = __str__

    class InvariantClassifierEMLPPlaceholder(nn.Module):
        """
        Placeholder for an EMLP Classifier invariant to the discovered symmetry.
        Uses a standard MLP for demonstration within the PyTorch loop.
        """
        def __init__(self, latent_dim: int, discovered_group: DiscoveredSymmetryGroup, num_classes: int = 2, hidden_ch: int = 64):
            super().__init__()
            self.latent_dim = latent_dim
            self.num_classes = num_classes
            self.discovered_group = discovered_group

            # --- EMLP Structure Definition (for reference) ---
            try:
                rep_in = emlp.reps.V(self.discovered_group)
                rep_out = num_classes * emlp.reps.Scalar(self.discovered_group)
                logging.info(f"Defined EMLP Input Rep: {rep_in}, Output Rep: {rep_out}")
                # self.emlp_layer = enn.EMLP(rep_in, rep_out, group=self.discovered_group, num_layers=3, ch=hidden_ch)
                # logging.info(f"EMLP Layer structure (requires JAX/Haiku or compatible backend):\n{self.emlp_layer}")
            except Exception as e:
                 logging.error(f"Failed to define EMLP representations/layer structure: {e}")
                 # self.emlp_layer = None

            # --- PyTorch Placeholder MLP ---
            logging.warning("Using standard PyTorch MLP as a placeholder for EMLP training loop.")
            self.placeholder_mlp = nn.Sequential(
                nn.Linear(latent_dim, hidden_ch),
                nn.ReLU(),
                nn.Linear(hidden_ch, hidden_ch),
                nn.ReLU(),
                nn.Linear(hidden_ch, num_classes)
            )

        def forward(self, z: Tensor) -> Tensor:
            # If self.emlp_layer was instantiated and compatible:
            # return self.emlp_layer(z)
            # Using placeholder:
            return self.placeholder_mlp(z)

    def train_invariant_classifier_epoch(
        inv_classifier: nn.Module, vae_model: ConvVAE, train_loader: DataLoader,
        optimizer: optim.Optimizer, epoch: int, device: torch.device, args: argparse.Namespace
    ) -> Tuple[float, float]:
        inv_classifier.train()
        vae_model.eval() # Freeze VAE
        train_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Invariant Cls Epoch {epoch+1}/{args.epochs_invariant_cls}", leave=False)
        target_map = {digit: i for i, digit in enumerate(args.digits)}

        for batch_idx, (data, target, _, _) in enumerate(pbar):
            target = torch.tensor([target_map[t.item()] for t in target], dtype=torch.long)
            data, target = data.to(device), target.to(device)

            # Get latent representation (mu) from VAE
            with torch.no_grad():
                mu, _ = vae_model.encoder(data)
                latent_input = mu

            optimizer.zero_grad()
            logits = inv_classifier(latent_input)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)

        avg_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = 100. * correct / total if total > 0 else 0
        # logging.info(f'Invariant Cls Epoch: {epoch+1} Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%') # Logged in main loop
        return avg_loss, train_acc

    def test_invariant_classifier_epoch(
        inv_classifier: nn.Module, vae_model: ConvVAE, test_loader: DataLoader,
        epoch: int, device: torch.device, args: argparse.Namespace, classifier_name: str = "Invariant Classifier"
    ) -> Tuple[float, float]:
        inv_classifier.eval()
        vae_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        target_map = {digit: i for i, digit in enumerate(args.digits)}
        with torch.no_grad():
            for data, target, _, _ in test_loader:
                target = torch.tensor([target_map[t.item()] for t in target], dtype=torch.long)
                data, target = data.to(device), target.to(device)

                # Get latent representation
                mu, _ = vae_model.encoder(data)
                latent_input = mu

                logits = inv_classifier(latent_input)
                test_loss += F.cross_entropy(logits, target, reduction='sum').item()
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = test_loss / total if total > 0 else 0
        test_acc = 100. * correct / total if total > 0 else 0
        # logging.info(f'====> {classifier_name} Epoch: {epoch+1} Test Loss: {avg_loss:.4f}, Test Acc: {test_acc:.2f}%') # Logged in main loop
        return avg_loss, test_acc

# --- Main Execution ---
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # --- Model Paths ---
    vae_path = os.path.join(args.model_dir, f"conv_vae_ld{args.latent_dim}.pt")
    transform_mlp_path = os.path.join(args.model_dir, f"transform_mlp_ld{args.latent_dim}.pt")
    oracle_path = os.path.join(args.model_dir, f"oracle_classifier.pt")
    flow_path = os.path.join(args.model_dir, f"flow_model_ld{args.latent_dim}.pt")
    gen_path = os.path.join(args.model_dir, f"gen_model_ld{args.latent_dim}.pt")
    inv_cls_path = os.path.join(args.model_dir, f"inv_classifier_ld{args.latent_dim}.pt")

    # --- Task 1: Data ---
    logging.info("Loading and preparing data...")
    train_loader, test_loader, original_train_loader, original_test_loader = get_dataloaders(args)

    # --- Task 2: VAE Training ---
    logging.info("\n--- Task 2: VAE Training ---")
    vae_model = ConvVAE(latent_dim=args.latent_dim).to(device)
    optimizer_vae = optim.Adam(vae_model.parameters(), lr=args.lr)
    scheduler_vae = ReduceLROnPlateau(optimizer_vae, 'min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)
    best_vae_loss = float('inf')
    vae_patience_counter = 0
    vae_history = {'train_loss': [], 'test_loss': []}
    vae_stopped_early = False

    if os.path.exists(vae_path) and args.skip_training:
        logging.info(f"Loading pre-trained VAE from {vae_path}")
        vae_model = load_model(vae_model, vae_path, device)
    else:
        logging.info("Training VAE...")
        kl_anneal_factor = 2.0 / args.epochs_vae # Rate to reach 1.0 by half epochs
        for epoch in range(args.epochs_vae):
            current_beta = min(args.beta_kl, args.beta_kl * kl_anneal_factor * epoch)
            train_loss = train_vae_epoch(vae_model, train_loader, optimizer_vae, epoch, device, args, current_beta)
            test_loss = test_vae_epoch(vae_model, test_loader, epoch, device, args, current_beta)
            vae_history['train_loss'].append(train_loss)
            vae_history['test_loss'].append(test_loss)
            logging.info(f"VAE Epoch {epoch+1}/{args.epochs_vae} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            scheduler_vae.step(test_loss)

            if test_loss < best_vae_loss:
                logging.info(f"VAE Test loss improved ({best_vae_loss:.4f} -> {test_loss:.4f}). Saving model...")
                best_vae_loss = test_loss
                save_model(vae_model, vae_path)
                vae_patience_counter = 0
            else:
                vae_patience_counter += 1
                logging.info(f"VAE Test loss did not improve. Patience: {vae_patience_counter}/{args.patience}")
                if vae_patience_counter >= args.patience:
                    logging.info(f"VAE early stopping triggered after epoch {epoch+1}.")
                    vae_stopped_early = True
                    break
        if not vae_stopped_early:
             logging.info(f"VAE training finished after {args.epochs_vae} epochs.")

        logging.info(f"Loading best VAE model with Test Loss: {best_vae_loss:.4f}")
        vae_model = load_model(vae_model, vae_path, device)

        # Plot VAE losses
        plot_losses(vae_history, 'VAE Training and Test Loss', os.path.join(args.plot_dir, 'vae_loss.png'))

        # Plot VAE reconstructions
        try:
            data_sample, _, _, _ = next(iter(test_loader))
            data_sample = data_sample.to(device)
            with torch.no_grad():
                recon_sample, _, _ = vae_model(data_sample)
            plot_reconstructions(data_sample, recon_sample, os.path.join(args.plot_dir, 'vae_reconstructions.png'))
        except Exception as e:
            logging.warning(f"Could not generate reconstruction plot: {e}")

        # Plot VAE latent space
        try:
            logging.info(f"Generating VAE latent space plot ({args.latent_plot_method.upper()})...")
            latent_vectors_list = []
            labels_list = []
            angles_list = []
            # Use a subset of the test loader for efficiency
            subset_indices = np.random.choice(len(test_loader.dataset), args.plot_subset_size, replace=False)
            subset_dataset = Subset(test_loader.dataset, subset_indices)
            subset_loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)

            with torch.no_grad():
                for data, labels, angles, _ in tqdm(subset_loader, desc="Encoding for Latent Plot"):
                    data = data.to(device)
                    mu, _ = vae_model.encoder(data)
                    latent_vectors_list.append(mu.cpu().numpy())
                    labels_list.append(labels.numpy())
                    angles_list.append(angles.numpy())

            latent_vectors = np.concatenate(latent_vectors_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            angles = np.concatenate(angles_list, axis=0)

            plot_latent_space(latent_vectors, labels, angles, args.latent_plot_method, "VAE Latent Space",
                              os.path.join(args.plot_dir, f'vae_latent_{args.latent_plot_method}'))
        except Exception as e:
            logging.warning(f"Could not generate latent space plot: {e}")


    # --- Task 3: Supervised Symmetry Discovery ---
    logging.info("\n--- Task 3: Supervised Symmetry Discovery ---")
    transform_model = TransformMLP(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim_mlp).to(device)
    optimizer_transform = optim.Adam(transform_model.parameters(), lr=args.lr)
    scheduler_transform = ReduceLROnPlateau(optimizer_transform, 'min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)
    best_transform_loss = float('inf')
    transform_patience_counter = 0
    transform_history = {'train_loss': [], 'test_loss': []}
    transform_stopped_early = False

    if os.path.exists(transform_mlp_path) and args.skip_training:
        logging.info(f"Loading pre-trained Transform MLP from {transform_mlp_path}")
        transform_model = load_model(transform_model, transform_mlp_path, device)
    else:
        z_a_train, z_b_train = create_latent_pairs(vae_model, train_loader, device, args)
        z_a_test, z_b_test = create_latent_pairs(vae_model, test_loader, device, args)

        if len(z_a_train) > 0 and len(z_a_test) > 0:
            pair_dataset_train = LatentPairDataset(z_a_train, z_b_train)
            pair_loader_train = DataLoader(pair_dataset_train, batch_size=args.batch_size, shuffle=True)
            pair_dataset_test = LatentPairDataset(z_a_test, z_b_test)
            pair_loader_test = DataLoader(pair_dataset_test, batch_size=args.batch_size, shuffle=False)

            logging.info("Training Transform MLP...")
            for epoch in range(args.epochs_transform_mlp):
                train_loss = train_transform_mlp_epoch(transform_model, pair_loader_train, optimizer_transform, epoch, device, args)
                test_loss = test_transform_mlp_epoch(transform_model, pair_loader_test, epoch, device, args)
                transform_history['train_loss'].append(train_loss)
                transform_history['test_loss'].append(test_loss)
                logging.info(f"Transform MLP Epoch {epoch+1}/{args.epochs_transform_mlp} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                scheduler_transform.step(test_loss)

                if test_loss < best_transform_loss:
                    logging.info(f"Transform MLP Test loss improved ({best_transform_loss:.4f} -> {test_loss:.4f}). Saving model...")
                    best_transform_loss = test_loss
                    save_model(transform_model, transform_mlp_path)
                    transform_patience_counter = 0
                else:
                    transform_patience_counter += 1
                    logging.info(f"Transform MLP Test loss did not improve. Patience: {transform_patience_counter}/{args.patience}")
                    if transform_patience_counter >= args.patience:
                        logging.info(f"Transform MLP early stopping triggered after epoch {epoch+1}.")
                        transform_stopped_early = True
                        break
            if not transform_stopped_early:
                logging.info(f"Transform MLP training finished after {args.epochs_transform_mlp} epochs.")

            logging.info(f"Loading best Transform MLP model with Test Loss: {best_transform_loss:.4f}")
            transform_model = load_model(transform_model, transform_mlp_path, device)

            # Plot Transform MLP losses
            plot_losses(transform_history, 'Transform MLP Training and Test Loss', os.path.join(args.plot_dir, 'transform_mlp_loss.png'))

            # Plot supervised latent transformation
            try:
                n_plot = min(args.plot_subset_size, len(z_a_test))
                indices = np.random.choice(len(z_a_test), n_plot, replace=False)
                z_a_sample = z_a_test[indices].to(device)
                with torch.no_grad():
                    z_b_pred_sample = transform_model(z_a_sample)
                plot_latent_transformation(z_a_sample.cpu().numpy(),
                                           z_b_test[indices].cpu().numpy(),
                                           z_b_pred_sample.cpu().numpy(),
                                           'Supervised Latent Transformation (Test Set)',
                                           os.path.join(args.plot_dir, 'supervised_transform.png'))
            except Exception as e:
                logging.warning(f"Could not generate supervised transform plot: {e}")

        else:
             logging.warning("Skipping Transform MLP training and plotting due to lack of latent pairs.")


    # --- Task 4: Unsupervised Symmetry Discovery ---
    logging.info("\n--- Task 4: Unsupervised Symmetry Discovery ---")
    # 4a. Train Oracle
    logging.info("Training Oracle Classifier...")
    oracle_model = OracleClassifier(input_dim=28*28, hidden_dim=args.hidden_dim_mlp, output_dim=len(args.digits)).to(device)
    optimizer_oracle = optim.Adam(oracle_model.parameters(), lr=args.lr)
    scheduler_oracle = ReduceLROnPlateau(optimizer_oracle, 'max', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True) # Maximize accuracy
    best_oracle_acc = 0.0
    oracle_patience_counter = 0
    oracle_history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    oracle_stopped_early = False

    if os.path.exists(oracle_path) and args.skip_training:
         logging.info(f"Loading pre-trained Oracle from {oracle_path}")
         oracle_model = load_model(oracle_model, oracle_path, device)
    else:
        for epoch in range(args.epochs_oracle):
            train_loss, train_acc = train_oracle_epoch(oracle_model, original_train_loader, optimizer_oracle, epoch, device, args)
            test_loss, test_acc = test_oracle_epoch(oracle_model, original_test_loader, epoch, device, args)
            oracle_history['train_loss'].append(train_loss)
            oracle_history['test_loss'].append(test_loss)
            oracle_history['train_acc'].append(train_acc)
            oracle_history['test_acc'].append(test_acc)
            logging.info(f"Oracle Epoch {epoch+1}/{args.epochs_oracle} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
            scheduler_oracle.step(test_acc)

            if test_acc > best_oracle_acc:
                logging.info(f"Oracle Test accuracy improved ({best_oracle_acc:.2f}% -> {test_acc:.2f}%). Saving model...")
                best_oracle_acc = test_acc
                save_model(oracle_model, oracle_path)
                oracle_patience_counter = 0
            else:
                oracle_patience_counter += 1
                logging.info(f"Oracle Test accuracy did not improve. Patience: {oracle_patience_counter}/{args.patience}")
                if oracle_patience_counter >= args.patience:
                    logging.info(f"Oracle early stopping triggered after epoch {epoch+1}.")
                    oracle_stopped_early = True
                    break
        if not oracle_stopped_early:
            logging.info(f"Oracle training finished after {args.epochs_oracle} epochs.")

        logging.info(f"Loading best Oracle model with Test Accuracy: {best_oracle_acc:.2f}%")
        oracle_model = load_model(oracle_model, oracle_path, device)

        # Plot Oracle losses and accuracies
        plot_losses(oracle_history, 'Oracle Classifier Training and Test Loss', os.path.join(args.plot_dir, 'oracle_loss.png'))
        plot_accuracies(oracle_history, 'Oracle Classifier Training and Test Accuracy', os.path.join(args.plot_dir, 'oracle_accuracy.png'))


    # 4b/c/d. Train Flow and Generator
    logging.info("Training Flow and Generator for Unsupervised Discovery...")
    flow_model = SimpleFlow(dim=args.latent_dim, num_layers=args.num_flow_layers, hidden_dim=args.hidden_dim_mlp).to(device)
    gen_model = GeneratorNet(latent_dim=args.latent_dim, num_generators=1, hidden_dim=args.hidden_dim_mlp // 2).to(device) # Assuming 1 generator for SO(2)

    optimizer_flow = optim.Adam(flow_model.parameters(), lr=args.lr * 0.1)
    optimizer_gen = optim.Adam(gen_model.parameters(), lr=args.lr)
    scheduler_flow = ReduceLROnPlateau(optimizer_flow, 'min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)
    scheduler_gen = ReduceLROnPlateau(optimizer_gen, 'min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)

    learned_generator_param = None
    final_gen_scale = None
    unsup_history = {'total_loss': [], 'oracle_loss': [], 'gen_loss': []}
    unsup_stopped_early = False

    if os.path.exists(flow_path) and os.path.exists(gen_path) and args.skip_training:
        logging.info(f"Loading pre-trained Flow model from {flow_path}")
        flow_model = load_model(flow_model, flow_path, device)
        logging.info(f"Loading pre-trained Generator model from {gen_path}")
        gen_model = load_model(gen_model, gen_path, device)
        # Estimate generator param and scale
        final_gen_scale = gen_model.gen_scale.item()
        # ... (code to estimate param as before) ...
        gen_model.eval()
        flow_model.eval()
        vae_model.eval()
        with torch.no_grad():
            try:
                data, _, _, _ = next(iter(train_loader))
                data = data.to(device)
                mu, logvar = vae_model.encoder(data)
                z = vae_model.reparameterize(mu, logvar)
                w, _ = flow_model(z)
                gen_params = gen_model(w)
                learned_generator_param = gen_params.mean(dim=0).cpu().numpy()
                logging.info(f"Loaded models, estimated generator param: {learned_generator_param}, scale: {final_gen_scale:.4f}")
            except Exception as e:
                logging.warning(f"Could not estimate generator param from loaded models: {e}")

    else:
        logging.info("Training Flow and Generator jointly...")
        best_unsup_loss = float('inf')
        unsup_patience_counter = 0
        for epoch in range(args.epochs_unsup):
            avg_loss, avg_oracle_loss, avg_gen_loss, current_gen_param = train_unsupervised_epoch(
                vae_model, oracle_model, flow_model, gen_model, train_loader,
                optimizer_flow, optimizer_gen, epoch, device, args
            )
            learned_generator_param = current_gen_param # Store params from last epoch
            final_gen_scale = gen_model.gen_scale.item() # Store final scale

            # Log epoch results
            logging.info(f"Unsup Epoch {epoch+1}/{args.epochs_unsup} - Avg Losses -> Total: {avg_loss:.4f}, Oracle: {avg_oracle_loss:.4f}, GenReg: {avg_gen_loss:.4f}")
            logging.info(f"Unsup Epoch {epoch+1}/{args.epochs_unsup} - Current Gen Param: {current_gen_param}, Scale: {final_gen_scale:.4f}")

            unsup_history['total_loss'].append(avg_loss)
            unsup_history['oracle_loss'].append(avg_oracle_loss)
            unsup_history['gen_loss'].append(avg_gen_loss)

            # Check if training was stable for this epoch
            if not math.isnan(avg_loss):
                scheduler_flow.step(avg_loss)
                scheduler_gen.step(avg_loss)

                # Early stopping based on total unsupervised loss
                if avg_loss < best_unsup_loss:
                    logging.info(f"Unsupervised loss improved ({best_unsup_loss:.4f} -> {avg_loss:.4f}). Saving models...")
                    best_unsup_loss = avg_loss
                    save_model(flow_model, flow_path)
                    save_model(gen_model, gen_path)
                    unsup_patience_counter = 0
                else:
                    unsup_patience_counter += 1
                    logging.info(f"Unsupervised loss did not improve. Patience: {unsup_patience_counter}/{args.patience * 2}")
                    if unsup_patience_counter >= args.patience * 2: # More patience for unsup
                        logging.info(f"Unsupervised training early stopping triggered after epoch {epoch+1}.")
                        unsup_stopped_early = True
                        break
            else:
                 logging.warning(f"Unsupervised epoch {epoch+1} resulted in NaN loss. Stopping training.")
                 unsup_stopped_early = True # Mark as stopped early due to instability
                 break # Stop if loss becomes NaN

        if not unsup_stopped_early:
             logging.info(f"Unsupervised training finished after {args.epochs_unsup} epochs.")

        # Load best models if they were saved and training didn't end in NaN
        if os.path.exists(flow_path) and os.path.exists(gen_path) and not math.isnan(best_unsup_loss):
             logging.info(f"Loading best unsupervised models with Loss: {best_unsup_loss:.4f}")
             flow_model = load_model(flow_model, flow_path, device)
             gen_model = load_model(gen_model, gen_path, device)
             # Re-estimate best generator param and scale
             final_gen_scale = gen_model.gen_scale.item()
             gen_model.eval()
             flow_model.eval()
             vae_model.eval()
             with torch.no_grad():
                 try:
                     data, _, _, _ = next(iter(train_loader))
                     data = data.to(device)
                     mu, logvar = vae_model.encoder(data)
                     z = vae_model.reparameterize(mu, logvar)
                     w, _ = flow_model(z)
                     gen_params = gen_model(w)
                     learned_generator_param = gen_params.mean(dim=0).cpu().numpy()
                     logging.info(f"Best estimated generator param: {learned_generator_param}, scale: {final_gen_scale:.4f}")
                 except Exception as e:
                     logging.warning(f"Could not estimate generator param from best models: {e}")
        else:
             logging.warning("Unsupervised training did not complete successfully or save models.")

        # Plot unsupervised losses
        plot_losses({'train_loss': unsup_history['total_loss']}, # Only have total loss here
                    'Unsupervised Training Loss (Total)',
                    os.path.join(args.plot_dir, 'unsupervised_loss_total.png'))
        plot_losses({'train_loss': unsup_history['oracle_loss']},
                    'Unsupervised Training Loss (Oracle)',
                    os.path.join(args.plot_dir, 'unsupervised_loss_oracle.png'))
        plot_losses({'train_loss': unsup_history['gen_loss']},
                    'Unsupervised Training Loss (Generator Reg.)',
                     os.path.join(args.plot_dir, 'unsupervised_loss_genreg.png'))

        # Plot unsupervised transformation in w space (if training was successful)
        if learned_generator_param is not None:
            try:
                logging.info("Generating unsupervised transformation plot...")
                # Get a batch of test data
                data_sample, _, _, _ = next(iter(test_loader))
                data_sample = data_sample.to(device)
                n_plot = min(args.plot_subset_size, data_sample.size(0))
                data_sample = data_sample[:n_plot]

                with torch.no_grad():
                    # Encode and flow to w
                    mu, logvar = vae_model.encoder(data_sample)
                    z = vae_model.reparameterize(mu, logvar)
                    w, _ = flow_model(z)

                    # Get generator matrix L using learned params
                    # Use the final learned params (or params from loaded best model)
                    gen_params_tensor = torch.tensor(learned_generator_param.reshape(1, -1), device=device).repeat(w.size(0), 1)
                    # Need to manually set the scale if reloading, otherwise use current model's scale
                    gen_model.gen_scale.data.fill_(final_gen_scale if final_gen_scale is not None else gen_model.gen_scale.item())
                    gen_params_rescaled = gen_model.gen_scale * torch.tanh(gen_params_tensor) # Apply scaling/tanh as in forward pass
                    L = gen_model.get_generator_matrix(gen_params_rescaled)

                    # Apply transformation
                    eps_L = args.eps_unsup * L
                    exp_eps_L = torch.matrix_exp(eps_L)
                    w_prime = torch.bmm(exp_eps_L, w.unsqueeze(-1)).squeeze(-1)

                plot_latent_transformation(w.cpu().numpy(),
                                           w_prime.cpu().numpy(),
                                           None, # No prediction here
                                           'Unsupervised Transformation in Flowed Space (w)',
                                           os.path.join(args.plot_dir, 'unsupervised_transform_w.png'))
            except Exception as e:
                logging.warning(f"Could not generate unsupervised transform plot: {e}")


    # --- Task 5: Bonus - Rotation Invariant Network ---
    if args.run_bonus:
        logging.info("\n--- Task 5: Bonus - Rotation Invariant Network ---")
        if not EMLP_AVAILABLE:
            logging.warning("Skipping Bonus Task because 'emlp' library is not available.")
        elif learned_generator_param is None or final_gen_scale is None:
             logging.warning("Skipping Bonus Task because unsupervised training failed to produce a generator parameter or scale.")
        else:
            logging.info("Building and Training Rotation Invariant Classifier...")
            try:
                # Use the final learned scale and parameter 'a'
                learned_a = learned_generator_param[0]
                L_learned_np = np.zeros((args.latent_dim, args.latent_dim))
                if args.latent_dim >= 2:
                    L_learned_np[0, 1] = -learned_a
                    L_learned_np[1, 0] = learned_a
                logging.info(f"Using Learned Generator Matrix (L) for EMLP (param a={learned_a:.4f}, scale={final_gen_scale:.4f}):\n{L_learned_np}")

                L_learned_jax = jnp.array(L_learned_np).reshape(1, args.latent_dim, args.latent_dim)
                discovered_group = DiscoveredSymmetryGroup(L_learned_jax, dim=args.latent_dim)
                logging.info(f"Successfully created custom group: {discovered_group}")

                inv_classifier = InvariantClassifierEMLPPlaceholder(
                    latent_dim=args.latent_dim,
                    discovered_group=discovered_group,
                    num_classes=len(args.digits)
                ).to(device)
                logging.info(f"Initialized Invariant Classifier (Placeholder): {inv_classifier}")

                optimizer_inv_cls = optim.Adam(inv_classifier.parameters(), lr=args.lr)
                scheduler_inv_cls = ReduceLROnPlateau(optimizer_inv_cls, 'max', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)
                best_inv_cls_acc = 0.0
                inv_cls_patience_counter = 0
                inv_cls_history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
                inv_cls_stopped_early = False

                if os.path.exists(inv_cls_path) and args.skip_training:
                    logging.info(f"Loading pre-trained Invariant Classifier from {inv_cls_path}")
                    inv_classifier = load_model(inv_classifier, inv_cls_path, device)
                else:
                    logging.info("Training Invariant Classifier (Placeholder MLP)...")
                    for epoch in range(args.epochs_invariant_cls):
                        train_loss, train_acc = train_invariant_classifier_epoch(inv_classifier, vae_model, train_loader, optimizer_inv_cls, epoch, device, args)
                        test_loss, test_acc = test_invariant_classifier_epoch(inv_classifier, vae_model, test_loader, epoch, device, args)
                        inv_cls_history['train_loss'].append(train_loss)
                        inv_cls_history['test_loss'].append(test_loss)
                        inv_cls_history['train_acc'].append(train_acc)
                        inv_cls_history['test_acc'].append(test_acc)
                        logging.info(f"Inv Cls Epoch {epoch+1}/{args.epochs_invariant_cls} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
                        scheduler_inv_cls.step(test_acc)

                        if test_acc > best_inv_cls_acc:
                            logging.info(f"Invariant Cls Test accuracy improved ({best_inv_cls_acc:.2f}% -> {test_acc:.2f}%). Saving model...")
                            best_inv_cls_acc = test_acc
                            save_model(inv_classifier, inv_cls_path)
                            inv_cls_patience_counter = 0
                        else:
                            inv_cls_patience_counter += 1
                            logging.info(f"Invariant Cls Test accuracy did not improve. Patience: {inv_cls_patience_counter}/{args.patience}")
                            if inv_cls_patience_counter >= args.patience:
                                logging.info(f"Invariant Classifier early stopping triggered after epoch {epoch+1}.")
                                inv_cls_stopped_early = True
                                break
                    if not inv_cls_stopped_early:
                         logging.info(f"Invariant Classifier training finished after {args.epochs_invariant_cls} epochs.")

                    logging.info(f"Loading best Invariant Classifier model with Test Accuracy: {best_inv_cls_acc:.2f}%")
                    inv_classifier = load_model(inv_classifier, inv_cls_path, device)

                    # Plot Invariant Classifier losses and accuracies
                    plot_losses(inv_cls_history, 'Invariant Classifier Training and Test Loss', os.path.join(args.plot_dir, 'inv_cls_loss.png'))
                    plot_accuracies(inv_cls_history, 'Invariant Classifier Training and Test Accuracy', os.path.join(args.plot_dir, 'inv_cls_accuracy.png'))


                logging.info("Final evaluation of the Invariant Classifier (Placeholder):")
                # Rerun test epoch for final numbers on loaded best model
                final_loss, final_acc = test_invariant_classifier_epoch(inv_classifier, vae_model, test_loader, -1, device, args) # Use epoch -1 for final eval log
                logging.info(f"Final Invariant Classifier Test Loss: {final_loss:.4f}, Test Accuracy: {final_acc:.2f}%")


            except Exception as e:
                logging.error(f"Error during EMLP setup or training: {e}", exc_info=True)
                logging.warning("Skipping EMLP part of the bonus task.")

    logging.info("\nAll tasks finished.")

if __name__ == "__main__":
    main()
