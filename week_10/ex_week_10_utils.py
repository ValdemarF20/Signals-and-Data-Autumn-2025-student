"""
Utility functions for Week 11 exercises on Relative Representations with Autoencoders

Contains:
- Data loading for MNIST
- Plotting functions for embeddings with PCA and anchors
- Model saving/loading with architecture verification
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import os
import hashlib
import inspect


# ========== Model Saving/Loading ==========

def get_model_architecture_hash(model_class, *args, **kwargs):
    """
    Generate a hash based on model architecture.
    
    Args:
        model_class: The model class (e.g., CNNAutoencoder)
        *args, **kwargs: Arguments used to instantiate the model
    
    Returns:
        MD5 hash string representing the architecture
    """
    # Try to get source code, fall back to model structure if unavailable
    try:
        source = inspect.getsource(model_class)
        config = f"{source}_{args}_{kwargs}"
    except (OSError, TypeError):
        # If source not available (e.g., in notebook), use model structure
        # Create a temporary instance to get the architecture
        temp_model = model_class(*args, **kwargs)
        # Use string representation of the model which includes architecture
        model_str = str(temp_model)
        config = f"{model_str}_{args}_{kwargs}"
        del temp_model  # Clean up
    
    return hashlib.md5(config.encode()).hexdigest()


def load_model_if_valid(model_path, model_class, device, *args, **kwargs):
    """
    Load model if it exists and architecture matches.
    
    Args:
        model_path: Path to saved model file
        model_class: The model class to instantiate
        device: Device to load model to (cpu/cuda)
        *args, **kwargs: Arguments to pass to model_class constructor
    
    Returns:
        (model, loaded): Tuple of (model instance or None, bool indicating if loaded)
    """
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            saved_hash = checkpoint.get('architecture_hash', None)
            current_hash = get_model_architecture_hash(model_class, *args, **kwargs)
            
            if saved_hash == current_hash:
                model = model_class(*args, **kwargs)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                print(f"Loaded pre-trained model from {model_path}")
                return model, True
            else:
                print(f"Model architecture changed, retraining...")
                return None, False
        except Exception as e:
            print(f"Error loading model: {e}, retraining...")
            return None, False
    return None, False


def save_model(model, model_path, model_class, *args, **kwargs):
    """
    Save model with architecture hash for validation.
    
    Args:
        model: The model instance to save
        model_path: Path to save model file
        model_class: The model class
        *args, **kwargs: Arguments used to instantiate the model
    """
    # Create models directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    arch_hash = get_model_architecture_hash(model_class, *args, **kwargs)
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture_hash': arch_hash,
    }, model_path)
    print(f"Model saved to {model_path}")


# ========== Data Loading ==========

def load_mnist_data(batch_size=128, validation_split=0.1, data_dir='./data'):
    """
    Load MNIST dataset with train, validation, and test splits.
    
    Args:
        batch_size: Batch size for DataLoaders
        validation_split: Fraction of training data to use for validation
        data_dir: Directory to store/load data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Transform: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    # Split train into train and validation
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# ========== Visualization with PCA and Anchors ==========

def fit_and_align_pca(data, ref_pca=None):
    """
    Fit PCA on data and optionally align with reference PCA.
    
    Returns:
        If ref_pca is None: (pca, data_2d)
        If ref_pca is not None: data_2d (aligned)
    """
    pca = PCA(n_components=2, random_state=42)
    data_pca = pca.fit_transform(data)
    
    if ref_pca is None:
        return pca, data_pca
    
    # Align signs with reference
    for i in range(2):
        dot_product = np.dot(pca.components_[i], ref_pca.components_[i])
        if dot_product < 0:
            pca.components_[i] = -pca.components_[i]
            data_pca[:, i] = -data_pca[:, i]
    
    return data_pca


def plot_data_list(data_list, labels_list, do_pca=True, ref_pca=None, is_relrep=True, 
                   anchors_list=None, title=None, output_file='embeddings_plot.png', return_fig=False):
    """
    Plot multiple datasets side-by-side.
    
    Args:
        data_list: List of data arrays (each [N, features])
        labels_list: List of label arrays
        do_pca: Whether to apply PCA
        ref_pca: Reference PCA for alignment (internal use)
        is_relrep: If True, use shared PCA for relative representations
        anchors_list: Optional list of anchor arrays to plot
        title: Plot title
        output_file: Output filename
        return_fig: If True, return figure instead of saving
    """
    n_plots = len(data_list)
    fig, axs = plt.subplots(1, n_plots, figsize=(7*n_plots, 4), squeeze=False)
    axs = axs.ravel()
    
    for i, (data, labels) in enumerate(zip(data_list, labels_list)):
        if do_pca:
            if ref_pca is None:
                pca, data_2d = fit_and_align_pca(data, ref_pca=ref_pca)
                pca_current = pca
            else:
                if is_relrep:
                    # For relative reps, use SAME PCA from first space
                    data_2d = ref_pca.transform(data)
                    pca_current = ref_pca
                else:
                    # Independent PCA
                    _, data_2d = fit_and_align_pca(data, ref_pca=None)
                    pca_current = None
            
            if i == 0 and ref_pca is None:
                ref_pca = pca_current
        else:
            data_2d = data
            pca_current = None
        
        scatter = axs[i].scatter(data_2d[:, 0], data_2d[:, 1],
                                c=labels, cmap='tab10', s=10, alpha=0.7)
        
        # Plot anchors if provided
        if anchors_list is not None:
            anchors = anchors_list[i]
            if do_pca and (pca_current is not None):
                anchors_2d = pca_current.transform(anchors)
            else:
                anchors_2d = anchors
            axs[i].scatter(anchors_2d[:, 0], anchors_2d[:, 1], 
                          s=100, marker="*", c='red', edgecolors='black', linewidths=1, zorder=10)
        
        # Colorbar
        cb = fig.colorbar(scatter, ax=axs[i], ticks=range(10))
        cb.set_label('Digit Class')
        
        axs[i].set_xlabel('PC 1')
        axs[i].set_ylabel('PC 2')
        if title is None:
            axs[i].set_title(f'Model {i+1}')
        else:
            axs[i].set_title(f'{title} {i+1}')
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        plt.close()


def plot_embeddings_2d(embeddings_list, labels_list, titles=None, 
                       figsize_per_plot=(6, 5), suptitle=None, save_path=None,
                       anchors_list=None):
    """
    Plot multiple 2D embedding spaces side by side.
    
    Note: Embeddings must already be 2D (e.g., after PCA).
    
    Args:
        embeddings_list: List of 2D numpy arrays, each of shape (n_samples, 2)
        labels_list: List of label arrays for coloring points
        titles: List of titles for each subplot (optional)
        figsize_per_plot: Figure size for each subplot
        suptitle: Overall title for the figure
        save_path: Path to save figure (optional)
        anchors_list: Optional list of anchor arrays to plot (must be 2D)
    
    Returns:
        fig, axes
    """
    n_plots = len(embeddings_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize_per_plot[0]*n_plots, figsize_per_plot[1]))
    
    # Handle single plot case
    if n_plots == 1:
        axes = [axes]
    
    for i, (embeddings, labels) in enumerate(zip(embeddings_list, labels_list)):
        # Convert to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Check that embeddings are 2D
        if embeddings.shape[1] != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}. Apply PCA first!")
        
        # Plot data points
        scatter = axes[i].scatter(embeddings[:, 0], embeddings[:, 1],
                                 c=labels, cmap='tab10', s=8, alpha=0.6, edgecolors='none')
        
        # Plot anchors if provided
        if anchors_list is not None and i < len(anchors_list):
            anchors = anchors_list[i]
            if isinstance(anchors, torch.Tensor):
                anchors = anchors.cpu().numpy()
            axes[i].scatter(anchors[:, 0], anchors[:, 1], 
                          s=100, marker="*", c='red', edgecolors='black', linewidths=1, zorder=10)
        
        # Set title
        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
        else:
            axes[i].set_title(f'Space {i+1}', fontsize=12)
        
        # Labels
        axes[i].set_xlabel('Dimension 1', fontsize=10)
        axes[i].set_ylabel('Dimension 2', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    # Add a shared colorbar on the right side of all plots
    # Create space for colorbar by adjusting subplot positions
    plt.tight_layout(rect=(0, 1.0, 0.95, 1))
    
    # Add colorbar in the space we created
    cbar_ax = fig.add_axes((0.96, 0.15, 0.02, 0.7))  # (left, bottom, width, height)
    cbar = fig.colorbar(scatter, cax=cbar_ax, ticks=range(10))
    cbar.set_label('Digit Class', rotation=270, labelpad=20, fontsize=10)
    
    # Add suptitle if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def plot_embeddings_comparison(embeddings1, embeddings2, labels, 
                               title1="Model 1", title2="Model 2", suptitle=None,
                               anchors1=None, anchors2=None):
    """
    Convenience function to plot two 2D embedding spaces side by side.
    
    Note: Both embeddings must already be 2D (e.g., after PCA).
    
    Args:
        embeddings1: First 2D embedding array (n_samples, 2)
        embeddings2: Second 2D embedding array (n_samples, 2)
        labels: Label array (same for both)
        title1: Title for first plot
        title2: Title for second plot
        suptitle: Overall title
        anchors1: Optional anchors for first plot (2D)
        anchors2: Optional anchors for second plot (2D)
    
    Returns:
        fig, axes
    """
    embeddings_list = [embeddings1, embeddings2]
    labels_list = [labels, labels]
    titles = [title1, title2]
    anchors_list = None
    if anchors1 is not None or anchors2 is not None:
        anchors_list = [anchors1, anchors2]
    
    return plot_embeddings_2d(embeddings_list, labels_list, titles=titles, 
                             suptitle=suptitle, anchors_list=anchors_list)


def plot_reconstructions(images_list, row_labels=None, num_samples=10, 
                         figsize=None, suptitle=None, save_path=None):
    """
    Plot multiple rows of image reconstructions for comparison.
    
    Args:
        images_list: List of image arrays, each of shape (n_samples, H, W) or (n_samples, 1, H, W)
        row_labels: List of labels for each row (e.g., ['Original', 'AE1 Recon', ...])
        num_samples: Number of samples to display
        figsize: Figure size tuple (width, height). If None, auto-computed
        suptitle: Overall title for the figure
        save_path: Path to save the figure (optional)
    
    Returns:
        fig, axes
    """
    if row_labels is None:
        row_labels = [f'Row {i+1}' for i in range(len(images_list))]
    
    n_rows = len(images_list)
    
    if figsize is None:
        # Increase vertical spacing to accommodate row titles
        figsize = (num_samples * 1.5, n_rows * 1.8)
    
    # Increase vertical spacing between rows
    fig, axes = plt.subplots(n_rows, num_samples, figsize=figsize, 
                             gridspec_kw={'hspace': 0.3})
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, images in enumerate(images_list):
        # Convert to numpy if needed
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        # Take only num_samples
        images = images[:num_samples]
        
        for col_idx in range(min(num_samples, len(images))):
            img = images[col_idx]
            
            # Handle different shapes: (H, W), (1, H, W), or (C, H, W)
            if img.ndim == 3:
                if img.shape[0] == 1:
                    img = img.squeeze(0)  # Remove channel dimension
                elif img.shape[0] > 1:
                    img = img.transpose(1, 2, 0)  # CHW to HWC
            
            axes[row_idx, col_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, col_idx].axis('off')
        
        # Add row label above the first image in each row (as a title)
        # Use the first axis to set a title for the entire row
        axes[row_idx, 0].text(-0.1, 1.15, row_labels[row_idx],
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=12, fontweight='bold',
                             va='bottom', ha='left')
    
    if suptitle:
        plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.99)
    
    # Adjust layout to make room for row labels above rows
    plt.tight_layout(rect=[0, 0, 1, 0.97 if suptitle else 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to '{save_path}'")
    
    return fig, axes
