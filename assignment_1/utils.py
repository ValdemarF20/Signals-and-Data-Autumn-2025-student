from torch.utils.data import DataLoader
from torch import nn, sqrt


def compute_channel_stats(dataset, batch_size=32):
    """
    Compute the channel-wise mean and standard deviation of all images in a dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): A dataset object that returns (image, label) pairs.
        batch_size (int): Batch size for processing the dataset.
    
    Returns:
        mean (tensor): A tensor containing the channel-wise mean.
        std (tensor): A tensor containing the channel-wise standard deviation.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    mean_sum = 0.0
    std_sum = 0.0
    total_images = 0

    for images, _ in dataloader:
        images = images.view(images.size(0), images.size(1), -1)
        
        total_images += images.size(0)
        
        mean_sum += images.mean(dim=2).sum(dim=0)
        std_sum += images.var(dim=2, unbiased=False).sum(dim=0)
    
    mean = mean_sum / total_images
    std = sqrt(std_sum / total_images)
    
    return mean, std

def get_dim_before_first_linear(layers, in_dim, in_channels, brain=False):
    """
    Calculate the dimensions before the first linear layer.
    DISCLAIMER: THIS IS IN NO WAY GUARANTEED TO WORK, AND IS PURELY FOR CONVENIENCE!
    If it fails, you will have to manually calculate the dimensions before the first linear layer

    Args:
        layers: Sequential layers to analyze
        in_dim: Input dimension (assumed square)
        in_channels: Number of input channels
        brain: If True, returns (height, width, channels) tuple. If False, returns total features.
    
    Returns:
        If brain=True: (height, width, channels) tuple
        If brain=False: total number of features (height * width * channels)
    

    """

    try:
        current_dim = in_dim
        current_channels = in_channels
        for layer in layers:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
                # If the layer padding is same we do not need to change the dimension of the input...
                if layer.padding == 'same':
                    if isinstance(layer, nn.Conv2d):
                        current_channels = layer.out_channels
                    # For MaxPool2d with padding='same', we need to calculate the output size
                    if isinstance(layer, nn.MaxPool2d):
                        # For padding='same', output size = ceil(input_size / stride)
                        stride = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
                        current_dim = (current_dim + stride - 1) // stride
                    continue
                vals = {
                    'kernel_size': layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0],
                    'stride': layer.stride if isinstance(layer.stride, int) else layer.stride[0],
                    'padding': layer.padding if isinstance(layer.padding, int) else layer.padding[0],
                    'dilation': layer.dilation if isinstance(layer.dilation, int) else layer.dilation[0]
                }
                current_dim = (current_dim + 2*vals['padding'] - vals['dilation']*(vals['kernel_size'])) // vals['stride'] + 1
            if isinstance(layer, nn.Conv2d):
                current_channels = layer.out_channels
        
            if isinstance(layer, nn.Linear):
                if brain:
                    return current_dim, current_channels
                else:
                    return current_dim * current_dim * current_channels
        
        # If no linear layer found, return the final dimensions
        if brain:
            return current_dim, current_channels
        else:
            return current_dim * current_dim * current_channels
    
    except:
        print("""
                Ooops! Something went wrong in getting the dimension of the data before the first linear layer!
                If you'll refer to the docstring of this function, you should see the disclaimer, that it may very well fail (it really is a hack)
                In this case, you should instead manually calculate the dimensionality of the linear layer, immediately following the last convolutional layer
                Good luck!
                """)

    # raise ValueError("No linear layer found in layers! Why are you even asking me?")