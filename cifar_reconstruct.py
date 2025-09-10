# import torch
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# from model import AutoEncoder
# import utils

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load checkpoint
# checkpoint_path = "checkpoints/cifar10/checkpoint.pt"
# checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# args = checkpoint['args']
# if not hasattr(args, 'num_mixture_dec'):
#     args.num_mixture_dec = 10

# arch_instance = utils.get_arch_cells(args.arch_instance)

# # Initialize model
# model = AutoEncoder(args, writer=None, arch_instance=arch_instance).to(device)
# model.load_state_dict(checkpoint['state_dict'], strict=False)
# model.eval()

# # Prepare CIFAR-10 test set
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x - 0.5)  # NVAE expects [-0.5,0.5]
# ])
# testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

# images, _ = next(iter(testloader))
# images = images.to(device)

# # Forward pass
# with torch.no_grad():
#     logits, *_ = model(images)
#     decoder = model.decoder_output(logits)
#     recon = decoder.mean() if hasattr(decoder, "mean") else decoder.sample()
    
# # Denormalize to [0,1]
# images_vis = (images + 0.5).clamp(0, 1).cpu()
# recon_vis = (recon + 0.5).clamp(0, 1).cpu()

# # Show images
# def show_images(orig, recon, n=10):
#     fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
#     for i in range(n):
#         axes[0,i].imshow(orig[i].permute(1,2,0).numpy())
#         axes[0,i].axis("off")
#         axes[1,i].imshow(recon[i].permute(1,2,0).numpy())
#         axes[1,i].axis("off")
#     axes[0,0].set_ylabel("Original", fontsize=12)
#     axes[1,0].set_ylabel("Recon", fontsize=12)
#     plt.show()
#     plt.savefig("cifar_reconstructions.png")

# show_images(images_vis, recon_vis, n=10)

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import AutoEncoder
import utils
from torch.serialization import add_safe_globals
from argparse import Namespace

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Allow argparse.Namespace globally for loading checkpoint
add_safe_globals([Namespace])  # just call it, no 'with'

# Checkpoint path
checkpoint_path = "checkpoints/cifar10/checkpoint.pt"

# Load checkpoint with weights_only=False
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
args = checkpoint['args']

# Fill missing defaults
if not hasattr(args, 'num_mixture_dec'):
    args.num_mixture_dec = 10

# Get architecture instance
arch_instance = utils.get_arch_cells(args.arch_instance)

# Initialize model
model = AutoEncoder(args, writer=None, arch_instance=arch_instance).to(device)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()

# Transform for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x - 0.5)
])

# Load test dataset
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

# Get one batch
images, _ = next(iter(testloader))
images = images.to(device)

# Forward pass
with torch.no_grad():
    recon, _, _ = model(images)

# Denormalize
images_vis = (images + 0.5).clamp(0, 1).cpu()
recon_vis = (recon + 0.5).clamp(0, 1).cpu()

# Plot function
def show_images(orig, recon, n=10):
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        axes[0,i].imshow(orig[i].permute(1,2,0).numpy())
        axes[0,i].axis("off")
        axes[1,i].imshow(recon[i].permute(1,2,0).numpy())
        axes[1,i].axis("off")
    axes[0,0].set_ylabel("Original", fontsize=12)
    axes[1,0].set_ylabel("Recon", fontsize=12)
    plt.show()
    plt.savefig("cifar_reconstructions.png")

show_images(images_vis, recon_vis, n=10)
