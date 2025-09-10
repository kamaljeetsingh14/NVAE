import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import AutoEncoder
import utils
from argparse import Namespace  # Needed for PyTorch 2.6+ safe loading

# --------------------------
# 0. Set device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------
# 1. Load checkpoint safely
# --------------------------
checkpoint_path = "checkpoints/cifar10/checkpoint.pt"

# Option A: add Namespace to safe globals (recommended for NVAE)
torch.serialization.add_safe_globals([Namespace])
checkpoint = torch.load(checkpoint_path, map_location=device)

args = checkpoint['args']

# Fill missing defaults (old checkpoints)
if not hasattr(args, 'num_mixture_dec'):
    args.num_mixture_dec = 10
if not hasattr(args, 'ada_groups'):
    args.ada_groups = False
if not hasattr(args, 'min_groups_per_scale'):
    args.min_groups_per_scale = 1

# --------------------------
# 2. Build model
# --------------------------
arch_instance = utils.get_arch_cells(args.arch_instance)
model = AutoEncoder(args, writer=None, arch_instance=arch_instance)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.to(device)
model.eval()

# --------------------------
# 3. Load 10 CIFAR-10 examples
# --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x - 0.5)  # NVAE expects [-0.5, 0.5]
])
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

images, _ = next(iter(testloader))
images = images.to(device)

# --------------------------
# 4. Forward pass / reconstruction
# --------------------------
with torch.no_grad():
    recon = model(images)

# Denormalize back to [0,1]
images_vis = (images + 0.5).clamp(0, 1)
recon_vis = (recon + 0.5).clamp(0, 1)

# --------------------------
# 5. Visualization
# --------------------------
def show_images(orig, recon, n=10):
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        axes[0, i].imshow(orig[i].permute(1,2,0).cpu().numpy())
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].permute(1,2,0).cpu().numpy())
        axes[1, i].axis("off")
    axes[0,0].set_ylabel("Original", fontsize=12)
    axes[1,0].set_ylabel("Recon", fontsize=12)
    plt.show()
    fig.savefig("cifar_reconstructions.png", bbox_inches='tight')

show_images(images_vis, recon_vis, n=10)
