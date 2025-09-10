import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import AutoEncoder
import utils


checkpoint_path = "checkpoints/cifar10/checkpoint.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
args = checkpoint['args']

# Fill missing defaults
if not hasattr(args, 'num_mixture_dec'):
    args.num_mixture_dec = 10

# get actual architecture
arch_instance = utils.get_arch_cells(args.arch_instance)

# Initialize model
model = AutoEncoder(args, writer=None, arch_instance=arch_instance)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x - 0.5)  # NVAE expects [-0.5,0.5]
])
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

images, _ = next(iter(testloader))


with torch.no_grad():
    recon, _, _ = model(images)
    
# Denormalize back to [0,1]
images_vis = images + 0.5
recon_vis = recon + 0.5

def show_images(orig, recon, n=10):
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        axes[0,i].imshow(orig[i].permute(1,2,0).clamp(0,1).numpy())
        axes[0,i].axis("off")
        axes[1,i].imshow(recon[i].permute(1,2,0).clamp(0,1).numpy())
        axes[1,i].axis("off")
    axes[0,0].set_ylabel("Original", fontsize=12)
    axes[1,0].set_ylabel("Recon", fontsize=12)
    plt.show()
    plt.savefig("cifar_reconstructions.png")

show_images(images_vis, recon_vis, n=10)
