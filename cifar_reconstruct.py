import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import AutoEncoder
import utils

# -------------------------------
# Helper function to show images
# -------------------------------
def show_images(orig, recon, n=10, save_path="cifar_reconstructions.png"):
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        axes[0, i].imshow(orig[i].permute(1,2,0).cpu().numpy())
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].permute(1,2,0).cpu().numpy())
        axes[1, i].axis("off")
    axes[0,0].set_ylabel("Original", fontsize=12)
    axes[1,0].set_ylabel("Recon", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Load checkpoint
# -------------------------------
checkpoint_path = "checkpoints/cifar10/checkpoint.pt"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
args = checkpoint['args']

# Ensure num_mixture_dec exists
if not hasattr(args, 'num_mixture_dec'):
    args.num_mixture_dec = 10

# Build architecture instance
arch_instance = utils.get_arch_cells(args.arch_instance)

# -------------------------------
# Initialize model
# -------------------------------
model = AutoEncoder(args, writer=None, arch_instance=arch_instance).to(device)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()
print(f"Checkpoint loaded: Epoch {checkpoint.get('epoch', 'N/A')}, Step {checkpoint.get('global_step', 'N/A')}")

# -------------------------------
# Prepare CIFAR-10 test set
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

images, labels = next(iter(testloader))
images = images.to(device)

print(f"Image shape: {images.shape}")
print(f"Image range: {images.min().item():.3f} to {images.max().item():.3f}")

# -------------------------------
# Forward pass - FIXED VERSION
# -------------------------------
with torch.no_grad():
    # Method 1: Try direct model call
    try:
        output = model(images)
        print(f"Model output length: {len(output)}")
        
        if len(output) >= 1:
            logits = output[0]
            print(f"Logits shape: {logits.shape}")
            print(f"Logits range: {logits.min().item():.3f} to {logits.max().item():.3f}")
            
            # Try different approaches to get reconstruction
            
            # Approach 1: Direct sigmoid on logits
            recon = torch.sigmoid(logits)
            
            # If that doesn't work, try decoder_output
            try:
                decoder = model.decoder_output(logits)
                if hasattr(decoder, 'mean'):
                    recon = decoder.mean()
                elif hasattr(decoder, 'sample'):
                    recon = decoder.sample()
                print("Used decoder_output approach")
            except:
                print("Using direct sigmoid approach")
                recon = torch.sigmoid(logits)
            
    except Exception as e:
        print(f"Error in forward pass: {e}")
        # Fallback: create dummy reconstruction for debugging
        recon = torch.zeros_like(images)

print(f"Reconstruction shape: {recon.shape}")
print(f"Reconstruction range: {recon.min().item():.3f} to {recon.max().item():.3f}")

# Ensure proper range for visualization [0, 1]
images_vis = images.clamp(0, 1)
recon_vis = recon.clamp(0, 1)

# -------------------------------
# Show images
# -------------------------------
show_images(images_vis, recon_vis, n=10, save_path="cifar_reconstructions_fixed.png")

# -------------------------------
# Print statistics if available
# -------------------------------
if len(output) > 1:
    try:
        log_q, log_p = output[1], output[2]
        if len(output) > 3:
            kl_all = output[3]
            print(f"Mean KL (all scales): {torch.mean(torch.stack(kl_all)):.4f}")
        print(f"Mean log_p (reconstruction likelihood): {torch.mean(log_p):.4f}")
    except:
        print("Could not compute statistics")

print("Reconstruction complete! Check 'cifar_reconstructions_fixed.png'")