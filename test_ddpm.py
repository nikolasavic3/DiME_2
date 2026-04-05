# test_ddpm.py  (updated)
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from ddpm import DDPM

def load_image(path, size=256):
    """
    Load a single image and convert it to a (1, C, H, W) tensor in [-1, 1].
    This is exactly what the DDPM expects.
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),           # [0,1]
        transforms.Normalize([0.5]*3, [0.5]*3)  # [0,1] → [-1,1]
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # add batch dim → (1, C, H, W)


def to_displayable(tensor):
    """Convert a (C, H, W) tensor in [-1, 1] to a numpy array for display."""
    img = tensor.detach().cpu().float()
    img = (img + 1) / 2
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def test_forward(x0):
    ddpm = DDPM()

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(to_displayable(x0[0]))
    axes[0].set_title("x0 (original)")
    axes[0].axis("off")

    for i, t in enumerate([20, 60, 200, 999]):
        x_t, _ = ddpm.forward(x0, t)
        axes[i+1].imshow(to_displayable(x_t[0]))
        axes[i+1].set_title(f"t={t}")
        axes[i+1].axis("off")

    plt.tight_layout()
    plt.savefig("test_forward.png", dpi=150)
    print("Saved test_forward.png")
    return ddpm


def test_reconstruct(ddpm, x0, tau=60):
    """
    The DiME-relevant test: corrupt to tau, denoise back, compare.
    """
    x_t, _ = ddpm.forward(x0, tau)
    print(f"Running reverse loop from t={tau}...")

    for t in range(tau, 0, -1):
        x_t = ddpm.reverse_step(x_t, t)
        if t % 20 == 0:
            print(f"  t={t}")

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].imshow(to_displayable(x0[0]))
    axes[0].set_title("original")
    axes[0].axis("off")

    x_noisy, _ = ddpm.forward(x0, tau)
    axes[1].imshow(to_displayable(x_noisy[0]))
    axes[1].set_title(f"corrupted  τ={tau}")
    axes[1].axis("off")

    axes[2].imshow(to_displayable(x_t[0]))
    axes[2].set_title("reconstructed")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("test_reconstruct.png", dpi=150)
    print("Saved test_reconstruct.png")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_ddpm.py path/to/image.jpg")
        print("e.g.:  python test_ddpm.py ~/data/celeba/img_align_celeba/000001.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Loading image: {image_path}")
    x0 = load_image(image_path)
    print(f"Tensor shape: {x0.shape}, range: [{x0.min():.2f}, {x0.max():.2f}]")

    print("\n=== Test 1: forward process ===")
    ddpm = test_forward(x0)

    print("\n=== Test 2: corrupt → reconstruct ===")
    test_reconstruct(ddpm, x0, tau=200)

    print("\nDone. Check test_forward.png and test_reconstruct.png")