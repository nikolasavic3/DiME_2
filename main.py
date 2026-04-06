# main.py
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from ddpm import DDPM
from classifier import CelebAClassifier
from generate import generate_counterfactual


def load_image(path, size=256):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def to_displayable(tensor):
    img = tensor.detach().cpu().float()
    img = (img + 1) / 2
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def save_results(x0, result):
    cf = result["cf"]
    diff = (cf - x0).abs()
    # Amplify difference map so subtle changes are visible
    diff_amplified = (diff * 5).clamp(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(to_displayable(x0[0]))
    axes[0].set_title("original")
    axes[0].axis("off")

    axes[1].imshow(to_displayable(cf[0]))
    status = "success" if result["success"] else "failed"
    axes[1].set_title(f"counterfactual ({status})")
    axes[1].axis("off")

    axes[2].imshow(to_displayable(diff_amplified[0]))
    axes[2].set_title(f"difference x5  L1={result['l1_dist']:.4f}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("result.png", dpi=150)
    print("Saved result.png")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py path/to/image.jpg target_label")
        print("  target_label: 0 or 1  (binary classifier)")
        print("  e.g.: python main.py celeba/img_align_celeba/000001.jpg 1")
        sys.exit(1)

    image_path = sys.argv[1]
    target_label = int(sys.argv[2])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    print("Loading image...")
    x0 = load_image(image_path).to(device)

    print("Loading DDPM...")
    ddpm = DDPM()

    print("Loading classifier...")
    classifier = CelebAClassifier(num_classes=2).to(device)
    classifier.load_state_dict(torch.load("checkpoints/classifier_smiling.pt", map_location=device))
    classifier.eval()

    print("\nGenerating counterfactual...")
    result = generate_counterfactual(
        x0=x0,
        ddpm=ddpm,
        classifier=classifier,
        target_label=target_label,
        tau=60,
        lambda_c_values=(5.0, 8.0, 10.0, 15.0),
        lambda_l1=0.05,
    )

    print(f"\nResult:")
    print(f"  Success:   {result['success']}")
    print(f"  lambda_c:  {result['lambda_c']}")
    print(f"  L1 dist:   {result['l1_dist']:.4f}")

    save_results(x0, result)