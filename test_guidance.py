# test_guidance.py
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from ddpm import DDPM
from classifier import CelebAClassifier
from guidance import guided_reverse_step


def load_image(path, size=256):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def to_displayable(tensor):
    img = tensor.detach().cpu().float()
    img = (img + 1) / 2
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def test_single_guided_step(x0, ddpm, classifier):
    """
    Test that one guided step runs without errors and that
    the guidance actually changes x0_hat relative to unguided.
    """
    print("Testing single guided step...")

    tau = 60
    target_label = 1
    lambda_c = 5.0
    lambda_l1 = 0.05

    # Corrupt image to tau
    x_t, _ = ddpm.forward(x0, tau)

    # Unguided: just predict x0_hat with no guidance
    x0_hat_unguided = ddpm.predict_x0(x_t, tau)

    # Guided: one full guided step
    x_prev, info = guided_reverse_step(
        x_t, tau, ddpm, classifier,
        target_label=target_label,
        lambda_c=lambda_c,
        lambda_l1=lambda_l1,
        x_orig=x0
    )

    print(f"  CE loss:    {info['ce_loss']:.4f}")
    print(f"  L1 loss:    {info['l1_loss']:.4f}")
    print(f"  Total loss: {info['total_loss']:.4f}")

    # Check that guidance changed something
    diff = (x_prev - x0_hat_unguided).abs().mean().item()
    print(f"  Mean pixel change from guidance: {diff:.6f}")
    assert diff > 0, "Guidance had no effect — something is wrong"
    print("  Gradient flow: OK")

    return x_t, x0_hat_unguided, x_prev


def test_full_guided_loop(x0, ddpm, classifier):
    """
    Run the full guided denoising loop from tau to 0.
    Watch the losses at each step — CE loss should fluctuate as
    guidance steers the image. With random classifier weights it
    won't converge meaningfully, but the mechanics should work.
    """
    print("\nRunning full guided loop...")

    tau = 60
    target_label = 1
    lambda_c = 5.0
    lambda_l1 = 0.05

    x_t, _ = ddpm.forward(x0, tau)
    ce_losses = []

    for t in range(tau, 0, -1):
        x_t, info = guided_reverse_step(
            x_t, t, ddpm, classifier,
            target_label=target_label,
            lambda_c=lambda_c,
            lambda_l1=lambda_l1,
            x_orig=x0
        )
        ce_losses.append(info["ce_loss"])

        if t % 15 == 0:
            prob = classifier.probability(
                x_t.to(next(classifier.parameters()).device),
                label=target_label
            ).item()
            print(f"  t={t:3d}  CE={info['ce_loss']:.4f}  "
                  f"p(target)={prob:.4f}")

    return x_t, ce_losses


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_guidance.py path/to/image.jpg")
        sys.exit(1)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    print("Loading image...")
    x0 = load_image(sys.argv[1]).to(device)

    print("Loading DDPM...")
    ddpm = DDPM()

    print("Loading classifier (random weights — just testing gradient flow)...")
    classifier = CelebAClassifier(num_classes=2).to(device)
    classifier.eval()

    # Test 1: single step
    x_t, x0_unguided, x0_guided = test_single_guided_step(x0, ddpm, classifier)

    # Test 2: full loop
    x_final, ce_losses = test_full_guided_loop(x0, ddpm, classifier)

    # Plot results
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(to_displayable(x0[0]))
    axes[0].set_title("original")
    axes[0].axis("off")

    axes[1].imshow(to_displayable(x_t[0]))
    axes[1].set_title("corrupted τ=60")
    axes[1].axis("off")

    axes[2].imshow(to_displayable(x0_unguided[0]))
    axes[2].set_title("unguided x̂₀")
    axes[2].axis("off")

    axes[3].imshow(to_displayable(x_final[0]))
    axes[3].set_title("guided (random clf)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig("test_guidance.png", dpi=150)

    # Plot CE loss over steps
    plt.figure(figsize=(8, 3))
    plt.plot(ce_losses)
    plt.xlabel("denoising step (τ → 0)")
    plt.ylabel("CE loss")
    plt.title("classifier loss during guided denoising")
    plt.tight_layout()
    plt.savefig("test_guidance_loss.png", dpi=150)

    print("\nSaved test_guidance.png and test_guidance_loss.png")