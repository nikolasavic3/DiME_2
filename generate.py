# generate.py
import torch
from tqdm import tqdm
from guidance import guided_reverse_step


def generate_counterfactual(
    x0,
    ddpm,
    classifier,
    target_label,
    tau=60,
    lambda_c_values=(5.0, 8.0, 10.0, 15.0),
    lambda_l1=0.05,
    seed=42,
):
    """
    Generate a counterfactual explanation for x0.

    The goal: find the smallest change to x0 that flips the classifier
    to predict target_label.

    Strategy (DiME):
      1. Corrupt x0 to noise level tau
      2. Run guided reverse diffusion from tau → 0
      3. Check if the result fools the classifier
      4. If not, increase lambda_c and repeat

    Args:
        x0:              original image, (1, C, H, W) in [-1, 1]
        ddpm:            DDPM instance
        classifier:      trained classifier
        target_label:    int, the class we want to flip toward
        tau:             int, noise level to corrupt to (DiME default: 60)
        lambda_c_values: tuple of guidance scales to try, in increasing order
        lambda_l1:       proximity loss weight
        seed:            random seed for reproducibility

    Returns:
        dict with:
          'cf':         counterfactual image tensor, or None if failed
          'success':    bool
          'lambda_c':   which lambda_c succeeded
          'l1_dist':    L1 distance from original
          'history':    loss values across all attempts
    """

    device = x0.device
    torch.manual_seed(seed)

    # Check what the classifier currently predicts
    with torch.no_grad():
        original_pred = classifier(x0).argmax(dim=1).item()

    print(f"Original prediction: {original_pred}")
    print(f"Target label:        {target_label}")

    if original_pred == target_label:
        print("Image already classified as target — no counterfactual needed.")
        return {
            "cf": x0,
            "success": True,
            "lambda_c": None,
            "l1_dist": 0.0,
            "history": [],
        }

    history = []

    # Corrupt once — same noisy starting point for all attempts
    # This is important: using the same x_tau makes lambda_c comparisons fair
    x_tau, _ = ddpm.forward(x0, tau)

    for lambda_c in lambda_c_values:
        print(f"\nTrying lambda_c={lambda_c}...")

        # Start from the same x_tau each attempt
        x_t = x_tau.clone()
        step_losses = []

        # Guided reverse loop: tau → 0
        for t in tqdm(range(tau, 0, -1), desc=f"  λ_c={lambda_c}"):
            x_t, info = guided_reverse_step(
                x_t, t, ddpm, classifier,
                target_label=target_label,
                lambda_c=lambda_c,
                lambda_l1=lambda_l1,
                x_orig=x0,
            )
            step_losses.append(info)

        # Check if the counterfactual fooled the classifier
        with torch.no_grad():
            cf_pred = classifier(x_t).argmax(dim=1).item()

        l1_dist = (x_t - x0).abs().mean().item()

        print(f"  CF prediction: {cf_pred}  |  L1 dist: {l1_dist:.4f}")

        history.append({
            "lambda_c": lambda_c,
            "cf_pred": cf_pred,
            "l1_dist": l1_dist,
            "step_losses": step_losses,
        })

        if cf_pred == target_label:
            print(f"  Success with lambda_c={lambda_c}")
            return {
                "cf": x_t,
                "success": True,
                "lambda_c": lambda_c,
                "l1_dist": l1_dist,
                "history": history,
            }

    # All lambda_c values exhausted — no flip achieved
    print("\nFailed to generate counterfactual.")
    return {
        "cf": x_t,   # return the last attempt anyway for inspection
        "success": False,
        "lambda_c": None,
        "l1_dist": l1_dist,
        "history": history,
    }