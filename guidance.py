# guidance.py
import torch
import torch.nn.functional as F


def get_guidance_gradient(x0_hat, classifier, target_label, lambda_l1, x_orig):
    """
    Compute the gradient that guides the denoising toward the target label.

    This is the core of DiME. Given an estimated clean image x̂₀, we ask:
    "in which direction should we move x̂₀ to make the classifier more
    confident about the target label?"

    That direction is the gradient of the classifier's log-probability
    with respect to x̂₀ — exactly like an adversarial attack, but applied
    gently at each denoising step rather than all at once.

    We also add an L1 proximity term that pulls x̂₀ back toward the
    original image, preventing unnecessary changes to irrelevant attributes.

    Args:
        x0_hat:       estimated clean image, shape (B, C, H, W), requires_grad
        classifier:   the model under observation (ResNet-18)
        target_label: int, the attribute class we want to flip toward
        lambda_l1:    float, weight of the proximity loss (keeps image close to original)
        x_orig:       the original clean image, used for the proximity loss

    Returns:
        grad: gradient tensor, same shape as x0_hat
    """

    # Make sure gradients can flow through x0_hat
    # We detach from the diffusion graph and create a fresh leaf
    x0_hat = x0_hat.detach().requires_grad_(True)

    # Forward pass through classifier
    logits = classifier(x0_hat)                        # (B, num_classes)

    # We want to maximise p(target | x̂₀), which means minimising
    # the cross-entropy loss toward the target label
    target = torch.tensor(
        [target_label] * x0_hat.shape[0],
        device=x0_hat.device
    )
    ce_loss = F.cross_entropy(logits, target)

    # L1 proximity: penalise deviation from the original image
    # This is what keeps the counterfactual close to the input —
    # we only want to change what's necessary to flip the label
    l1_loss = (x0_hat - x_orig.to(x0_hat.device)).abs().mean()

    # Combined loss
    loss = ce_loss + lambda_l1 * l1_loss

    # Backpropagate to get gradient w.r.t. x̂₀
    loss.backward()

    grad = x0_hat.grad.detach()

    return grad, ce_loss.item(), l1_loss.item()


def apply_guidance(x0_hat, grad, lambda_c):
    """
    Apply the guidance gradient to x̂₀.

    We subtract the gradient because we're doing gradient descent on the loss
    (which we want to minimise). This nudges x̂₀ in the direction that makes
    the classifier more confident about the target label.

    Args:
        x0_hat:   estimated clean image
        grad:     gradient from get_guidance_gradient()
        lambda_c: float, guidance scale (how hard we push toward target)

    Returns:
        x0_hat_guided: nudged clean image estimate
    """
    x0_hat_guided = x0_hat - lambda_c * grad

    # Clip to valid image range after nudging
    x0_hat_guided = x0_hat_guided.clamp(-1, 1)

    return x0_hat_guided


def guided_reverse_step(x_t, t, ddpm, classifier, target_label,
                        lambda_c, lambda_l1, x_orig):
    """
    One full guided denoising step — this is what gets called in the loop.

    Combines:
      1. predict x̂₀ from x_t
      2. compute and apply guidance gradient to x̂₀
      3. take the reverse diffusion step using the nudged x̂₀

    The trick for step 3: we can't just pass the nudged x̂₀ directly to
    reverse_step() because that function re-estimates x̂₀ internally.
    Instead we reconstruct a "guided x_t" that is consistent with the
    nudged x̂₀, and pass that to reverse_step().

    Args:
        x_t:          noisy image at timestep t
        t:            current timestep (int)
        ddpm:         DDPM instance
        classifier:   ResNet-18 under observation
        target_label: int
        lambda_c:     guidance scale
        lambda_l1:    proximity loss weight
        x_orig:       original clean image

    Returns:
        x_{t-1}:  denoised image for next step
        info:     dict with loss values for logging
    """

    # Step 1: estimate clean image from current noisy state
    x0_hat = ddpm.predict_x0(x_t, t)

    # Step 2: compute guidance gradient and apply it
    grad, ce_loss, l1_loss = get_guidance_gradient(
        x0_hat, classifier, target_label, lambda_l1, x_orig
    )
    x0_hat_guided = apply_guidance(x0_hat, grad, lambda_c)

    # Step 3: reconstruct a guided x_t consistent with x0_hat_guided
    # We re-noise x0_hat_guided back to level t, then take the reverse step.
    # This ensures the reverse step's internal x̂₀ estimate aligns with
    # the direction we guided it.
    alpha_prod = ddpm.scheduler.alphas_cumprod[t]
    sqrt_alpha  = alpha_prod ** 0.5
    sqrt_one_minus = (1 - alpha_prod) ** 0.5

    # Extract the noise that was in x_t
    with torch.no_grad():
        t_tensor = torch.tensor([t] * x_t.shape[0], device=x_t.device)
        eps_hat = ddpm.unet(x_t, t_tensor).sample

    # Reconstruct x_t using the guided x̂₀ and the same noise direction
    x_t_guided = sqrt_alpha * x0_hat_guided + sqrt_one_minus * eps_hat

    # Step 4: standard reverse step from the guided x_t
    x_prev = ddpm.reverse_step(x_t_guided, t)

    info = {
        "ce_loss": ce_loss,
        "l1_loss": l1_loss,
        "total_loss": ce_loss + lambda_l1 * l1_loss,
    }

    return x_prev, info