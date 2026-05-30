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
        x0_hat:       estimated clean image, shape (B, C, H, W)
        classifier:   the model under observation (ResNet-18)
        target_label: int, the attribute class we want to flip toward
        lambda_l1:    float, weight of the proximity loss
        x_orig:       the original clean image, used for the proximity loss

    Returns:
        grad:     raw gradient tensor, same shape as x0_hat
        ce_loss:  float, classification loss value
        l1_loss:  float, proximity loss value
    """
    x0_hat = x0_hat.detach().requires_grad_(True)
    use_cuda = x0_hat.device.type == "cuda"

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
        logits = classifier(x0_hat)

    target = torch.full((x0_hat.shape[0],), target_label, device=x0_hat.device, dtype=torch.long)
    ce_loss = F.cross_entropy(logits, target)
    l1_loss = (x0_hat - x_orig.to(x0_hat.device)).abs().mean()
    loss = ce_loss + lambda_l1 * l1_loss

    grad = torch.autograd.grad(loss, x0_hat, retain_graph=False, create_graph=False)[0].detach()

    return grad, ce_loss.item(), l1_loss.item()


def apply_guidance(x0_hat, grad, lambda_c, mask=None):
    """
    Apply the guidance gradient to x̂₀, optionally masked by a Grad-CAM spatial map.

    The Grad-CAM mask attenuates the gradient spatially so that classifier
    guidance is concentrated in semantically relevant regions (e.g., mouth for
    Smiling) while background pixels receive reduced updates.

    The mask is soft: non-masked regions still receive 20% of the gradient
    so the diffusion process stays globally coherent.

    Args:
        x0_hat:   estimated clean image
        grad:     raw gradient from get_guidance_gradient()
        lambda_c: float, guidance scale
        mask:     optional (1, 1, H, W) spatial mask in [0, 1] from Grad-CAM

    Returns:
        x0_hat_guided: updated clean image estimate
    """
    if mask is not None:
        # Soft mask: important regions get full gradient, rest get 20%
        soft_mask = 0.2 + 0.8 * mask.to(grad.device)
        grad = grad * soft_mask

    x0_hat_guided = x0_hat - lambda_c * grad
    return x0_hat_guided.clamp(-1, 1)


def guided_reverse_step(x_t, t, ddpm, classifier, target_label,
                        lambda_c, lambda_l1, x_orig, mask=None):
    """
    One full guided denoising step.

    Combines:
      1. predict x̂₀ from x_t
      2. compute guidance gradient, normalize, apply Grad-CAM mask, apply step
      3. reconstruct a guided x_t consistent with the nudged x̂₀
      4. take the standard reverse diffusion step

    Args:
        x_t:          noisy image at timestep t
        t:            current timestep (int)
        ddpm:         DDPM instance
        classifier:   ResNet-18 under observation
        target_label: int
        lambda_c:     guidance scale (step size in normalized gradient space)
        lambda_l1:    proximity loss weight
        x_orig:       original clean image
        mask:         optional Grad-CAM spatial mask, (1, 1, H, W) in [0, 1]

    Returns:
        x_{t-1}: denoised image for next step
        info:    dict with loss values for logging
    """
    x0_hat, eps_hat = ddpm.predict_x0(x_t, t, return_eps=True)

    grad, ce_loss, l1_loss = get_guidance_gradient(
        x0_hat, classifier, target_label, lambda_l1, x_orig
    )
    x0_hat_guided = apply_guidance(x0_hat, grad, lambda_c, mask=mask)

    # Reconstruct x_t consistent with the guided x̂₀, then reverse step
    alpha_prod = ddpm.alphas_cumprod[t]
    sqrt_alpha = alpha_prod ** 0.5
    sqrt_one_minus = (1 - alpha_prod) ** 0.5
    x_t_guided = sqrt_alpha * x0_hat_guided + sqrt_one_minus * eps_hat

    x_prev = ddpm.reverse_step(x_t_guided, t)

    info = {
        "ce_loss": ce_loss,
        "l1_loss": l1_loss,
        "total_loss": ce_loss + lambda_l1 * l1_loss,
    }

    return x_prev, info
