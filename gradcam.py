import torch
import torch.nn.functional as F


def compute_gradcam_mask(classifier, x0, target_label):
    """
    Grad-CAM spatial mask: where does the classifier look when predicting target_label?

    Hooks ResNet-18's final conv block (layer4), runs one forward+backward pass
    on the clean original image x0, and returns a spatial heatmap that peaks
    in regions most influential for target_label.

    Args:
        classifier:   CelebAClassifier instance (ResNet-18 wrapper)
        x0:           clean original image, (1, C, H, W) in [-1, 1]
        target_label: int, the class we want to flip toward

    Returns:
        mask: (1, 1, H, W) float tensor in [0, 1], upsampled to image resolution
    """
    feats, grads = {}, {}

    fh = classifier.model.layer4.register_forward_hook(
        lambda m, i, o: feats.update({"v": o})
    )
    bh = classifier.model.layer4.register_full_backward_hook(
        lambda m, gi, go: grads.update({"v": go[0]})
    )

    # Use float32 and fresh leaf to avoid conflicts with autocast / inference_mode
    x = x0.detach().float().requires_grad_(True)
    logits = classifier.model(x)
    classifier.zero_grad()
    logits[0, target_label].backward()

    fh.remove()
    bh.remove()

    feat = feats["v"].float()   # (1, C, h, w)
    grad = grads["v"].float()   # (1, C, h, w)

    # Channel weights = mean gradient per channel (standard Grad-CAM)
    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = F.relu((weights * feat).sum(dim=1, keepdim=True))  # (1, 1, h, w)

    if cam.max() > 0:
        cam = cam / cam.max()

    # Upsample to full image resolution
    cam = F.interpolate(cam, size=x0.shape[2:], mode="bilinear", align_corners=False)
    return cam.detach()
