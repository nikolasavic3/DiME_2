# ddpm.py
import torch
from diffusers import DDPMScheduler, UNet2DModel

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DDPM:
    """
    Thin wrapper around a pretrained DDPM.

    We only need three things for DiME:
      1. forward()   — corrupt a clean image x0 to noise level t
      2. predict_x0() — given noisy x_t, estimate the clean image
      3. reverse_step() — take one denoising step from t to t-1

    We use diffusers' DDPMScheduler which gives us the noise schedule
    (the alphas and betas) without us having to define them ourselves.
    """

    def __init__(self, model_id="google/ddpm-celebahq-256", timesteps=1000):
        self.device = get_device()
        print(f"Using device: {self.device}")

        # The scheduler holds the noise schedule: all the alphas, betas,
        # and precomputed values like sqrt_alpha_prod we need for q(x_t|x_0)
        self.scheduler = DDPMScheduler.from_pretrained(
            model_id,
        )
        # Override total timesteps if needed
        self.scheduler.set_timesteps(timesteps)
        self.T = timesteps

        # The UNet: takes (noisy_image, timestep) → predicted noise
        self.unet = UNet2DModel.from_pretrained(model_id)
        self.unet = self.unet.to(self.device)
        self.unet.eval()  # we never train this

    def forward(self, x0, t):
        """
        Forward process: corrupt clean image x0 to noise level t.

        Mathematically: x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε
        where ε ~ N(0, I) and ᾱ_t is the cumulative noise schedule.

        This is the closed-form solution — we don't have to apply noise
        t times one-by-one, we can jump directly to any noise level.

        Args:
            x0: clean image, shape (B, C, H, W), values in [-1, 1]
            t:  int, noise level (0 = clean, T = pure noise)

        Returns:
            x_t: noisy image at level t, same shape as x0
            noise: the noise that was added (we'll need this later)
        """
        x0 = x0.to(self.device)

        # Sample noise
        noise = torch.randn_like(x0)

        # t needs to be a tensor of shape (B,) for the scheduler
        t_tensor = torch.tensor([t] * x0.shape[0], device=self.device)

        # The scheduler computes sqrt(ᾱ_t) and sqrt(1-ᾱ_t) for us
        x_t = self.scheduler.add_noise(x0, noise, t_tensor)

        return x_t, noise

    def predict_x0(self, x_t, t):
        """
        Given noisy image x_t at timestep t, estimate the clean image x̂₀.

        This is the key quantity DiME applies classifier guidance to.
        The UNet predicts the noise ε, and then we rearrange:

            x_t = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * ε
            =>  x̂₀ = (x_t - sqrt(1-ᾱ_t) * ε̂) / sqrt(ᾱ_t)

        We need x̂₀ (not x_{t-1}) because the classifier was trained on
        clean images — it can't meaningfully process pure noise.

        Args:
            x_t: noisy image, shape (B, C, H, W)
            t:   int, current timestep

        Returns:
            x0_hat: estimated clean image, shape (B, C, H, W)
        """
        x_t = x_t.to(self.device)
        t_tensor = torch.tensor([t] * x_t.shape[0], device=self.device)

        with torch.no_grad():
            # UNet predicts the noise component ε̂
            eps_hat = self.unet(x_t, t_tensor).sample

        # Retrieve precomputed schedule values for this timestep
        # These are cumulative products: ᾱ_t
        alpha_prod = self.scheduler.alphas_cumprod[t]
        sqrt_alpha_prod     = alpha_prod ** 0.5
        sqrt_one_minus_alpha = (1 - alpha_prod) ** 0.5

        # Rearrange the forward equation to solve for x̂₀
        x0_hat = (x_t - sqrt_one_minus_alpha * eps_hat) / sqrt_alpha_prod

        # Clip to valid image range — the estimate can drift outside [-1,1]
        x0_hat = x0_hat.clamp(-1, 1)

        return x0_hat

    def reverse_step(self, x_t, t):
        """
        Take one denoising step: x_t → x_{t-1}.

        This is the standard DDPM reverse step (no guidance here —
        guidance is applied separately in guidance.py by modifying x_t
        before this step is called).

        Args:
            x_t: noisy image at timestep t
            t:   int, current timestep

        Returns:
            x_{t-1}: slightly less noisy image
        """
        x_t = x_t.to(self.device)
        t_tensor = torch.tensor([t] * x_t.shape[0], device=self.device)

        with torch.no_grad():
            eps_hat = self.unet(x_t, t_tensor).sample

        # scheduler.step() implements the full DDPM reverse formula:
        # x_{t-1} = mu(x_t, x̂₀) + sigma_t * z   where z ~ N(0,I)
        output = self.scheduler.step(eps_hat, t, x_t)

        return output.prev_sample