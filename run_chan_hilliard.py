import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from time import time
from tqdm import trange
from pathlib import Path

class ChanHilliard(nn.Module):
    """
    Implements the 3D Cahn-Hilliard equation using spectral methods (FFT)
    and periodic boundary conditions. GPU acceleration is supported via PyTorch.

    Attributes:
        nx, ny, nz: Grid resolution in each spatial direction.
        Lx, Ly, Lz: Physical domain size.
        dx, dy, dz: Grid spacing.
        M: Mobility parameter.
        kappa: Gradient energy coefficient.
        a: Free energy coefficient.
        dt: Time step size.
        steps: Number of time steps to run.
        device: torch.device (CPU or GPU).
    """

    def __init__(
        self,
        nx: int = 256,
        ny: int = 256,
        nz: int = 256,
        Lx: float = 100.0,
        Ly: float = 100.0,
        Lz: float = 100.0,
        M: float = 3.0,
        kappa: float = 2.5,
        a: float = 3.0,
        dt: float = 0.01,
        steps: int = 5000,
        device: torch.device = None,
    ):
        super().__init__()

        # Grid and domain parameters
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dz = Lz / nz

        self.dt = dt
        self.steps = steps
        self.a = a
        self.kappa = kappa
        self.M = M

        # Use GPU if available
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Initialize concentration field
        self.c = self.init_simulation().to(self.device)

        # Precompute Fourier space components
        self.precompute_frequency_domain()

    def init_simulation(self):
        """
        Initializes the concentration field with small random noise around 0.5.
        """
        c = 0.5 + 0.01 * torch.randn((self.nx, self.ny, self.nz), dtype=torch.float32)
        return c

    def f_prime(self):
        """
        Derivative of the double-well potential f(c) = a c²(1 - c)².
        """
        return 2 * self.a * self.c * (self.c - 1) * (2 * self.c - 1)

    def clip_0_1(self, x):
        """
        Ensures the concentration field stays within physical bounds [0, 1].
        """
        return torch.clamp(x, 0.0, 1.0)

    def precompute_frequency_domain(self):
        """
        Precomputes the wave vectors and denominator for the spectral solution.
        """
        kx = 2 * torch.pi * torch.fft.fftfreq(self.nx, d=self.dx).to(self.device)
        ky = 2 * torch.pi * torch.fft.fftfreq(self.ny, d=self.dy).to(self.device)
        kz = 2 * torch.pi * torch.fft.fftfreq(self.nz, d=self.dz).to(self.device)

        Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing="ij")
        K_squared = Kx**2 + Ky**2 + Kz**2

        # Registering as non-trainable tensors for GPU compatibility
        self.register_buffer("K_squared", K_squared)
        self.register_buffer(
            "denominator", 1 + self.dt * self.M * self.kappa**2 * K_squared**2
        )

    def forward(self):
        """
        Performs one time step update of the Cahn-Hilliard equation using spectral methods.
        """
        c_hat = torch.fft.fftn(self.c)
        f_hat = torch.fft.fftn(self.f_prime())
        numerator = c_hat - self.dt * self.M * self.K_squared * f_hat
        c_new = torch.real(torch.fft.ifftn(numerator / self.denominator))

        # Update the field with clamped values
        self.c = self.clip_0_1(c_new)
        return self.c

    def run(self, show: bool = False, show_interval: int = 500, save_path:Path=Path('result_img')):
        """
        Runs the simulation for the given number of steps.

        Args:
            show: If True, saves mid-slice plots every `show_interval` steps.
            show_interval: Interval of steps at which to plot/save a 2D slice.
        """
        start_time = time()
        if show:
            save_path.mkdir(exist_ok=True)

        for step in trange(self.steps):
            # You're actually calling the __call__ method of the ChanHilliard object
            # and because ChanHilliard is a subclass
            # of nn.Module, and you’ve defined a forward method, 
            # PyTorch automatically makes self() equivalent to calling self.forward().
            c = self()

            if show and step % show_interval == 0:
                mid_slice = c[:, :, self.nz // 2].detach().cpu().numpy()
                plt.imshow(mid_slice, cmap="RdBu", vmin=0, vmax=1)
                plt.title(f"Step {step}")
                plt.pause(0.01)
                plt.savefig(save_path.joinpath(f"step_{step:04}.png"))
                plt.clf()

        end_time = time()
        print(f"Simulation finished in {end_time - start_time:.2f} seconds")
        print(
            f"Final field: shape={c.shape}, min={c.min():.4f}, max={c.max():.4f}, mean={c.mean():.4f}, std={c.std():.4f}"
        )
        return c


if __name__ == "__main__":
    model = ChanHilliard(nx=256, ny=256, nz=256, steps=5000, device=torch.device("cuda:0"))
    model = torch.compile(model)

    final_c = model.run(show=False)
