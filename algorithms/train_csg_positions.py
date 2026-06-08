import numpy as np
import torch
import taichi as ti
from matplotlib import pyplot as plt

from simulator.csg_metrics import _gouge
from simulator.csg_metrics import _residual
from simulator.csg_simulator import CSGSimulator
from simulator.csg_simulator_delta import CSGSimulatorDelta
from simulator.csg_metrics import *

T = 64
N_ITERS = 128
LR = 5e-3
RENDER_EVERY = 1  # replay the full trajectory animation every N Adam iters


# --- Setup ---
sim = CSGSimulator(resolution=32, max_steps=T, k_init=10.0, target_shape="sphere")
sim.target_params["radius"][None] = 0.4
sim.target_params["center"][None] = [0.5, 0.5, 0.5]
sim.tool_radius[None] = 0.05
sim.tool_height[None] = 0.15
sim.bake_target_grid()
sim.set_target_volume()

R = sim.resolution
dx = sim.dx

# --- Init parameters (T tool positions, random in the unit cube) ---
init = np.random.uniform(0.2, 0.8, size=(T, 3)).astype(np.float32)
params = torch.tensor(init, requires_grad=True)
opt = torch.optim.Adam([params], lr=LR)


X = []
losses = []
gradients = []
gouges, residuals = [], []
dices, asds, hs95s = [], [], []


# --- GUI for live rendering ---
gui = ti.GUI("CSG Training", res=(1024, 768))


def render_trajectory(sim, T, label=""):
    """Step through stock[0..T], showing the tool at each position.

    sim.forward() has already populated every stock slot for the current
    trajectory, so we just advance current_step and re-render each frame.
    """
    for t in range(T):
        if not gui.running:
            return
        sim.set_current_step(t)
        sim.render(
            cam_pos=(2.0, 2.0, 1.6),
            cam_target=(0.5, 0.5, 0.5),
            cam_up=(0.0, 0.0, 1.0),
            show_stock=True,
            show_target=True,
            show_tool=(t < T),  # hide tool on the final frame
        )
        gui.set_image(sim.raymarch_buffer)
        gui.text(f"{label}  step {t}/{T}", pos=(0.02, 0.97),
                 color=0xFFFFFF, font_size=18)
        gui.show()


def train():
    for it in range(N_ITERS):
        X.append(it)
        if not gui.running:
            break

        # Push current params into Taichi
        sim.tool_pos.from_torch(params.detach())

        # Forward + backward
        with ti.ad.Tape(loss=sim.loss):
            sim.forward(T)

        # Pull gradient back into PyTorch and step
        params.grad = sim.tool_pos.grad.to_torch()
        opt.step()
        opt.zero_grad()

        loss = float(sim.loss[None])
        losses.append(loss)
        print(f"iter {it:3d} | loss = {loss:.5f}")

        # --- Gradient diagnostics ---
        grad = sim.tool_pos.grad.to_torch()[:T]  # (T, 3)
        pos = params.detach()                     # (T, 3)

        grad_norms = grad.norm(dim=1)             # per-timestep ‖∇‖
        gnp = grad.numpy()
        pnp = pos.numpy()

        print(f"\n--- iter {it} gradient report ---")
        print(f"loss              = {float(sim.loss[None]):.6f}")
        print(f"‖grad‖ global     = {grad.norm().item():.6e}")
        print(f"‖grad‖ per-step   min={grad_norms.min().item():.3e}  "
            f"median={grad_norms.median().item():.3e}  "
            f"max={grad_norms.max().item():.3e}")
        print(f"grad mean (xyz)   = {gnp.mean(axis=0)}")
        print(f"grad std  (xyz)   = {gnp.std(axis=0)}")
        print(f"grad abs.max (xyz)= {np.abs(gnp).max(axis=0)}")
        print(f"# zero-grad steps = {(grad_norms < 1e-10).sum().item()} / {T}")
        print(f"# nan/inf in grad = {(~torch.isfinite(grad)).sum().item()}")

        # Show the actual numbers for first, middle, last few timesteps
        print("\nper-step detail  [pos] -> [grad]")
        sample_idx = [0, 1, 2, T//4, T//2, 3*T//4, T-3, T-2, T-1]
        for t in sample_idx:
            p = pnp[t]
            g = gnp[t]
            print(f"  t={t:3d}  pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  "
                f"grad=({g[0]:+.3e},{g[1]:+.3e},{g[2]:+.3e})  ‖g‖={grad_norms[t]:.3e}")

        # Render every few interations
        if it % RENDER_EVERY == 0:
            render_trajectory(sim, T, label=f"iter {it}")

        stock = sim.stock.to_numpy()[T - 1]
        target = sim.target.to_numpy()

        pred_mask = sdf_to_mask(stock)
        target_mask = sdf_to_mask(target)

        gouge = _gouge(pred_mask, target_mask) * (dx**3)
        residual = _residual(pred_mask, target_mask) * (dx**3)
        dice = dice_score(pred_mask, target_mask)
        asd = average_surface_distance(pred_mask, target_mask) * dx
        haus95 = hd95(pred_mask, target_mask) * dx

        gouges.append(gouge)
        residuals.append(residual)
        dices.append(dice)
        asds.append(asd)
        hs95s.append(haus95)

        print("Gouge :", gouge)
        print("Residual :", residual)
        print("Dice :", dice)
        print("ASD  :", asd)
        print("HD95 :", haus95)
        print()

    # --- Save result ---
    np.save("trajectory.npy", params.detach().numpy())
    print("Saved to trajectory.npy")

    # Final replay
    if gui.running:
        sim.tool_pos.from_torch(params.detach())
        sim.forward(T)
        render_trajectory(sim, T, label="final")
    

def plot():
    target_volume = sim.target_volume[None]
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))

    axs[0][0].plot(X, dices)
    axs[0][0].set_xlabel("Iteration")
    axs[0][0].set_title("Dice Score")

    axs[0][1].plot(X, asds)
    axs[0][1].set_xlabel("Iteration")
    axs[0][1].set_title("ASD")
    axs[0][1].set_ylim(0, 1)

    axs[1][0].plot(X, hs95s)
    axs[1][0].set_xlabel("Iteration")
    axs[1][0].set_title("HD95")
    axs[1][0].set_ylim(0, 1)

    axs[1][1].plot(X, losses)
    axs[1][1].set_xlabel("Iteration")
    axs[1][1].set_title("Loss")

    axs[2][0].plot(X, gouges, label="Gouge Volume (should go down to 0)")
    axs[2][0].axhline(target_volume, color='r', linestyle='--', label='Target Volume (upper bound)')
    axs[2][0].legend()
    axs[2][0].set_xlabel("Iteration")
    axs[2][0].set_title("Gouge Volume")
    axs[2][0].set_ylim(0, 1)

    axs[2][1].plot(X, residuals, label="Residual Volume (should go down to 0)")
    axs[2][1].legend()
    axs[2][1].set_xlabel("Iteration")
    axs[2][1].set_title("Residual Volume")
    axs[2][1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


try:
    train()
finally:
    plot()