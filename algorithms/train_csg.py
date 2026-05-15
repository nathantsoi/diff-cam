import numpy as np
import torch
import taichi as ti

from simulator.csg_simulator import CSGSimulator


T = 64
N_ITERS = 100
LR = 5e-3
RENDER_EVERY = 1  # replay the full trajectory animation every N Adam iters

# --- Setup ---
sim = CSGSimulator(resolution=32, max_steps=T, k_init=10.0, target_shape="sphere")
sim.target_params["radius"][None] = 0.3
sim.target_params["center"][None] = [0.5, 0.5, 0.5]
sim.tool_radius[None] = 0.05
sim.tool_height[None] = 0.15

# --- Init parameters (T tool positions, random in the unit cube) ---
init = np.random.uniform(0.2, 0.8, size=(T, 3)).astype(np.float32)
params = torch.tensor(init, requires_grad=True)
opt = torch.optim.Adam([params], lr=LR)

# --- GUI for live rendering ---
gui = ti.GUI("CSG Training", res=(1024, 768))


def render_trajectory(sim, T, label=""):
    """Step through stock[0..T], showing the tool at each position.

    sim.forward() has already populated every stock slot for the current
    trajectory, so we just advance current_step and re-render each frame.
    """
    for t in range(T + 1):
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


# --- Training loop ---
for it in range(N_ITERS):
    if not gui.running:
        break

    # Push current params into Taichi
    sim.tool_pos.from_torch(params.detach())

    # Forward + backward (Tape records kernels and runs them in reverse).
    # This populates stock[0..T] for the current trajectory.
    with ti.ad.Tape(loss=sim.loss):
        sim.forward(T)

    # Pull gradient back into PyTorch and step
    params.grad = sim.tool_pos.grad.to_torch()
    opt.step()
    opt.zero_grad()

    print(f"iter {it:3d} | loss = {float(sim.loss[None]):.5f}")

    # Every few iterations, replay the trajectory in the GUI so you can
    # watch the tool move and the stock get carved away.
    if it % RENDER_EVERY == 0:
        render_trajectory(sim, T, label=f"iter {it}")

# --- Save result ---
np.save("trajectory.npy", params.detach().numpy())
print("Saved to trajectory.npy")

# Final replay
if gui.running:
    sim.tool_pos.from_torch(params.detach())
    sim.forward(T)
    render_trajectory(sim, T, label="final")