import numpy as np
import torch
import taichi as ti
from matplotlib import pyplot as plt
import os
import shutil
import tempfile
import datetime
 
try:
    from taichi.tools import VideoManager
except Exception:  # older Taichi (< 1.0) exposed it as ti.VideoManager
    from taichi import VideoManager

from simulator.csg_metrics import _gouge
from simulator.csg_metrics import _residual
from simulator.csg_simulator import CSGSimulatorDelta
from simulator.csg_metrics import *

T = 64
N_ITERS = 128
LR = 5e-3
RENDER_EVERY = 1  # replay the full trajectory animation every N Adam iters


RECORD_VIDEO = True
VIDEO_FPS = 24
VIDEO_EVERY = 1   # save a video every N iterations

RUN_DIR = os.path.join("runs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
VIDEO_DIR = os.path.join(RUN_DIR, "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)
print(f"[run] saving videos to {VIDEO_DIR}")

# --- Setup ---
sim = CSGSimulatorDelta(resolution=32, max_steps=T, k_init=10.0, target_shape="sphere",
                   tool_start=(0.5, 0.5, 1.0))
sim.target_params["radius"][None] = 0.4
sim.target_params["center"][None] = [0.5, 0.5, 0.5]
sim.tool_radius[None] = 0.05
sim.tool_height[None] = 0.15
sim.bake_target_grid()
sim.set_target_volume()

R = sim.resolution
dx = sim.dx

# --- Init parameters (T-1 per-step displacements) ---
init = np.random.uniform(-0.05, 0.05, size=(T - 1, 3)).astype(np.float32)
params = torch.tensor(init, requires_grad=True)
opt = torch.optim.Adam([params], lr=LR)


X = []
losses = []
gradients = []
gouges, residuals = [], []
dices, asds, hs95s = [], [], []


# --- GUI for live rendering ---
gui = ti.GUI("CSG Training", res=(1024, 768))

def _frame_uint8(buffer_field):
    """Pull the raymarch buffer to a uint8 RGB array.
 
    raymarch_buffer is shape (W, H, 3) float in [0, 1] -- the same (width,
    height) ordering Taichi's image tools expect, so we DON'T transpose here.
    VideoManager applies Taichi's orientation, producing a frame oriented
    exactly like the GUI window.
    """
    img = buffer_field.to_numpy()
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def record_iteration_video(
    sim,
    gui,
    T,
    out_path,
    fps=24,
    label="",
    keep_frames=False,
    mp4=True,
    gif=False,
):
    """Render stock[0..T], show it live, and save one video for this iteration.
 
    Drop-in replacement for render_trajectory() when you also want a file.
    Each frame is rendered once and reused for both the GUI and the encoder.
 
    sim         : CSGSimulatorDelta, already forwarded for this iteration.
    gui         : the ti.GUI for live display. Pass None to skip the live view.
    T           : number of trajectory steps.
    out_path    : final video path, e.g. runs/<ts>/videos/iter_0007.mp4
    fps         : frames per second of the output.
    label       : caption shown on the live GUI (not burned into the video).
    keep_frames : keep the intermediate PNGs instead of deleting the temp dir.
    mp4 / gif   : which container(s) to build.
    """
    frames_dir = tempfile.mkdtemp(prefix="csg_frames_")
    vm = VideoManager(output_dir=frames_dir, framerate=fps, automatic_build=False)
 
    try:
        for t in range(T):
            if gui is not None and not gui.running:
                break
 
            sim.set_current_step(t)
            sim.render(
                cam_pos=(2.0, 2.0, 1.6),
                cam_target=(0.5, 0.5, 0.5),
                cam_up=(0.0, 0.0, 1.0),
                show_stock=True,
                show_target=True,
                show_tool=(t < T),  # hide tool on the final frame
            )
 
            # Same buffer feeds both the encoder and the live window.
            vm.write_frame(_frame_uint8(sim.raymarch_buffer))
 
            if gui is not None:
                gui.set_image(sim.raymarch_buffer)
                gui.text(
                    f"{label}  step {t}/{T}",
                    pos=(0.02, 0.97),
                    color=0xFFFFFF,
                    font_size=18,
                )
                gui.show()
 
        vm.make_video(gif=gif, mp4=mp4)
 
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if mp4:
            shutil.move(vm.get_output_filename(".mp4"), out_path)
        if gif:
            shutil.move(
                vm.get_output_filename(".gif"),
                os.path.splitext(out_path)[0] + ".gif",
            )
    except Exception as e:
        print(f"[video] failed to build {out_path}: {e}")
        print(
            "[video] mp4/gif encoding needs ffmpeg on your PATH "
            "(e.g. `conda install ffmpeg`, `brew install ffmpeg`, "
            "or `apt install ffmpeg`)."
        )
    finally:
        if not keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)
 

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

        # Push current params (the displacements) into Taichi's tool_delta.
        # tool_delta has shape max_steps; we fill the first T-1 entries.
        sim.tool_delta.from_torch(params.detach())

        # Forward + backward. forward() calls reconstruct_positions() first,
        # so the cumulative-sum scan that builds tool_pos from tool_delta is
        # inside the Tape and its gradients are recorded.
        with ti.ad.Tape(loss=sim.loss):
            sim.forward(T)

        # Pull the gradient w.r.t. the DELTAS (not positions) back into PyTorch.
        # tool_delta.grad already contains both the direct path (cut t's swept
        # segment) and the indirect path (every later cut), summed by autodiff.
        params.grad = sim.tool_delta.grad.to_torch()[:T - 1]
        opt.step()
        opt.zero_grad()

        loss = float(sim.loss[None])
        losses.append(loss)
        print(f"iter {it:3d} | loss = {loss:.5f}")

        # --- Gradient diagnostics ---
        # Gradient is w.r.t. the deltas; positions are read back from the
        # reconstructed tool_pos field (slots 0..T-1) for reporting.
        grad = sim.tool_delta.grad.to_torch()[:T - 1]   # (T-1, 3)
        pos = sim.tool_pos.to_torch()[:T]               # (T, 3) reconstructed

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
        print(f"# zero-grad steps = {(grad_norms < 1e-10).sum().item()} / {T - 1}")
        print(f"# nan/inf in grad = {(~torch.isfinite(grad)).sum().item()}")

        # Show the actual numbers for first, middle, last few timesteps.
        # pos has T rows (reconstructed positions); grad has T-1 rows (deltas).
        # Sample within the delta range so both lookups are valid.
        print("\nper-step detail  [pos] -> [delta grad]")
        sample_idx = [0, 1, 2, (T - 1)//4, (T - 1)//2, 3*(T - 1)//4,
                      T-4, T-3, T-2]
        for t in sample_idx:
            p = pnp[t]
            g = gnp[t]
            print(f"  t={t:3d}  pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  "
                f"grad=({g[0]:+.3e},{g[1]:+.3e},{g[2]:+.3e})  ‖g‖={grad_norms[t]:.3e}")

        # Record a video of the full trajectory for this iteration.
        if RECORD_VIDEO and it % VIDEO_EVERY == 0:
            out_path = os.path.join(VIDEO_DIR, f"iter_{it:04d}.mp4")
            record_iteration_video(sim, gui, T, out_path,fps=VIDEO_FPS, label=f"iter {it}")

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
    # Save both the learned displacements and the reconstructed positions.
    np.save("trajectory_deltas.npy", params.detach().numpy())
    sim.tool_delta.from_torch(params.detach())
    sim.reconstruct_positions(T - 1)
    np.save("trajectory.npy", sim.tool_pos.to_torch()[:T].numpy())
    print("Saved deltas to trajectory_deltas.npy and positions to trajectory.npy")

    # Final replay
    if gui.running:
        sim.tool_delta.from_torch(params.detach())
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