# Differentiable Simulation vs. Hard Carving Evaluation

This document explains the hybrid forward architecture used in GradMill (`train_csg.py`), detailing why optimization requires differentiable soft CSG subtraction (`smooth_max`), why evaluation requires step-count-invariant hard CSG subtraction (`max`), and how end-to-end gradients flow through the differentiable pipeline.

---

## 1. Why `forward_hard` is Non-Differentiable

In real machining and in interactive visualization (such as WebGPU `voxel.js`), material removal is an exact Boolean subtraction:

$$\phi_{\text{stock}}^{t+1} = \max(\phi_{\text{stock}}^t, -\phi_{\text{tool}}^t)$$

Implemented in `CSGSimulatorDelta.apply_cut_hard` (`simulator/csg_simulator.py`):
```python
@ti.kernel
def apply_cut_hard(self, t: ti.i32):
    for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
        p = ti.Vector([(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz])
        tool_d = self.tool_sdf_sharp(p, t)
        self.stock[t + 1, i, j, k] = ti.max(self.stock[t, i, j, k], -tool_d)
```

If we ran `forward_hard()` inside Taichi's autodiff tape (`ti.ad.Tape`), gradients would vanish ($\nabla = 0$) almost everywhere due to two mathematical barriers:

1. **Hard Maximum Subtraction (`ti.max`)**:
   The derivative $\frac{\partial}{\partial \text{tool\_d}} \max(\text{stock}, -\text{tool\_d})$ is strictly **$0$** whenever $-\text{tool\_d} < \text{stock}$. If a tool trajectory starts in open air outside the stock boundary, or passes slightly above a surface without physically intersecting it, the gradient is zero. The optimization receives zero directional feedback on where to move the cutter to begin removing material.
2. **Sharp Tool Geometry (`tool_sdf_sharp`)**:
   `tool_sdf_sharp` enforces hard clipping ($\max(d_{xy}, 0.0)$ and $\max(d_z, 0.0)$) to form sharp capped cylinder boundaries. Outside the immediate zero-crossing boundary layer, spatial derivatives hit hard gates or non-smooth discontinuities.

---

## 2. Why Optimization Requires Soft Carving (`forward`)

To enable gradient descent from arbitrary or random initial paths, `CSGSimulatorDelta.apply_cut` replaces hard Boolean subtraction with a differentiable Log-Sum-Exp ($LSE$) approximation:

$$\text{smooth\_max}(a, b, k) = \frac{1}{k} \log\left(e^{k a} + e^{k b}\right)$$

```python
@ti.kernel
def apply_cut(self, t: ti.i32):
    for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
        kv = self.k[None] / self.k_ref
        p = ti.Vector([(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz])
        tool_d = self.tool_sdf(p, t)
        self.stock[t + 1, i, j, k] = smooth_max(self.stock[t, i, j, k], -tool_d, kv)
```

Because exponential functions are strictly positive ($e^{kb} > 0$ for all $b \in \mathbb{R}$), $\frac{\partial}{\partial b} \text{smooth\_max}(a, b, k) > 0$ everywhere. Even when the tool is several voxels away from the stock ($\text{tool\_d} > 0$), every voxel exerts a smooth, non-zero gradient pushing or pulling the swept tool volume toward the target geometry.

However, Log-Sum-Exp introduces a **step-count accumulation bias**: summing exponentials sequentially over $T$ steps ($T \approx 100\text{--}200$) causes material around the tool to erode continuously. While beneficial for wide basin convergence during optimization, soft carving removes significantly more material than physical machining.

---

## 3. How Gradients Flow End-to-End (`train_csg.py`)

During training optimization (`algorithms/train_csg.py`), the forward pass is wrapped in Taichi's autodiff tape:

```python
with ti.ad.Tape(loss=sim.loss):
    sim.forward(T)  # Runs soft apply_cut inside Tape

grad = sim.tool_delta.grad.to_torch()[:T - 1]
```

The backward flow of gradients executes in reverse order across the simulation timeline:

```
[sim.loss]
    │  ▲  ∂loss / ∂stock[T-1]  (via compute_loss against target SDF)
    ▼  │
[sim.stock[T-1]]
    │  ▲  ∂stock[t+1] / ∂tool_d  (via smooth_max exponential tails)
    ▼  │
[sim.tool_sdf(p, t)]
    │  ▲  ∂tool_d / ∂tool_pos[t]  (via continuous swept-capsule distance field)
    ▼  │
[sim.tool_pos[t]]
    │  ▲  ∂tool_pos / ∂tool_delta  (via advance_position / reconstruct_positions)
    ▼  │
[sim.tool_delta.grad] ──► PyTorch params.grad ──► Adam / SGD Optimizer
```

1. **Loss Computation (`compute_loss`)**: Evaluates occupancy discrepancies between the final carved stock `sim.stock[T-1]` and the target shape `sim.target`, seeding `stock.grad`.
2. **Soft Carving (`apply_cut`)**: Taichi autodiff backpropagates through `smooth_max`. Voxel occupancy errors push back on `tool_d` at every timestep $t$.
3. **Swept SDF (`tool_sdf`)**: Converts spatial scalar gradients $\frac{\partial L}{\partial \text{tool\_d}}$ into 3D vector forces $\frac{\partial L}{\partial \text{tool\_pos}[t]}$ on the segment endpoints.
4. **Trajectory Scan (`advance_position` / `reconstruct_positions`)**: Backpropagates through cumulative sum scans and differentiable feed/rapid speed clipping to accumulate total derivatives in `sim.tool_delta.grad`.
5. **PyTorch Bridge**: `sim.tool_delta.grad.to_torch()` transfers the exact trajectory gradients to PyTorch tensors (`params.grad`) to update neural network or waypoint parameters.

---

## 4. The Hybrid Architecture: Decoupling Optimization and Evaluation

To preserve end-to-end differentiability during training while ensuring accurate physical metrics and visualization, GradMill separates the two execution regimes:

| Execution Regime | Method Called | Tape Active? | Carving Union | Tool Geometry | Use Cases |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Optimization Pass** | `sim.forward(T)` | **Yes** (`ti.ad.Tape`) | Differentiable `smooth_max` | Smoothed (`tool_sdf`) | Policy gradient computation (`tool_delta.grad`) inside training loops. |
| **Evaluation Pass** | `sim.forward_hard(T)` | **No** | Boolean `ti.max` | Crisp (`tool_sdf_sharp`) | Validation scores (`eval_metrics`), STL export (`export_stls`), video rendering (`render_run_video.py`). |

### Code Pattern in `train_csg.py`
Whenever evaluation metrics or mesh exports run outside the optimization step, `forward_hard(T)` is explicitly invoked:

```python
# 1. Training optimization step (soft pass inside Tape)
with ti.ad.Tape(loss=sim.loss):
    sim.forward(T)
optimizer.step()

# 2. Evaluation step (hard pass outside Tape)
if do_eval:
    sim.forward_hard(T)          # Carves physical stock without touching gradients
    m = eval_metrics(sim, T, dx) # Reports exact deployable Dice / ASD / HD95
```
