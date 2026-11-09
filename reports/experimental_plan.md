# DiffCAM experimental plan: positioning, baselines, and datasets

Assessment date: 2026-11-08  
Repository inspected: `core` at `a94cfde`

This plan synthesizes and reconciles:

- [Differentiable CNC Simulator Paper Framing.md](Differentiable%20CNC%20Simulator%20Paper%20Framing.md)
- [OVPR differentiable simulation proposal](ovpr-differentiable-simulation.tex)
- [Repository status and RSS plan](status-and-rss-plan.tex)

The assessment uses source inspection and public primary-source papers, repositories, and dataset documentation. External methods were not installed or benchmarked during this assessment. Public availability does not establish successful reproduction; pin revisions, verify licenses and dependencies, and run a small reproduction pilot before committing to a comparison.

## Recommendation

For the current repository, conventional CAM, derivative-free trajectory optimization (CMA-ES and CEM), and CNC-Net should be the primary baselines. PPO, SAC, and SHAC/AHAC become essential when DiffCAM implements the multi-part policy-learning claim in the OVPR proposal. Comparing the current per-instance Adam trajectory optimizer only against RL would confound single-instance trajectory refinement with amortized policy training and leave the strongest practical alternatives untested.

At the same time, the paper should retain the robotics theoretical framing from [Differentiable CNC Simulator Paper Framing.md](Differentiable%20CNC%20Simulator%20Paper%20Framing.md): **analytical smoothing (differentiable relaxed cutting) versus randomized smoothing (zero-order search over hard Boolean cutting)** across discontinuous Heaviside tool–workpiece boundaries (Suh et al., ICML 2022; Pang et al., T-RO 2023). Because CMA-ES and CEM on hard cutting are zero-order randomized-smoothing estimators over trajectory parameters, this theoretical duality can be tested rigorously in a per-instance trajectory optimization setting before introducing neural policy learning.

The central research question should be:

> How do first-order analytical gradients through a smoothly relaxed, history-dependent material removal kernel compare against zero-order randomized search over hard cutting when refining canonical CAM trajectories (facing, contouring, adaptive clearing), and can closed-loop disagreement control preserve smooth optimization gains under independent hard posted-program execution?

A credible RSS contribution needs a demonstrated mechanism or algorithm beyond combining SDF subtraction, linear annealing, Adam, and a soft holder penalty. The strongest candidate is a **disagreement-aware continuation and soft-to-hard de-biasing algorithm** that actively controls the discrepancy between relaxed multi-pass optimization and hard posted execution, supported by a de-confounded 2×2 solver comparison, dimensionality-scaling analysis, and physical machining evidence.

## 1. Current repository status

The current implementation supports a narrower, more focused first paper than the OVPR pitch or the full RL+DiffSim vision in the framing memo.

| Claim in the pitch / framing memo | Current evidence in `core` (`a94cfde`) | Consequence for the research plan |
|---|---|---|
| Continuous PSDF state avoids dense-grid memory | Stock is allocated as `(T+1, Nx, Ny, Nz)` in `float32` with gradients | Drop grid-free memory claims; report actual $8(T+1)N_xN_yN_z$ byte scaling |
| Sample-efficient policy learning (PPO/SAC/SHAC) | [`algorithms/optimize.py`](../algorithms/optimize.py) optimizes a per-instance delta trajectory | Focus the primary study on trajectory optimization; stage policy learning as Phase 7 |
| Analytic gradients produce collision-free paths | Soft holder penalties (evaluated on post-cut stock) and hard diagnostics exist | Fix holder chronology to pre-cut stock and enforce hard chronological safety gates |
| Flute-level CWE angle & mechanistic force model | Heuristic tip-engagement volume and stress-strength breakage proxy (`kc`, `f_ref`) | Frame process metrics around cycle time, air-cut fraction, and volumetric engagement rate ($\text{MRR}_t$) unless calibrated against force sensors |
| End-to-end machining/TAMP automation | Fixed single tool, analytic targets, supplied or random initial trajectory | Focus the first paper on single-tool continuous motion refinement from CAM seeds |
| Real-part evaluation | Ten Titan STEP models are present in [`parts/`](../parts/) | Target import, reachable single-tool setup definitions, and reproducible CAM seeds are required |

Relevant implementation:

- [`algorithms/optimize.py`](../algorithms/optimize.py): per-instance NumPy Adam, delta actions, linear sharpness annealing, and hard-Dice checkpoint selection.
- [`simulator/csg_simulator.py`](../simulator/csg_simulator.py): analytic primitive functions, dense stock history, smooth/hard cutting, motion clipping, and holder penalties.
- [`cam/`](../cam/): physical units, work coordinates, RS274/Haas posts, modal parser, exact-stop trajectory planning, and hard carve execution.
- [`parts/`](../parts/): `Titan-M1.step` through `Titan-M10.step`.

### Publication-blocking correctness issues to fix first

1. **Vertical and sloped cutter sweep approximation (`tool_sdf`, `tool_sdf_sharp`, `tool_sdf_tip`, `tool_sdf_sharp_tip`):** Both soft and hard cutter SDFs project the query point onto the toolpath segment in 2D XY (`pa_xy.dot(ba_xy) / ba_len2`) and linearly interpolate tool Z at that XY parameter. For a pure vertical plunge ($\Delta x = \Delta y = 0$), `h_param = 0`, so the SDF evaluates only the top endpoint rather than the swept vertical cylinder. For simultaneous 3-axis sloped ramps, the XY-closest point is not the 3D minimizer of the swept capped cylinder.
2. **Post-cut holder collision chronology (`compute_holder_penalty`, `holder_overlap_at`, `holder_min_clearance_at`, `holder_overlap_total`, `compute_diagnostics`):** Holder penetration is currently gated by `self.stock[t + 1, i, j, k]` (post-cut stock) rather than `self.stock[t, i, j, k]` (pre-cut stock), allowing a plunging cutter to credit material removed during step $t$ as already cleared for the holder above it.

The pitch and framing memo also require more precise language:

- Conventional B-Rep Boolean pipelines do not directly provide the volumetric trajectory derivatives needed here; saying that B-Reps inherently yield zero gradients in all formulations is too broad.
- Autodiff provides derivatives of the implemented smooth surrogate, not automatically unbiased derivatives of physical machining.
- Analytic primitives do not make the stored stock history grid-free, resolution-independent, or an exact Euclidean SDF after repeated CSG compositions.
- A finite soft collision penalty does not guarantee collision freedom; hard chronological verification of the posted program is mandatory.
- A single-tool trajectory optimizer implements the continuous motion refinement layer of a TAMP system; discrete tool selection and operation sequencing remain separate work.

## 2. Novelty relative to prior work

Recommended thesis:

> DiffCAM uses differentiable, history-dependent material removal to refine canonical CAM trajectories (facing, contouring, adaptive clearing), using closed-loop soft-to-hard disagreement control to bridge the gap between analytical smoothing and hard posted-path execution.

This should be supported by four contributions:

1. **Algorithmic mechanism:** A **Disagreement-Aware Continuation and Soft-to-Hard De-Biasing** method that dynamically calibrates soft-union over-erosion bias (`loss_shift`) and adapts sharpness $\kappa$ / trust-region step acceptance using periodic hard-carve disagreement feedback.
2. **Theoretical and empirical resolution of analytical vs. randomized smoothing in milling:** A controlled 2×2 comparison (First-Order Gradient vs. Zero-Order Search $\times$ Smooth vs. Hard Objective) across trajectory degrees of freedom $D$, testing when analytical gradients beat covariance/population search and how topological CAM initialization prevents local-minima trapping.
3. **Process and geometric refinement:** Simultaneous improvement of hard volumetric accuracy, cycle time, air-cutting fraction, and peak volumetric engagement under strict zero-gouge and chronological holder-safety gates.
4. **Independent NC and physical validation:** Verification that optimization gains survive G-code posting, exact-stop acceleration planning, independent swept-volume verification (CAMotics), and physical 3-axis machining in wax and Aluminum 6061.

Hard checkpoint selection and ordinary linear $k$-annealing are mandatory internal baselines; they must not, by themselves, be presented as the new algorithm.

### Closest related methods

**CNC-Net (Yavartanoo et al., CVPR 2024).** Self-supervised deep learning architecture that carves 3D CAD targets from scratch using implicit differentiable subtraction and includes per-instance fine-tuning. It must be addressed directly: first in the uninitialized-carving study (Experiment 1) and second on a compatible single-tool 3-axis shape subset. [Official code](https://github.com/myavartanoo/CNC-Net_PyTorch), [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yavartanoo_CNC-Net_Self-Supervised_Learning_for_CNC_Machining_Operations_CVPR_2024_paper.pdf).

**CNCGEN (Zhou et al., 2026).** Explicitly conditions 3-axis machining generation on evolving stock and uses a frozen learned Surrogate Verifier for differentiable geometric feedback. Evolving-stock reasoning alone is therefore not a novelty claim. DiffCAM's distinction is computing analytical derivatives directly through the volumetric removal mechanics without requiring supervised B-Rep datasets or a learned neural verifier. Note that the public HuggingFace release is a 20-example preview rather than the full dataset described in the paper. [Paper](https://arxiv.org/abs/2609.39738), [release](https://huggingface.co/datasets/jimmy-zhou/CNCGEN).

**Implicit neural-field process planning (Dutta et al., 2026), design-for-machinability (Chen & Kara, TOG 2026), and four-axis continuous toolpath optimization (Zhang et al., CGF 2024).** These already optimize continuous toolpath geometry or part machinability with gradients, ruling out a broad "first gradient-based machining" claim. DiffCAM differs by differentiating through sequential, history-dependent 3-axis stock removal and engagement mechanics for a fixed target part initialized from canonical CAM templates. [Neural-field paper](https://arxiv.org/html/2511.17578v2), [four-axis project](https://haisenzhao.github.io/FourAxis/index.html), [design-for-machinability](https://arxiv.org/abs/2610.15301).

**Lichter and Hadházi, "Towards Differentiable Machining" (2026).** The work's existence is confirmed only by a BME institutional mini-symposium announcement identifying it as a vision paper. Neither runnable code nor the detailed technical/experimental assertions attributed to it in the framing memo could be verified from primary sources. Do not rely on its unverified architecture or quantitative claims as the primary baseline contrast unless the full paper is obtained and verified. [Institutional announcement](https://www.mit.bme.hu/hirek/2026-02-23/33-miniszimpozium).

**Analytical vs. randomized smoothing and differentiable policy learning (Suh et al., ICML 2022; Pang et al., T-RO 2023; SHAC, ICLR 2022; AHAC, ICML 2024).** Suh et al. and Pang et al. establish the bias–variance tradeoff between first-order gradients through smoothed contact and zero-order randomized smoothing across Heaviside discontinuities. This directly grounds our analysis of smooth cutting gradients vs. hard-cutting CMA-ES/CEM and horizon truncation $H$. When policy learning is evaluated in Stage 7, SHAC and AHAC are required alongside PPO and SAC. [SHAC implementation](https://github.com/NVlabs/DiffRL), [AHAC project](https://adaptive-horizon-actor-critic.github.io/).

**Soft-material cutting and manipulation (DiSECt, RSS 2021; PlasticineLab, ICLR 2021; DiffSkill, ICLR 2022; TopoCut, 2025).** These motivate the RSS problem framing and demonstrate local-minima trapping in uninitialized gradient descent (DiffSkill), but soft-tissue FEM/MPM fracture mechanics are not direct 3-axis milling baselines. [DiSECt paper](https://arxiv.org/abs/2105.12244), [PlasticineLab](https://github.com/hzaskywalker/PlasticineLab).

## 3. Primary baselines and parameterization contract

| Condition / Baseline | Research question answered | Implementation and priority |
|---|---|---|
| `CAM-Base` (Unmodified CAM) | What is the nominal geometric and cycle-time performance of standard CAM? | **Mandatory anchor:** FreeCAD / algorithmic Facing, Contouring, and Adaptive Clearing seeds |
| `No-Init + Smooth-Adam` & `No-Init + Hard-CEM` | Do local gradients or zero-order search fail without topological CAM initialization? | **Mandatory (Exp. 1):** Straight-line / random initialization on pocket and slot tasks |
| `CNC-Net` | How does uninitialized / learned implicit primitive carving compare on 3D shapes? | **High-priority external baseline (Exp. 1 & 5):** Official PyTorch repo on compatible 3-axis task subset |
| `CAM + Hard-CMA-ES` & `CAM + Hard-CEM` | How well does randomized smoothing over exact hard cutting refine CAM paths? | **Mandatory (Exp. 2):** `pycma` (and `sep-CMA-ES` at high $D$) + CEM on hard simulation objective |
| `CAM + Smooth-CMA-ES` & `CAM + Smooth-CEM` | Does the smooth objective help derivative-free search, or are derivatives essential? | **Mandatory 2×2 control (Exp. 2):** `pycma` + CEM optimizing the exact same smooth objective as Adam |
| `CAM + Fixed-k Smooth-Adam` | What happens when optimizing analytical gradients without continuation? | **Mandatory internal baseline:** Fixed sharpness $k \in \{20, 60, 120\}$ |
| `CAM + Annealed Smooth-Adam` (Hard Selection) | How strong is standard linear $k$-annealing with hard checkpoint selection? | **Mandatory internal reference:** Current [`algorithms/optimize.py`](../algorithms/optimize.py) recipe |
| `CAM + Disagreement-Continuation Adam` (Ours) | Does closed-loop soft-to-hard de-biasing and disagreement trust-region control beat fixed annealing? | **Primary algorithmic contribution (Exp. 2 & 3):** Adaptive `loss_shift` + disagreement-gated $\kappa$/step control |

Implementation sources:

- [FreeCAD CAM / Path workbench](https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/CAM_Adaptive.md)
- [OpenCAMLib](https://github.com/aewallin/opencamlib), for additional surface-path geometry algorithms
- [pycma](https://github.com/CMA-ES/pycma), for standard and separable CMA-ES
- [CNC-Net official implementation](https://github.com/myavartanoo/CNC-Net_PyTorch)

### Match the parameterization and sweep dimensionality ($K$)

Raw per-step displacements `tool_delta` of shape $(T, 3)$ have $D = 3T$ parameters ($D = 768\text{–}1536$ for $T = 256\text{–}512$). Full-covariance CMA-ES (`pycma`) scales as $\mathcal{O}(D^2)$ and degrades above $D \approx 300$, whereas first-order Adam scales as $\mathcal{O}(D)$. To ensure a fair comparison that does not artificially handicap CMA-ES—while simultaneously testing how first-order derivatives scale with trajectory degrees of freedom—implement a **shared cubic B-spline / control-point residual parameterization**:

$$\Delta_t(\theta) = \Delta_t^{\text{CAM}} + \sum_{j=1}^{K} B_{t,j}\,\theta_j, \quad \theta \in \mathbb{R}^{K \times 3}, \quad B \in \mathbb{R}^{T \times K}$$

where $B$ is a fixed, precomputed cubic B-spline basis matrix mapping $K$ control points to $T$ simulator steps, and the backward pass projects gradients via $\nabla_\theta \mathcal{L} = B^\top (\nabla_\Delta \mathcal{L})$.

- Evaluate all methods across $K \in \{16, 32, 64, 128\}$ ($D \in \{48, 96, 192, 384\}$) using standard `pycma`, CEM, and Adam on identical parameter bounds and initializations ($\theta_0 = \mathbf{0}$, corresponding to unmodified `CAM-Base`).
- At full per-step resolution ($K = T$, $D = 3T$), compare Adam against diagonal/separable `sep-CMA-ES` (`CMAOptions({'CMA_diagonal': True})`) and CEM.
- Always preserve and report the unmodified `CAM-Base` ($\theta = \mathbf{0}$) result as the anchor.

### Separate derivatives from smoothing (The 2×2 Solver Contract)

Comparing Adam on a smooth objective against CMA-ES/CEM on a hard objective changes both the optimizer class and the objective landscape. Execute the full 2×2 matrix:

| Optimizer class | Smooth simulation objective ($\mathcal{L}_{\text{smooth}}$) | Hard simulation objective ($\mathcal{L}_{\text{hard}}$) |
|---|---|---|
| **First-order gradient** | Fixed-$k$ Adam, Annealed Adam, Disagreement-Continuation Adam, L-BFGS | Diagnostic only (subgradients exist only on smooth kinematics/barriers; volumetric removal gradient is zero a.e.) |
| **Zero-order / derivative-free** | `Smooth-CMA-ES`, `Smooth-CEM` | `Hard-CMA-ES`, `Hard-CEM` |

Score every resulting trajectory using the **exact same hard posted-path evaluator**.

For gradient verification and bias–variance analysis (Experiment 3):
- Verify Taichi autodiff against central finite differences of the **same smooth objective** $\mathcal{L}_{\text{smooth}}$ away from regime-switching gates.
- To study the Suh et al. (2022) bias–variance trade-off across sharpness $k$ and horizon $H$, compare the analytical smooth gradient $\nabla_\theta \mathcal{L}_{\text{smooth}}(\theta; k)$ against the zero-order randomized-smoothing estimator $\widehat{\nabla}_{\theta,\sigma} \mathcal{L}_{\text{hard}}(\theta) = \frac{1}{N_{\text{mc}}\sigma^2}\sum_{i=1}^{N_{\text{mc}}} (\mathcal{L}_{\text{hard}}(\theta + \epsilon_i) - \mathcal{L}_{\text{hard}}(\theta))\epsilon_i$ with $\epsilon_i \sim \mathcal{N}(0, \sigma^2 I)$, reporting empirical variance and cosine alignment against a high-sample ($N_{\text{mc}} = 2048$) reference as a cutter crosses a stepover edge and internal corner.

### Specify the new mechanism: Disagreement-Aware Continuation

In [`CSGSimulatorDelta`](../simulator/csg_simulator.py), $M$ overlapping smooth cuts with sharpness $\kappa = k / k_{\text{ref}}$ over-erode the stock SDF by up to $\frac{\log M}{\kappa}$ voxels relative to hard `ti.max` carving:

$$0 \le \operatorname{smax}_{\kappa}(a_1, \dots, a_M) - \max_i a_i \le \frac{\log M}{\kappa}.$$

Consequently, fixed linear annealing suffers from two failure modes: (i) soft occupancy thinks narrow pockets/holes are cleared before the hard cutter reaches them, and (ii) late in training, soft gradients push the tool into target surfaces or holder collisions when $\kappa$ and step size are mismatched.

We formulate **Disagreement-Aware Continuation (`DisagreementContinuation`)** with two coupled mechanisms evaluated at periodic hard-carve validation intervals (every $M_{\text{val}}$ iterations, matched to the checkpoint budget of baseline annealing):

1. **Closed-Loop Soft-to-Hard Bias Calibration (`loss_shift` adaptation):** Measure the boundary disagreement between the terminal soft stock $\widetilde{\phi}_T$ and hard stock $\phi_T^{\text{hard}}$ over the active target band $\mathcal{B} = \{x_i : |\phi_{\text{target}}(x_i)| \le 2\text{ vox}\}$:
   $$\widehat{\delta}_{\text{bias}} = \operatorname{median}_{x_i \in \mathcal{B}}\!\left(\phi_T^{\text{hard}}(x_i) - \widetilde{\phi}_T(x_i)\right), \quad \texttt{loss\_shift} \leftarrow (1 - \eta)\,\texttt{loss\_shift} + \eta\,\operatorname{clip}(\widehat{\delta}_{\text{bias}}, 0, \delta_{\max}).$$
   Shifting $\widetilde{\phi}_T$ by $+\texttt{loss\_shift}$ inside [`compute_loss`](../simulator/csg_simulator.py#L1183) aligns the soft zero-level set with the true deployable hard carve without losing smooth spatial gradients.
2. **Disagreement-Gated Sharpness & Trust-Region Step Acceptance:** Track the soft-versus-hard Dice gap $\Delta_{\text{gap}} = \text{Dice}_{\text{soft}} - \text{Dice}_{\text{hard}}$ and hard feasibility (pre-cut holder overlap $= 0$ and hard gouge fraction $\le \tau_{\text{gouge}}$). If an optimization window improves $\mathcal{L}_{\text{smooth}}$ but degrades $\text{Dice}_{\text{hard}}$ by more than $\epsilon_{\text{tol}}$ or induces a hard collision/gouge violation, backtrack $\theta$ to the last accepted hard checkpoint, multiply sharpness $k \leftarrow \min(k_{\max}, \gamma_k k)$ ($\gamma_k > 1$), and decay the trust-region learning rate $\text{lr} \leftarrow \gamma_{\text{lr}}\,\text{lr}$.

## 4. Policy-learning baselines for the OVPR claims (Stage 7)

After the per-instance trajectory-optimization study (Stages 1–6) is complete, evaluate the OVPR amortized policy-learning claim in a dedicated experiment:

- Implement a continuous residual Gymnasium environment (`ResidualCAMEnv`) sharing the exact same control-point/delta action bounds, initial CAM seeds, pre-cut holder safety checks, and reward terms as Stage 3.
- Compare **Residual PPO** and **Residual SAC** (trained on the hard cutting mode via Stable-Baselines3) against **SHAC** and **AHAC** (trained with truncated horizon $H$ gradients through the smooth cutting mode paired with a hard/smooth critic).
- Report training compute (simulation steps and GPU wall time), held-out part generalization (zero-shot policy rollout), and few-step per-instance gradient fine-tuning separately.

## 5. Datasets and task sources

Combine a controlled procedural suite Mapped to the three canonical CAM strategies in the framing memo with external machining geometry and local STEP parts.

### Core procedural benchmark suite (3 Task Families $\times$ 10 Held-Out Geometries)

Following [Differentiable CNC Simulator Paper Framing.md](Differentiable%20CNC%20Simulator%20Paper%20Framing.md#L102-L109), structure the primary benchmark around three canonical 3-axis operations, each with parameterized geometric variations (3 development/validation instances for tuning + 10 held-out test instances per family):

1. **Family 1: Facing on Asymmetric / Stepped Stock (`Facing-Init`):** Non-rectangular (L-shaped, T-shaped, stepped-island, or chamfered-polygon) top stock profile requiring planar or multi-step facing. Initialized with bounding-box **Direction-Parallel Zig-Zag (Raster) Facing** with 90° stepover turns. *Optimization challenge:* bounding-box raster passes waste substantial time air-cutting outside the asymmetric stock footprint and decelerate at sharp reversals.
2. **Family 2: Multi-Feature Pocket & Island Contouring (`Contour-Init`):** 2.5D/3D cavities with sharp internal corners (60°–90°), narrow bottleneck channels, and 1–2 internal islands. Initialized with **Contour-Parallel Offset Loops**. *Optimization challenge:* offset loops suffer engagement/force spikes in internal corners, leave uncut residual cusps in narrow bottlenecks, or execute redundant passes around closely spaced islands.
3. **Family 3: Variable-Width Curved Slot & Sloped-Floor Roughing (`Adaptive-Init`):** Curved channels and deep pockets with variable width and sloped/contoured 3-axis floors. Initialized with **Circular Trochoidal / Adaptive Clearing** loops along the medial axis. *Optimization challenge:* circular trochoids spend ~50% of each loop air-cutting and experience uneven chip load when channel width or floor depth varies, whereas smooth gradients can deform circular loops into compressed B-spline/elliptical loops.

### External datasets and real CAD models

| Resource | Available material | Recommended use and limitations |
|---|---|---|
| **DeepMS-DataGen** | FreeCAD generator producing operation sequences, intermediate STL geometry, voxel grids, and removal labels | Scalable source of multi-feature 3-axis intermediate stock states; generate and validate actual toolpaths under the shared evaluator |
| **MFCAD / MFCAD++ / MFInstSeg** | Machining-feature CAD geometry and face labels | External held-out geometry generalization tests after single-tool accessibility filtering |
| **CNCGEN preview** | 20 synthetic preview examples on HuggingFace (reference paths and final voxels; no full splits) | Unit, stock-bound, and path-adapter validation plus small qualitative comparisons; do not claim full-benchmark reproduction |
| **ABC / CNC-Net subset** | CAD models used by CNC-Net | Secondary comparison subset for uninitialized and single-tool shape carving after 3-axis accessibility filtering |
| **Local Titan M1–M10** | Ten STEP models in [`parts/`](../parts/) | Practical industrial case studies and physical machining targets on reachable single-tool subproblems |

Primary sources:

- [DeepMS-DataGen](https://github.com/jaimemaqueda/DeepMS-DataGen) & [DeepMS](https://github.com/jaimemaqueda/DeepMS)
- [AAGNet resources for MFCAD, MFCAD++, and MFInstSeg](https://github.com/whjdark/AAGNet)
- [CNCGEN public release](https://huggingface.co/datasets/jimmy-zhou/CNCGEN)
- [CNC-Net official repository](https://github.com/myavartanoo/CNC-Net_PyTorch)

For every task (procedural or imported), record a frozen JSON task specification containing: target SDF/mesh, initial stock dimensions, G54 origin, tool radius/height/cut-height, holder radius/height, reachable target mask (excluding inaccessible undercuts or multi-setup features), nominal CAM seed `.npy` and `.nc`, and split membership (`dev` vs. `test`).

## 6. Independent geometry and process validation

Use independent G-code simulators as validation oracles, not optimizer baselines.

1. **High-Resolution Chronological Posted-NC Evaluator (Internal Reference Oracle):** Parse the emitted RS274/Haas `.nc` file via [`cam/gcode_parser.py`](../cam/gcode_parser.py), reconstruct the exact-stop motion profile via [`cam/trajectory_planner.py`](../cam/trajectory_planner.py), and execute the corrected 3D swept-cylinder hard carve at $2\times$ evaluation grid resolution while checking pre-cut holder clearance, rapid-in-material collisions, and stock containment at every sub-step.
2. **CAMotics (Primary External Open-Source G-Code Verifier):** Run headless `camotics` CLI / Python bridge to export the independently simulated cut-workpiece triangle mesh from the posted `.nc` file and compare volumetric IoU, Chamfer Distance, and surface deviation against the target mesh. [Repository](https://github.com/CauldronDevelopmentLLC/CAMotics).
3. **FusionCut / MachineBench (Optional Secondary Evaluators):** Where environment dependencies permit, run the NC-only verification path. [FusionCut repository](https://github.com/bankh/FusionCut).

Validate all evaluators first against **analytic known-volume fixtures**: horizontal slots, pure vertical plunges, 3D sloped ramps, repeated overlapping cuts, narrow pockets, through-holes, and rapid moves whose endpoints are in air but whose interior crosses solid stock.

## 7. Experimental protocol and metrics

### Controls and budgets

Hold constant across all compared solvers on a given task:
- Initial CAM seed (`CAM-Base`), B-spline control-point basis $B$ (dimension $K$), and parameter bounds.
- Stock, target, cutter, holder, setup, and G54 coordinate conventions.
- Feed/rapid speed limits, `dt`, and G-code post-processor.
- Validation checkpoint frequency and hyperparameter-tuning budgets (tuned strictly on `dev` geometries; frozen on `test` geometries across 5 seeds per stochastic method).

Report both **matched simulation-evaluation budget** (counting forward+backward passes as 2 simulation work units and forward-only hard/soft evaluations as 1 unit) and **matched GPU/CPU wall-clock time** (separating Taichi JIT compilation warm-up from steady-state optimization time).

### Outcomes (Co-Primary Geometric and Process Metrics)

Every trajectory is graded on the **posted `.nc` program** under the independent hard evaluator:

1. **Hard Feasibility Gates (Pass/Fail & Violation Magnitude):**
   - **Pre-cut holder collision volume** ($\text{mm}^3$, must be $0$ for feasibility).
   - **Rapid-in-stock collision volume** ($\text{mm}^3$, must be $0$ for feasibility).
   - **Hard overcut / gouge fraction and volume** ($\text{mm}^3$, must satisfy $\text{gouge\_frac} \le \tau_{\text{gouge}}$, e.g., $0.005$).
2. **Primary Geometric Accuracy Metrics:**
   - **Hard Posted-Path Volumetric IoU ($\uparrow$) and Dice ($\uparrow$)** against the reachable target mask.
   - **Residual uncut stock volume ($\downarrow$, $\text{mm}^3$ and fraction)** and **Gouge volume ($\downarrow$, $\text{mm}^3$ and fraction)** with explicit denominators.
   - **Surface Finish Deviation ($\downarrow$, mm):** Average Surface Distance (ASD), 95th-percentile Hausdorff Distance (HD95), and 99th-percentile / maximum surface deviation on target faces.
3. **Co-Primary Process Efficiency & Load Metrics (reported across all runs, and ranked among geometrically feasible, collision-free paths):**
   - **Posted Machining Cycle Time $T_{\text{cycle}}$ ($\downarrow$, seconds)** under controller feed/rapid and acceleration limits.
   - **Air-Cutting Time Fraction ($\downarrow$, %)** of total cycle time.
   - **Peak Volumetric Engagement / Cutting Force Proxy ($\downarrow$, $\text{mm}^3/\text{s}$ and N):** Maximum per-segment material removal rate $\max_t \text{MRR}_t$ and peak force proxy $F_{\text{cut,max}}$.
4. **Diagnostic & Disagreement Metrics:**
   - **Soft-to-Hard Dice Gap** ($|\text{Dice}_{\text{soft}} - \text{Dice}_{\text{hard}}|$) and **Carve-to-Posted Disagreement** ($|\text{Dice}_{\text{hard}} - \text{Dice}_{\text{posted}}|$).

### Core mechanism ablations (Experiment 3)

- **Continuation & De-biasing:** Fixed-$k$ Adam vs. Linear-$k$ Annealing vs. Morphology-Adaptive Schedule vs. **Disagreement-Aware Continuation** (with and without adaptive `loss_shift` and trust-region backtracking).
- **Checkpoint selection:** Soft-loss selection vs. Hard-Dice + feasibility selection.
- **Initialization (Experiment 1):** Straight-line / random init vs. `Facing-Init`, `Contour-Init`, and `Adaptive-Init`.
- **Parameterization & Dimensionality ($K$-sweep):** B-spline control points $K \in \{16, 32, 64, 128\}$ vs. raw per-step deltas ($K=T$) vs. absolute positions (`tool_pos`) under matched executed-path constraints.
- **Smoothing sharpness ($k$) & Horizon truncation ($H$):** Empirical gradient cosine similarity and variance of $\nabla_\theta \mathcal{L}_{\text{smooth}}(\theta; k)$ vs. randomized hard estimator $\widehat{\nabla}_{\theta,\sigma}\mathcal{L}_{\text{hard}}$ across $k \in \{10, 20, 40, 80, 160, 320\}$ and backpropagation horizons $H \in \{4, 8, 16, 32, 64, \text{Full}\}$ using `save_state`/`restore_state`/`forward_from`.
- **Safety chronology:** Holder penalty disabled vs. post-cut holder penalty (legacy bug) vs. pre-cut chronological holder penalty.

## 8. Execution order and decision gates

Note: **Stage 4 (New mechanism)** is intentionally scheduled *before* **Stage 5 (Held-out external evidence)** so that algorithm development and hyperparameter selection occur strictly on the procedural development split without test-set leakage.

| Stage | Deliverable | Decision gate |
|---|---|---|
| **0. Reference code & environment** | Clone/pin external reference repos (`pycma`, `CNC-Net`, `DeepMS-DataGen`, `CNCGEN` preview, `CAMotics`); verify reproducible `uv` environment and CPU/GPU test suite | All reference repos pinned by commit SHA; `pytest` passes 100% |
| **1. Simulator correctness & oracle** | Fix 3D/vertical swept-cylinder SDFs (`tool_sdf*`) and pre-cut holder chronology (`stock[t]`); build known-volume analytic fixtures and posted-NC + CAMotics verification bridge | Analytic volume error $< 1\%$, subdivision invariance holds on 3D plunges/ramps, and pre-cut holder catches adversarial plunge collisions |
| **2. Task suite, CAM seeds & spline basis** | Implement 3 procedural task families (3 dev + 10 test geometries each), FreeCAD/algorithmic seed generators (`Facing`, `Contour`, `Adaptive`), and differentiable B-spline basis $B$ ($K \in \{16..128\}$) | Every task has a verified JSON spec, reachable target mask, nominal `.nc` program, and verified AD gradient through $B\theta$ |
| **3. Controlled 2×2 solver & mechanism study** | Run Exp. 1 (No-Init vs. CAM-Init), Exp. 2 (2×2 Smooth/Hard $\times$ Adam/CMA-ES/CEM across $K$), and Exp. 3 ($k$-bias/variance and horizon $H$ sweeps) on `dev` and procedural `test` splits | Quantify exact regime where analytical derivatives outperform CMA-ES/CEM at matched budget and objective |
| **4. Disagreement-aware continuation** | Implement and tune `DisagreementContinuation` (adaptive `loss_shift` + disagreement-gated $k$/step trust region) on `dev` split; evaluate on procedural `test` split | Statistically significant reduction in soft/hard gap and improvement in hard posted IoU / cycle time over linear annealing at matched validation budget |
| **5. External benchmarks & Titan parts** | Evaluate frozen solvers on `CNC-Net` compatible subset, `DeepMS-DataGen` / `MFCAD++` held-out parts, and reachable single-tool operations on `Titan-M1..M10.step` | Gains persist on external CAD topologies and survive independent CAMotics / posted-NC verification |
| **6. Physical CNC validation** | Execute matched `CAM-Base` vs. `DiffCAM` posted G-code trials in machinable wax and Aluminum 6061-T6; record cycle time, spindle load, and 3D scanned/CMM deviation | Simulation cycle-time / engagement improvements transfer to physical hardware within dimensional tolerances |
| **7. Policy-learning extension** | Build `ResidualCAMEnv` and benchmark Residual PPO/SAC vs. SHAC/AHAC on multi-part training and held-out generalization | Amortized policy inference and few-step fine-tuning trade-offs quantified for the OVPR milestone |
