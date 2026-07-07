# Differentiable Tool-Break Probability

This document specifies a **differentiable estimator of the probability that the cutter breaks** during a trajectory, suitable for inclusion in the GradMill soft objective alongside `w_residual`, `w_air`, `w_prox`, and `w_traj_prox`. The goal is a single scalar $P_{\text{break}}\in[0,1]$ per trajectory whose gradient w.r.t. the tool path (`tool_delta`) is well-defined everywhere, so Adam can trade breakage risk against material removal and air-cut cost.

The analysis is built on primitives the simulator already exposes: the SDF occupancy field `stock_occ = sigmoid(stock)`, the swept-tool occupancy `tool_occ = sigmoid(-tool_sdf)`, and the `smooth_max` / `LSE` soft primitives used throughout `apply_cut`. It is the differentiable counterpart of the **hard** breakage rule in `design.md` ($R_{\text{breakage}}=-100$ iff $F_{\text{cut}}>F_{\max}$), and follows the same soft/hard split documented in `gradients.md`: a soft, smooth surrogate for optimization; the hard threshold remains for eval/RL reward.

---

## 1. Why a *probability*, and why it must be differentiable

The existing hard rule is a step function:

$$R_{\text{breakage}} = \begin{cases} -100 & F_{\text{cut}} > F_{\max} \\ 0 & \text{otherwise} \end{cases}$$

This has the same pathology as `forward_hard` (§1 of `gradients.md`): the derivative is zero almost everywhere and undefined at the threshold. A trajectory that nominally sits just under $F_{\max}$ receives **no gradient signal** to back off, even though it is one transient away from snapping. We need a smooth risk surface that:

1. Is ~0 for comfortably sub-threshold loads,
2. Rises sharply as load approaches $F_{\max}$,
3. Saturates near 1 for severe overload,
4. Has finite, non-vanishing gradient everywhere w.r.t. the path.

A **probability** (rather than a deterministic flag) is the right object because breakage is genuinely stochastic: carbide is brittle, strength is flaw-controlled (Weibull), and the dynamic load amplification from entry shock / chip clogging / chatter is uncertain. Modeling the load margin as a random variable and asking $P(\text{load} > \text{strength})$ gives exactly the smooth, saturating surface above.

---

## 2. Cutting-load model from SDF occupancy

Per cutting step $t$, the swept-tool volume in engagement with remaining stock is already computed by the existing `w_air` / `w_prox` kernels as the overlap of `tool_occ` and `stock_occ`. We define the **engaged chip volume**:

$$V_{\text{chip}}(t) = \sum_{p\in\text{grid}} \text{tool\_occ}(p,t)\;\cdot\;\text{stock\_occ}(p,t) \;\cdot\; v_{\text{vox}}$$

where $v_{\text{vox}}$ is the physical voxel volume (mm³) and `stock_occ` is evaluated on the *pre-cut* stock `stock[t]` (the material the tooth actually meets), consistent with the air-cut loop reading `stock[t+1]` semantics in `apply_cut`. This sum is a soft, differentiable analogue of the design.md force law $F_{\text{cut}}\propto\sum_{p}\max(0,-\phi_{\text{stock}}(p))$ — replacing the hard `max(0,-φ)` with the smooth occupancy product.

The **nominal tangential cutting force** is then:

$$\mu_F(t) = k_c \cdot \frac{V_{\text{chip}}(t)}{\Delta t \cdot D}$$

with $k_c$ the specific cutting force of the workpiece (Al ≈ 600–800 N/mm²), $D$ the tool diameter, and $\Delta t$ the per-step time. The $V_{\text{chip}}/(\Delta t\cdot D)$ form recovers the standard $F_t \approx k_c\,a_p\,f_z$ relation when the swept geometry resolves to a rectangular chip of depth $a_p$ and feed $f_z$. $\mu_F(t)$ is fully differentiable in `tool_delta` through `tool_occ`.

### 2.1 Dynamic amplification — the dominant uncertainty

Steady-state $\mu_F$ does not snap tools; **transients** do. We model the *peak* load seen by a tooth as the nominal plus a random amplification $\alpha_t$:

$$F_t = \alpha_t \cdot \mu_F(t), \qquad \alpha_t \ge 1$$

$\alpha_t$ aggregates entry shock, plowing at low chip load, chip recutting/clogging, runout, and regenerative chatter. We model it log-normally:

$$\log \alpha_t \sim \mathcal{N}(\mu_\alpha,\;\sigma_\alpha^2)$$

with $\mu_\alpha$ capturing the mean overload factor (process-dependent; 1.5–3× for aggressive aluminum cuts) and $\sigma_\alpha$ the run-to-run scatter. Two regime switches make $\alpha_t$ depend on the path itself (so risk responds to optimization):

- **Low chip load → plowing:** when $V_{\text{chip}}/D \to 0$ the edge rubs rather than shears; rub/scrape drives $\mu_\alpha$ up. We set $\mu_\alpha(t) = \mu_{\alpha,0} + c_{\text{plow}}\,\exp(-V_{\text{chip}}(t)/V_0)$.
- **High engagement → clog/chatter:** large $V_{\text{chip}}$ raises the chatter margin; $\mu_\alpha(t) = \mu_{\alpha,0} + c_{\text{chatter}}\,\sigma_\alpha\,\text{softplus}(V_{\text{chip}}(t)-V_{\text{clog}})$.

These keep $\alpha_t$ a smooth, path-dependent function of $V_{\text{chip}}$, so breakage risk flows gradient into both *how much* and *how aggressively* the tool engages.

---

## 3. Tool strength as a random variable

Carbide is brittle and fails at the weakest flaw in the stressed volume — a textbook **Weibull** weakest-link model. The transverse/bending strength $S$ of a tooth under the engaged volume $V_{\text{chip}}$ has survival:

$$P(S > s) = \exp\!\left[-\frac{V_{\text{chip}}}{V_0}\left(\frac{s}{\sigma_0}\right)^m\right]$$

with Weibull modulus $m$ (carbide $m\approx 5\text{–}10$), characteristic strength $\sigma_0$ at reference volume $V_0$. Two consequences, both physically right and both fed by the differentiable $V_{\text{chip}}$:

1. **Volume effect:** larger engagement → more material under stress → higher chance of a critical flaw. The mean strength *drops* with $V_{\text{chip}}$ as $\bar S \propto V_{\text{chip}}^{-1/m}$.
2. The strength CDF at the operating volume is
$$F_S(s) = 1 - \exp\!\left[-\frac{V_{\text{chip}}}{V_0}\left(\frac{s}{\sigma_0}\right)^m\right].$$

For a tool, the relevant stress is the bending stress at the root of the engaged flute, $\sigma = F_t \cdot L_{\text{eff}} / Z$, with $L_{\text{eff}}$ the effective stickout (lever arm) and $Z$ the section modulus ($\propto D^3$). The deflection/stickout dependence ($\propto L^3$) enters here: longer tools raise both the bending stress and the chatter vulnerability.

---

## 4. Breakage probability: stress–strength interference

Breakage occurs when the peak load exceeds the tooth strength: $F_t > S$. With $F_t = \alpha_t\mu_F$ and the strength distribution above, the per-step breakage probability is:

$$P_{\text{break}}(t) = P\!\left(\alpha_t\,\mu_F(t) > S\right) = \int_0^\infty f_{\alpha\mu_F}(x)\,F_S(x)\,dx$$

where $f_{\alpha\mu_F}$ is the PDF of the (random) peak load and $F_S$ the strength CDF. This is the **stress–strength interference** integral.

### 4.1 Closed-form soft surrogate (the differentiable estimator)

To stay differentiable and cheap (no numerical integration in the Taichi kernel), we use the standard log-normal-vs-log-normal approximation. Assume the peak load $X=\alpha_t\mu_F$ is log-normal and the strength $S$ is *also* approximated log-normal (the Weibull tail is well approximated by a matched log-normal in the upper-tail region that matters for breakage). Then the interference has a closed form in log-space:

$$P_{\text{break}}(t) \;\approx\; \Phi\!\left(\frac{\mathbb{E}[\ln X] - \mathbb{E}[\ln S]}{\sqrt{\sigma_{\ln X}^2 + \sigma_{\ln S}^2}}\right)$$

Substituting the model:

$$\boxed{\;P_{\text{break}}(t) \;\approx\; \sigma_{\text{risk}}\!\left(\frac{\ln\mu_F(t) + \mu_\alpha(t) - \ln\bar S(V_{\text{chip}}(t))}{\sqrt{\sigma_\alpha^2 + (\pi^2/6m^2)}}\right)\;}$$

where:
- $\bar S(V) = \sigma_0 (V_0/V)^{1/m}$ is the mean strength at engaged volume $V$,
- $\sigma_{\ln S}^2 = \pi^2/(6m^2)$ is the log-variance of the Weibull (from $\zeta(2)=\pi^2/6$),
- $\sigma_{\text{risk}}(\cdot)$ is a **soft sign** implemented as `sigmoid` — the same smooth saturator already used throughout the codebase for `stock_occ`/`target_occ`. This keeps $P_{\text{break}}\in(0,1)$ with everywhere-finite gradient.

This is the core estimator. It is a **smoothed step** in the load margin: when $\mu_F \ll \bar S$ the argument is large-negative and $P_{\text{break}}\to 0$; when $\mu_F \gg \bar S$ it saturates to 1; the transition band is set by the combined log-variance. Crucially it is **differentiable in `tool_delta`** through $\mu_F$ and $V_{\text{chip}}$, with the sigmoid providing non-zero gradient even when nominally sub-threshold — exactly the gradient signal the hard rule lacks.

### 4.2 Trajectory-level aggregation

A tool breaks if **any** step exceeds its strength (series system). Assuming approximate independence across steps (justified while transients are modeled as independent draws of $\alpha_t$):

$$P_{\text{break}}^{\text{traj}} = 1 - \prod_{t=0}^{T-1}\bigl(1 - P_{\text{break}}(t)\bigr)$$

For a cheap, well-conditioned surrogate we use the log/soft-plus form (numerically stable, differentiable):

$$P_{\text{break}}^{\text{traj}} \;\approx\; 1 - \exp\!\left(-\sum_t P_{\text{break}}(t)\right) \;\le\; \min\!\left(1,\,\textstyle\sum_t P_{\text{break}}(t)\right)$$

For gradient purposes we can also use the **soft maximum** (LSE) over steps, reusing the existing `smooth_max` primitive, to focus the risk on the worst step:

$$P_{\text{break}}^{\text{traj}} \;\approx\; \text{smooth\_max}_t\bigl(P_{\text{break}}(t);\, k_{\text{agg}}\bigr)$$

The product form is the physically correct series model; the LSE form is cheaper and makes the gradient concentrate on the single most dangerous step (often the entry or a direction-reversal transient). Both are differentiable; the product form is recommended for the soft objective.

---

## 5. Loss term and gradient flow

Add breakage risk to the objective as a one-sided barrier (like `w_air`/`w_prox`):

$$\mathcal{L}_{\text{break}} = w_{\text{break}} \cdot P_{\text{break}}^{\text{traj}}$$

with $w_{\text{break}}$ a weight tuned against `w_residual`. Because $P_{\text{break}}^{\text{traj}}$ is built from `sigmoid`/`smooth_max`/soft `tool_occ`/`stock_occ` products, its gradient flows through the same Taichi autodiff tape as the existing losses:

$$\nabla_{\text{tool\_delta}} \mathcal{L}_{\text{break}} = w_{\text{break}} \cdot \frac{\partial P_{\text{break}}^{\text{traj}}}{\partial \mu_F}\,\frac{\partial \mu_F}{\partial V_{\text{chip}}}\,\frac{\partial V_{\text{chip}}}{\partial \text{tool\_occ}}\,\frac{\partial \text{tool\_occ}}{\partial \text{tool\_delta}}$$

Every factor is smooth and finite; there is **no vanishing-gradient barrier** at the threshold (unlike the hard rule). The gradient pushes the path to (a) reduce engaged volume at risky steps, (b) avoid the low-chip-load plowing regime (via the $\mu_\alpha$ plow term), and (c) avoid the high-engagement clog regime — i.e. it discovers the same adaptive/light-radial roughing behavior that a machinist would choose to protect the tool.

### 5.1 Soft/hard split

Following `gradients.md`:
- **Optimization (`forward`/`apply_cut`):** use the soft $P_{\text{break}}^{\text{traj}}$ surrogate above so Adam gets continuous risk gradient.
- **Eval / RL reward:** keep the hard rule from `design.md` ($-100$ if the *hard*-carved $F_{\text{cut}}$ exceeds $F_{\max}$), computed outside the Tape. The soft surrogate is a training-time stand-in for the hard eval criterion; the `loss_shift` de-biasing mechanism already in the codebase is the template for keeping the soft surrogate honest w.r.t. the hard carve.

---

## 6. Parameter summary

| Symbol | Meaning | Typical / source |
|---|---|---|
| $k_c$ | specific cutting force (Al) | 600–800 N/mm² |
| $D$ | tool diameter | tool config |
| $m$ | Weibull modulus (carbide) | 5–10 |
| $\sigma_0, V_0$ | characteristic strength / ref. volume | tool datasheet / calibration |
| $\mu_{\alpha,0}$ | mean dynamic overload | 1.5–3 |
| $\sigma_\alpha$ | overload scatter | 0.2–0.5 (log-space) |
| $c_{\text{plow}}, V_0$ | plowing-regime parameters | fit to entry-shock data |
| $c_{\text{chatter}}, V_{\text{clog}}$ | clog/chatter-regime parameters | fit to chatter-margin data |
| $L_{\text{eff}}$ | effective stickout (lever arm) | holder + gauge length |
| $k_{\text{agg}}$ | LSE aggregation sharpness | reused from `smooth_max` `kv` |
| $w_{\text{break}}$ | loss weight | tuned vs. `w_residual` |

---

## 7. Calibration & caveats

- **Relative, not absolute.** Without real breakage logs, $\mu_\alpha,\sigma_\alpha,m$ are weakly known. Treat $P_{\text{break}}^{\text{traj}}$ as a **relative risk comparator** across path configurations (A riskier than B), not an absolute failure rate. Calibrate $\mu_\alpha,\sigma_\alpha$ from a small set of known-good/known-bad cuts, or from spindle-current peaks if available.
- **Transient model is the weak link.** The log-normal $\alpha_t$ with path-dependent mean is a coarse aggregator for entry shock and chatter; if chatter margin is the real failure mode, couple this term to the existing speed-limit / trajectory-proximity fields rather than to $V_{\text{chip}}$ alone.
- **Volume effect is real but second-order.** The $V_{\text{chip}}^{-1/m}$ strength scaling matters most when comparing very different engagements (slotting vs. light roughing); for small path perturbations it is nearly constant and the dominant gradient comes from $\mu_F$.
- **Time window.** $P_{\text{break}}^{\text{traj}}$ is per-trajectory. A per-job or per-life probability requires a cumulative-time Weibull lifetime term ($1-\exp(-(t/\eta)^\beta)$, §2 of the companion note) layered on top; out of scope for the per-trajectory soft objective but straightforward to add if a tool-life constraint is wanted.
- **Stickout/deflection.** $L_{\text{eff}}$ enters the strength side via bending stress; for long L:D tools this dominates and the model correctly raises risk — but the lever-arm should come from the actual toolholder+gauge geometry, and the existing toolholder-collision barrier (`enforce_z_floor` / holder penetration) should be kept as a hard constraint regardless of the soft risk term.

---

## 8. Relationship to existing code

The estimator is designed to drop into the existing loss structure with minimal new infrastructure:

- **Engaged volume** $V_{\text{chip}}(t)$: reuses the `tool_occ · stock_occ` overlap already computed by the `w_air`/`w_prox` kernels (`simulator/csg_simulator.py`), evaluated on `stock[t]`. No new SDF evaluation needed.
- **Soft primitives:** `sigmoid`, `smooth_max` (LSE), and the occupancy fields are exactly those used in `apply_cut` and the loss kernels (§2 of `gradients.md`), so the autodiff tape covers the new term for free.
- **Loss bookkeeping:** add `w_break`, `diag_break` fields alongside `w_air`/`diag_air` for observability, and expose the per-step $P_{\text{break}}(t)$ in the diagnostics so the optimizer's risk behavior is visible.
- **Hard eval:** unchanged — the `design.md` $-100$ hard breakage rule remains the eval/RL criterion; the soft surrogate only steers optimization toward paths that satisfy it.
