# T1.0 independent carve and chronology design

Status: fixture design plus implementation reconciliation. Input design commit was `352205bfc05e982d323fab210de8700539f8e544`; T0.0 contracts were acknowledged after their timing-validation update. Executed validation is recorded in `reference_convergence.json` and the T1.0 handoff, not inferred from this design.

## 1. Independence and model boundary

The only production sources consulted for conventions were `simulator/csg_simulator.py` and `tests/core_test.py`. The reference implementation may import the pure T0.0 contract records and NumPy, but must not import `simulator`, `cam`, or transitively reuse production sweep/SDF helpers. An AST import guard will enforce this restriction.

The existing simulator has a **flat capped cutting cylinder**. It has no ball-end cutter. Ball-end cases are therefore **not applicable**, not passed. A discriminated ball tool can be added later with its own independent derivations.

Reference geometry is evaluated in float64 physical millimetres in the **work** frame (`+Z` up). Machine-frame motion is converted through the inverse rigid transform; tilted work/machine Z axes are explicitly unsupported in this reference revision. The tool reference point is cutter bottom-centre. For base position `p=(x,y,z_b)`, the flat cutting solid is

```
C(p) = {(X,Y,Z): (X-x)^2 + (Y-y)^2 <= r^2,
                  z_b <= Z <= z_b+h_c}.
```

T0.0 tooling includes a cutting cylinder, a non-cutting shank from `h_c` to `holder_bottom_mm`, and a coaxial holder from that offset upward. Neither shank nor holder removes material. Geometry-heavy fixtures use a flush shank/holder with cutter radius and `holder_bottom_mm=h_c`, rather than pretending the mandatory assembly is absent. They are not physical safety qualifications. A finite tool or holder boundary has zero volume, but the discrete convention is still fixed:

- removal uses closed-set membership (`distance <= 0`);
- collision requires strict interior overlap (`distance < 0`) and positive sampled volume;
- tangency is reported as contact, not penetration;
- a requested physical clearance margin is a separate millimetre value.

The initial stock is an axis-aligned box unless a fixture explicitly says otherwise. `B=(60,40,12)` below means `[0,60] x [0,40] x [0,12]` in millimetres.

No fixture includes a machine table or workholding beneath the stock. In particular, the through-hole is a geometric statement, not authorization to cut through a real setup.

## 2. Chronology and ownership semantics

These rules need to agree with T0.0 and T1.2/T1.3:

1. The plan's initial pose is state, not an implicit cut. An empty plan removes zero volume even if its initial cutter overlaps stock.
2. Every feed segment includes its start and end occupancy. Zero-length feed contributes one stationary cutting solid and zero duration.
3. A rapid segment never removes material, regardless of whether its cutter crosses stock. Both cutter and holder are checked for rapid collision.
4. Feed samples are processed in chronological order: subtract the current cutter, then check the holder against the remaining stock. This represents cutter-leading-holder motion rather than testing a whole-segment holder union against stale pre-segment stock.
5. Repeating or reversing a feed cannot add already-removed volume. Repetition can still change collision counts and time-integrated quantities.
6. Evaluation diagnostics continue after a modeled collision so that removal, unique contact, and integrated overlap have explicit meanings. A downstream fail-fast executor must label that different coverage policy.
7. An arc is unsupported unless its plane, centre, handedness, endpoints, and timing are fully available. It is never silently replaced by one chord.

Adjacent segments must have coincident endpoints in the same physical frame; discontinuities are either invalid or explicit separate motion, not teleports.

For each linear segment define `p(lambda)=p0+lambda*(p1-p0)`, `0<=lambda<=1`. At sample `j`, feed updates stock with `S <- S \ C(p(lambda_j))` before holder diagnostics. Rapid leaves `S` unchanged. Overlap volume at a sample is measured against that instantaneous remaining stock, not a saved terminal stock union.

Report distinct diagnostics:

- `collision_sample_count`: number of station/segment events with positive penetration volume; duplicate segment-boundary events remain explicit;
- `unique_contacted_volume_mm3`: spatial union of sampled holder/rapid contacts, counted once;
- `time_integrated_overlap_mm3_s`: trapezoidal time integral of instantaneous overlap;
- `first_contact`: segment id and interval in normalized motion and physical time;
- `sampled_min_clearance_mm`: a sampled diagnostic, not an exact continuous-distance claim.

For segment duration `tau`, uniform temporal stations have trapezoid weights `tau/(2m)` at the two endpoints and `tau/m` at interior stations. Zero-duration feed has zero integrated quantity even though its stationary cutting volume is nonzero.

## 3. Dense motion sampling and bounded uncertainty

### 3.1 Stations

At spatial edge `v`, use uniform dyadic stations including both endpoints. For a linear segment of centre displacement `ell3=|p1-p0|`, choose

```
s(v) = min(v/2, r/4, h_c/4)       # millimetres; explicit settings may tighten it
m    = next_power_of_two(max(1, ceil(ell3/s(v)))).
```

The same `m` samples the near-vertical `Z` excursion; no XY-only subdivision is allowed. Zero-length uses `m=1` but only one distinct geometric station. For diagnostics, per-segment station records may retain duplicate endpoints.

With `m` equal intervals, every continuous pose is within

```
delta_t = ell3/(2m)                # millimetres, Hausdorff translation bound
```

of a sampled pose. If `U` is the true swept cutting solid and `U_m` the union at sampled poses, then

```
U_m subset U subset U_m dilated_by_ball(delta_t).
```

Sampling only loses removed material in the continuous model. Dyadic refinement nests stations, so on a fixed voxel lattice removed occupancy and sampled removed volume are monotone nondecreasing.

### 3.2 Spatial/temporal volume bracket

A voxel-centre volume is an estimate, not the exact continuous analytic volume. The reference will additionally compute a certified slow bracket using exact float64 box/cylinder distances. For cubic cells of edge `v`, let `rho=sqrt(3)*v/2` (the centre-to-corner distance).

For each cell centre `x`:

- **definitely removed lower bound:** the whole cell is inside initial stock and inside one sampled feed cutter. A sufficient test is `d_stock(x)<-rho` and `d_Cj(x)<-rho` for some `j`;
- **possibly removed upper bound:** the cell can intersect initial stock and the true feed sweep. Include it if `d_stock(x)<=rho` and, for some segment/sample, `d_Cj(x)<=rho+delta_t(segment)`.

Multiply the two cell counts by `v^3` to obtain `L <= V_continuous <= U`. The same bounds enclose the station-sampled voxel-centre estimate. Rapids are excluded from both. Repeated cuts accumulate set unions, not volumes per segment.

Refine `v <- v/2` and at least double temporal `m`. For unchanged geometry, full/uncertain cell nesting makes `L` nondecreasing and `U` nonincreasing up to float64 roundoff. Record:

- voxel-centre estimate and removed-cell count;
- `[L,U]`, `U-L` in mm^3, `rho` and each segment's `delta_t` in mm;
- successive estimate changes and bracket widths;
- any case where refinement budget is exhausted before its required absolute uncertainty budget.

An analytic fixture passes containment only when its independently derived `V*` lies in `[L,U]` (apart from a declared float64 numerical epsilon). The saved convergence report records absolute widths and monotonic refinement rather than retro-fitting a numeric acceptance budget to observed output. Any downstream evaluator requiring a tighter absolute mm^3 budget must declare it before comparing results. There is no universal percentage tolerance. Comparison roundoff may use a tiny scale-derived numerical epsilon based on millimetre lengths and voxel counts; geometric acceptance comes from the physical bracket.

### 3.3 Chronological safety is tri-state

A sampled no-contact result alone does not prove continuous safety. Within each temporal subinterval:

- to rule out collision, use stock immediately after the station's ordered feed cut, inflate moving assembly positions by `delta_t`, and account for the spatial cell bound; stock only shrinks thereafter;
- if this conservative possible-contact check is clear, that interval is safe under the model;
- if robust post-cut penetration exists at a station, report collision;
- if possible contact exists but robust penetration is not established, report **unknown**, not safe and not a fabricated collision.

This is a time-refined conservative screen. A cutter-leading-holder fixture can be flagged at coarse `m` and become safe when the cutter's axial lead exceeds temporal and spatial bounds. Genuine holder/rapid collisions must become robust failures under refinement; expected-safe fixtures must become safe, not merely asymptotically zero volume.

First-contact time is an interval between the last clear and first robust-contact stations. Unique volume and integrated overlap retain their spatial/temporal uncertainty. Clearance uses absolute millimetres; volumes use absolute mm^3 derived from the active grid bracket, and integrated quantities use mm^3*s.

## 4. Independent volume derivation for monotone-Z linear motion

The following derivation is independent of the production XY-closest-point sweep. For one linear segment with flat cutter radius `r`, height `h_c`, base heights `z0,z1`, and total horizontal displacement `ell=sqrt(dx^2+dy^2)`, define

```
I(z) = {lambda in [0,1]: b(lambda) <= z <= b(lambda)+h_c}
q(z) = length(I(z)),
```

where `b(lambda)=z0+lambda*(z1-z0)`. For monotone `b`, `I(z)` is one interval. At height `z`, the sweep cross-section is a line of disk centres of length `ell*q(z)` with radius-`r` caps. Its area is `pi*r^2 + 2*r*ell*q(z)` when it is wholly inside stock XY.

If stock covers those cross-sections in XY, its vertical intersection therefore has exact volume

```
V = pi*r^2*W + 2*r*ell*Q,
W = measure{z in stock Z: I(z) is nonempty},
Q = integral_over_stock_Z q(z) dz.
```

This formula explicitly clips to stock and includes both endpoint cutter occupancies. For horizontal motion, `q=1` wherever the fixed cutter covers stock Z. For zero horizontal displacement the second term vanishes. The ramp and near-vertical fixtures below use piecewise-linear `q`, not numerical values inferred from the production simulator.

For a stock cross-section smaller than the cutter disk, replace `pi*r^2` by the independently computed stock/sweep cross-section area. For reversal and repeated paths, take the geometric union before integrating; do not add segment volumes.

## 5. Fixture catalogue

All positions are `(x,y,z_base)` in the work frame, in mm. Unless stated otherwise, stock is `B`, the flush assembly uses cutter radius with `holder_bottom_mm=h_c`, the motion is feed, and `V*` is exact **continuous** removed volume, not an expected coarse voxel count. Exact expressions are authoritative; decimals are review aids.

### Pure volume and motion cases

| ID | Geometry and ordered motion | Independent derivation and expected result |
|---|---|---|
| `empty_preengaged` | `r=3,h_c=6`, initial `(20,20,4)`, no segments | `V*=0`: the engaged initial pose alone is not a cut. |
| `plunge_above` | `r=3,h_c=20`, `(20,20,18) -> (20,20,8)` | Entire disk is inside stock XY. Cutter union starts above stock and reaches 4 mm into its 12 mm height: `V*=9*pi*4 = 36*pi = 113.09733552923255 mm^3`. Using the full `D+h_c` tool extent would incorrectly count air. |
| `plunge_preengaged` | Same tool, `(20,20,10) -> (20,20,6)` | Start already cuts the top 2 mm and descends another 4 mm; union intersects stock over `z=6..12`: `V*=9*pi*(2+4) = 54*pi = 169.64600329384882 mm^3`, not `36*pi`. |
| `horizontal_slot` | `r=3,h_c=20`, `(14,20,0) -> (38,20,0)` | Cutter covers all stock Z. Exact 2-D capsule area is rectangular middle `2*r*24=144` plus two end disks totalling `9*pi`. `V*=12*(144+9*pi) = 2067.2920065876975 mm^3`. The starting/end disks must not be omitted as "initial placement." |
| `sloped_ramp` | `r=2,h_c=12`, `(8,20,8) -> (32,20,-4)` | `ell=24`, downward `D=12`; union spans stock Z, so `W=12`. Here `I(z)=[(8-z)/12,(20-z)/12] intersect [0,1]`. Thus `q(z)=(z+4)/12` for `0<=z<=8`, and `q(z)=(20-z)/12` for `8<=z<=12`. `Q=16/3+10/3=26/3` mm. `V*=4*pi*12 + 2*2*24*(26/3) = 48*pi+832 = 982.7964473723101 mm^3`. |
| `near_vertical` | `r=3,h_c=10`, `(20,20,30) -> (20.8,20.3,-10)` | `D=40`, `ell=sqrt(0.73)`, `W=12`. For every `0<=z<=12`, `I(z)=[(30-z)/40,(40-z)/40]`, so `q=1/4` and `Q=3`. `V*=108*pi + 2*3*sqrt(0.73)*3 = 108*pi+18*sqrt(0.73) = 354.6712133292692 mm^3`. A nonzero horizontal component is intentional. |
| `reversal` | `r=3,h_c=20`, `(12,20,0) -> (40,20,0) -> (26,20,0)` | The half-length return lies inside the first capsule and its dyadic stations are a subset of the outward stations. Union remains one length-28 capsule: `V*=12*(2*3*28+9*pi) = 12*(168+9*pi) = 2355.2920065876977 mm^3`. It equals single-segment `12 -> 40`, not the sum of two cuts. Arbitrary segment partition/phasing need not give identical discrete occupancy until refined. |
| `repeated_plunge` | Same tool as `plunge_above`, feed `18 -> 8 -> 18 -> 8` at `x=y=20` | All three sweeps have the same stock intersection: `V*=36*pi = 113.09733552923255 mm^3`, exactly the one-plunge result. Segment work/count diagnostics may grow; unique removed volume may not. |
| `through_hole` | `r=4,h_c=6`, `(20,20,16) -> (20,20,-2)` | Swept vertical range is `-2..22`, clipping to stock height 12. `V*=16*pi*12 = 192*pi = 603.1857894892403 mm^3`; the cylindrical hole has no remaining floor in this geometric model. |
| `zero_length_feed` | `r=3,h_c=6`, one feed `(20,20,4) -> (20,20,4)`, duration 0 | One stationary solid intersects stock over `z=4..10`: `V*=9*pi*6 = 54*pi = 169.64600329384882 mm^3`. No division by segment length; integrated overlap is zero. |

The `horizontal_slot` also tests initial in-stock placement: omitting either endpoint disk gives the wrong answer. `plunge_above`, `plunge_preengaged`, and `through_hole` distinguish displacement, initial overlap, and stock clipping.

Ball-end plunge: **N/A for current production geometry**. It must not be reported as a passing fixture.

### Ordered holder and rapid cases

| ID | Geometry and ordered motion | Expected geometry and chronology |
|---|---|---|
| `cutter_leading_holder_safe` | Narrow stock `[17.5,22.5]^2 x [0,12]`; `r=6,h_c=4`, holder `R=10,H=20`; feed `(20,20,14) -> (20,20,5)` | The square corner radius is `2.5*sqrt(2)`, leaving radial cutter coverage margin `6-2.5*sqrt(2)=2.464466094...` mm. Continuous removal is stock area `25` times depth `7`: **`V*=175 mm^3`**. After each instantaneous cut, remaining stock in the holder's radial column lies below the cutter base while the holder begins 4 mm above it; expected chronological verdict is **safe**. A whole-segment check against pre-segment stock sees holder disk height `z=9..12`, an overlap up to `25*3=75 mm^3`, and is a documented conservative false positive—not the reference verdict. |
| `genuine_holder_collision` | Full stock `B`; same cutter/holder and motion as above | Continuous cutter removal is **`V*=36*pi*7 = 252*pi = 791.6813487046279 mm^3`**. Holder bottom is `18-9*lambda`; its top-stock contact is at `lambda=2/3`, with penetration for `lambda>2/3`. The cutter clears only radius 6, leaving an annulus `6<rho<=10`. At completion, annular residual contact spans `z=9..12`: **`unique_contacted_volume*=pi*(10^2-6^2)*3 = 192*pi = 603.1857894892403 mm^3`** and safety must **fail**. For 1 s duration, `V_overlap(lambda)=576*pi*max(0,lambda-2/3)`, so **`time_integrated_overlap*=32*pi = 100.53096491487338 mm^3*s`**. With `m=3` uniform stations only `lambda=1` has positive penetration; with `m=6`, exactly two stations do, illustrating why count is not unique volume or time integral. |
| `rapid_through_solid` | Stock `B`, `r=3,h_c=4`, flush radius-3 holder from tool-relative height 4; rapid `(-10,20,6) -> (70,20,6)`, duration 8 s | **`V_removed*=0`**, and continuous final stock equals initial stock despite crossing it. First cutter contact is at centre x=-3, `lambda=7/80=0.0875`, penetration immediately after; rapid safety must **fail**. Continuous unique cutter-only contact is `60*6*4 = 1440 mm^3`, retained as a separate component diagnostic. The mandatory assembly also intersects stock from z=10..12, so total unique assembly contact is **`60*6*6 = 2160 mm^3`**. For a traversal with vertical overlap `a`, the integrated overlap formula is `(duration/80)*a*pi*r^2*60`; cutter-only is `216*pi`, while the full six-mm assembly is **`324*pi = 1017.876019763093 mm^3*s`**. These nonzero collision quantities must never become carved volume. |

The safe and unsafe holder cases share exactly the same tool and motion and differ only in stock width. This isolates chronology from a trivial "wide holder is always bad" rule.

A `zero_length_rapid` variant places the rapid cutter strictly inside full stock: removal is zero and the initial rapid station must fail. These are software geometry cases only, not machining recommendations.

## 6. Radius and grid-phase variants

The curved boundary must not be tested only at convenient cell centres.

1. For `plunge_above`, `horizontal_slot`, `near_vertical`, and `zero_length_feed`, use `r={2.35,3.0,3.65}`; for the ramp use `{1.35,2.0,2.65}`; for the through-hole use `{3.35,4.0,4.65}`. Substitute the radius in the closed-form expression rather than fitting expected values from output.
2. Add an XY fractional-voxel offset `(0.13*v,0.29*v)` to interior tool paths while retaining sufficient stock margins. For full-stock capsule cases this changes lattice phase without changing `V*`.
3. For full assembly phase tests, translate stock, all motion endpoints, and tool/holder setup together by `(0.13*v,0.29*v,0.37*v)` relative to a fixed machine-anchored sampling lattice, with grid bounds padded to contain both variants. Every continuous `V*` and collision expectation above is invariant. Do **not** assert identical voxel counts across phases.
4. Run all analytic families at a base grid and selected plunge/ramp/near-vertical/holder cases at two dyadic spatial and temporal refinements. Use pairwise radius/phase coverage rather than an uncontrolled Cartesian sweep.

For the narrow-stock safe case, phase translation moves stock and tool together; translating only the tool would destroy the intended radial stock containment.

## 7. Test and handoff plan

`fixtures.py` contains declarative physical inputs, exact expected expressions/constants, and derivations. Expected numbers are not produced by calling the reference sweep itself. `reference_carve.py` provides the slow chronological reference and conservative bounds. Tests verify:

- every tabled analytic value against its independent formula and the `[L,U]` bracket;
- monotone sampled removal under dyadic temporal refinement on a fixed grid;
- nondecreasing `L`, nonincreasing `U`, shrinking uncertainty under spatial/temporal refinement, within declared float64 epsilon;
- exact one-pass/repeated/reversed occupancy equality on identical grids;
- empty-plan no-cut, zero-length feed, and zero-length rapid without NaNs;
- safe coarse-screen false positive followed by fine time-resolved safety;
- robust genuine-holder and rapid failures, including unique volume versus count versus integrated volume expectations;
- unchanged continuous expectations across radius/phase variants, with separately reported voxel discretization bias;
- static independence from production geometry imports.

The implementation acceptance command is `uv run python -m pytest tests/reference_geometry_test.py -q`; `DIFFCNC_REFERENCE_CONVERGENCE=1` enables the explicit refinement test. `verification/reference_convergence.json` is generated separately by `uv run python -m verification.fixtures --output verification/reference_convergence.json` and records task/motion hashes, exact expectations, grid/temporal settings, brackets, collision states, failures, and unknown coverage. On the tested narrow-stock grid, two poses cause undercutting and a coarse false collision, two to three poses leave the conservative bound unknown, and four or more establish the expected safe chronology. Contact-volume uncertainty is reported through sampled refinement diagnostics, not falsely labelled a certified continuous contact bracket.

Downstream consumers need both the continuous expectation and the discretization record. Passing only a scalar volume or only an occupancy mask would hide either voxel bias or temporal under-sampling.

## 8. T0.0 consumer requirements

The contract should expose: physical stock/grid bounds and lattice phase; cutter bottom-centre/tool and holder geometry; optional ball discriminant; work-to-machine rigid transform; initial tool/machine state; stable ordered segment ids, rapid/feed/line/arc data, frames, feed/duration and continuity; closed removal versus strict collision policy; and explicit coverage status/reasons.

`ReferenceSettings` (spatial edge, temporal displacement limit, refinement cap, collision continuation policy) belongs outside task/tool/setup hashes but inside evaluator discretization metadata. Results need separate removed volume, occupancy/grid metadata, unique contact, count, integrated overlap, first-contact location, spatial/temporal uncertainties, and `safe/collision/unknown`; numeric zero must differ from unknown.

These requirements were sent to the coordinator during the read-only phase. Any contract disagreement discovered in consumer review will be recorded as a requested change rather than silently worked around in fixture semantics.
