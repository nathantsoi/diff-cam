# f64 finite-difference input-cast control

## Question and result

After `simulator_array` was changed to preserve full float64 optimizer/tool-motion inputs, the full default suite had one failure in `tests/precision_test.py::test_full_f64_matches_shared_prefix_index_two_direction`:

- its old finite-difference reference no longer matched;
- the separate full-f64 AD/FD agreement check still passed;
- the f32-prefix test was unchanged and passed.

The isolated control below establishes that the difference is the removal of frontend float32 narrowing, not a weakened comparison bound. It is **not** a new production-f64 validation run: it deliberately restores the old narrowing only in one independent diagnostic process.

## Controlled configuration

- Controller commit: **f6569fc**.
- Entry point: `tests/precision_test.py::_reference_prefix_child`, using the owned physical 48-step prefix of the 75-step pilot path, direction index 2.
- Runtime: `Arch.x64`, CPU. `precision_metadata()` reported requested, declared and runtime `default_fp = f64`; float fields, kernels and adjoints use the declared precision.
- Source geometry arrays retain their documented `np.float32` representation and immutable hashes. They are not silently redefined as a different source.
- In this diagnostic child only, `algorithms.objectives.simulator_array` was temporarily replaced with `np.asarray(values, dtype=np.float32)`, the old frontend behavior. The original function was restored in `finally`. The simulator fields and runtime remained f64. No controller file or historical primary/fine-evidence bytes were changed.
- Exceptional-validation distinction: the ordinary bit-exact f64 `tool_delta` test is a real small-grid candidate test. This control specifically does **not** qualify as preserved-float64-input production evidence.

Actual control invocation, whose output initially named a fresh directory and then ran:

```sh
TI_OFFLINE_CACHE_FILE_PATH=/tmp/diffcam-ti-worker-2 \
UV_CACHE_DIR=/tmp/diffcam-uv-worker-2 \
DIFFCAM_ADFD_SMOOTHMAX=canonical DIFFCAM_ADFD_AXIS=2 \
uv run python -I /tmp/diffcam-t41b-check-fd-cast-control.py \
  --repository /home/ntsoi/src/cnc/diffcam-wt/T4.1b-wiring \
  --output /tmp/diffcam-t41b-fd-cast-control.QNU6Ln/prefix-48-f64-legacy-cast
```

The control process exited 0. Its wrapper/output log is `/tmp/claude-1000/-home-ntsoi-src-cnc/71a073bc-1a3f-4b27-91d8-cf973ed89fc4/tasks/bukjesyn0.output`.

## Values at epsilon = 0.00625

| Input path | AD gradient at the nominal point | Central finite difference |
|---|---:|---:|
| Legacy reference before the dtype fix | — | −0.34076627325215547 |
| Isolated control with old float32 narrowing restored | −0.3407647056710862 | −0.34076627324900244 |
| Preserved float64 theta/normalized tool deltas | −0.34076472292434656 | −0.3407654171104779 |

The restored control differs from the historical reference by only about **3.2e-12**, comfortably reproducing that reference. Preserving f64 input shifts the expected central FD because it changes the actual normalized theta values previously lost to float32 rounding; it is not a reason to restore narrowing or to select a value that happens to agree.

For the preserved-f64 case, `abs(AD − central FD)` is approximately **6.94e-7**, below the declared, unchanged **2e-6** bound for this physical-prefix protocol. This is not an assertion of global finite-difference accuracy, an error bound for other coordinates or a machine-safety statement.

## Retained source evidence

- Isolated control: `/tmp/diffcam-t41b-fd-cast-control.QNU6Ln/prefix-48-f64-legacy-cast/control.json`, which explicitly records `is_current_production_f64_input: false`, plus the adjacent `prefix.json` and `control.log`.
- Preserved-f64 prefix from the original full-suite run: `/tmp/diffcam-t41b-default-f6569fc.fncF9s/pytest-temp/test_full_f64_matches_shared_p0/prefix-48-f64-native/prefix.json`.
- Original failing-suite log (one failure, 616 passed, 11 skipped): `/tmp/diffcam-t41b-default-f6569fc.fncF9s/pytest.log`.
- Corrected f64 prefix-test rerun: `/tmp/diffcam-t41b-f64-prefix-rerun.g9Jg73/pytest.log` (**1 passed**, 42.87 s). The subsequent ca3976c full default rerun is `/tmp/diffcam-t41b-default-ca3976c.uxQoF9/pytest.log` (**617 passed, 11 skipped**, 546.80 s).
- Control script: `/tmp/diffcam-t41b-check-fd-cast-control.py`, intentionally outside the controller repository.

The coordinator authorized a narrow correction of only the f64 expected reference and the stale description of input-delta dtype. Commit **7685d58** contains those changes. It does **not** change the 2e-6 agreement bound or the f32-prefix test, and does not modify historical evaluation artifacts.
