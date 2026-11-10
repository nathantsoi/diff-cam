# Results package

Generated only from locally available typed records and the existing task-cluster audit machinery. This is not a finished 39-task campaign, a physical machining result, or a claim that a program is safe to run.

## Input availability

| dataset | state | expected_run_records | observed_run_records | reason |
|---|---|---|---|---|
| pilot_1 | available | 7 | 7 | validated immutable audit; all expected runs have D-04 verdicts |
| pilot_2 | pending | 7 | 1 | no finalized audit/report; 1/7 run records visible |
| pilot_init | pending | 14 | 14 | no finalized audit/report; 14/14 run records visible |
| campaign | absent | NA | NA | declared input directory is absent |

## Main method-versus-CAM outcomes

Overall rows only below; `family_breakdown.csv` retains every family separately. Pilot results are bounded one-task/dev evidence. Campaign main rows are restricted to the test split. A delta is method minus CAM, not an automatically beneficial direction. NA is unavailable, not zero.

| dataset | method | dialect | expected_tasks | geometry_pass_rate | safety_pass_rate | completion_pass_rate | delta_cycle_s | cycle_ci_lower | cycle_ci_upper | cycle_holm_p | cycle_paired_n | delta_air_fraction | air_holm_p | air_paired_n | no_improvement_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot_1 | annealed-adam | rs274 | 1 | 0 | 1 | 0 | NA | NA | NA | NA | 0 | NA | NA | 0 | NA |
| pilot_1 | annealed-adam | haas | 1 | 0 | 1 | 0 | NA | NA | NA | NA | 0 | NA | NA | 0 | NA |
| pilot_1 | fixed-adam | rs274 | 1 | 0 | 1 | 0 | NA | NA | NA | NA | 0 | NA | NA | 0 | NA |
| pilot_1 | fixed-adam | haas | 1 | 0 | 1 | 0 | NA | NA | NA | NA | 0 | NA | NA | 0 | NA |
| pilot_1 | hard-cmaes | rs274 | 1 | 0 | 1 | 0.5 | NA | NA | NA | NA | 0 | NA | NA | 0 | NA |
| pilot_1 | hard-cmaes | haas | 1 | 0 | 1 | 0.5 | NA | NA | NA | NA | 0 | NA | NA | 0 | NA |
| pilot_1 | nominal-cam | rs274 | 1 | 1 | 1 | 1 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pilot_1 | nominal-cam | haas | 1 | 1 | 1 | 1 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pilot_1 | smooth-cmaes | rs274 | 1 | 0 | 1 | 0 | NA | NA | NA | NA | 0 | NA | NA | 0 | NA |
| pilot_1 | smooth-cmaes | haas | 1 | 0 | 1 | 0 | NA | NA | NA | NA | 0 | NA | NA | 0 | NA |
| pilot_2 | annealed-adam | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pilot_2 | fixed-adam | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pilot_2 | hard-cmaes | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pilot_2 | nominal-cam | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pilot_2 | smooth-cmaes | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pilot_init | annealed-adam | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pilot_init | fixed-adam | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pilot_init | hard-cmaes | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pilot_init | nominal-cam | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pilot_init | smooth-cmaes | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| campaign | nominal-cam | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

### Unfavourable and limiting findings

- **pilot_1 / annealed-adam / rs274**: geometry_pass_rate = 0 over 1 declared task units.
- **pilot_1 / annealed-adam / rs274**: completion_pass_rate = 0 over 1 declared task units.
- **pilot_1 / annealed-adam / haas**: geometry_pass_rate = 0 over 1 declared task units.
- **pilot_1 / annealed-adam / haas**: completion_pass_rate = 0 over 1 declared task units.
- **pilot_1 / fixed-adam / rs274**: geometry_pass_rate = 0 over 1 declared task units.
- **pilot_1 / fixed-adam / rs274**: completion_pass_rate = 0 over 1 declared task units.
- **pilot_1 / fixed-adam / haas**: geometry_pass_rate = 0 over 1 declared task units.
- **pilot_1 / fixed-adam / haas**: completion_pass_rate = 0 over 1 declared task units.
- **pilot_1 / hard-cmaes / rs274**: geometry_pass_rate = 0 over 1 declared task units.
- **pilot_1 / hard-cmaes / rs274**: completion_pass_rate = 0.5 over 1 declared task units.
- **pilot_1 / hard-cmaes / haas**: geometry_pass_rate = 0 over 1 declared task units.
- **pilot_1 / hard-cmaes / haas**: completion_pass_rate = 0.5 over 1 declared task units.
- **pilot_1 / smooth-cmaes / rs274**: geometry_pass_rate = 0 over 1 declared task units.
- **pilot_1 / smooth-cmaes / rs274**: completion_pass_rate = 0 over 1 declared task units.
- **pilot_1 / smooth-cmaes / haas**: geometry_pass_rate = 0 over 1 declared task units.
- **pilot_1 / smooth-cmaes / haas**: completion_pass_rate = 0 over 1 declared task units.

## Initialization axis

| dataset | method | init | scope | expected_runs | observed_runs | failed_runs | collapse_count | collapse_rate | failure_details | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| pilot_1 | annealed-adam | cam | primary | 1 | 1 | 0 | NA | NA | [] | plain-CAM reference; no non-CAM denominator |
| pilot_1 | fixed-adam | cam | primary | 1 | 1 | 0 | NA | NA | [] | plain-CAM reference; no non-CAM denominator |
| pilot_1 | hard-cmaes | cam | primary | 2 | 2 | 0 | NA | NA | [] | plain-CAM reference; no non-CAM denominator |
| pilot_1 | nominal-cam | cam | primary | 1 | 1 | 0 | NA | NA | [] | plain-CAM reference; no non-CAM denominator |
| pilot_1 | smooth-cmaes | cam | primary | 2 | 2 | 0 | NA | NA | [] | plain-CAM reference; no non-CAM denominator |
| pilot_2 | annealed-adam | pending | pending | 1 | 0 | 0 | NA | NA | [] | no finalized audit/report; 1/7 run records visible |
| pilot_2 | fixed-adam | pending | pending | 1 | 0 | 0 | NA | NA | [] | no finalized audit/report; 1/7 run records visible |
| pilot_2 | hard-cmaes | pending | pending | 2 | 0 | 0 | NA | NA | [] | no finalized audit/report; 1/7 run records visible |
| pilot_2 | nominal-cam | cam | primary | 1 | 1 | 0 | NA | NA | [] | no finalized audit/report; 1/7 run records visible |
| pilot_2 | smooth-cmaes | pending | pending | 2 | 0 | 0 | NA | NA | [] | no finalized audit/report; 1/7 run records visible |
| pilot_init | annealed-adam | cam | exploratory_cam_perturbation | 1 | 1 | 0 | NA | NA | [] | exploratory CAM perturbation; not a plain-CAM parent or init-axis comparison |
| pilot_init | annealed-adam | cam | primary | 1 | 1 | 0 | NA | NA | [] | no finalized audit/report; 14/14 run records visible |
| pilot_init | annealed-adam | random | optimizer | 2 | 2 | 0 | NA | NA | [] | no finalized audit/report; 14/14 run records visible |
| pilot_init | annealed-adam | straight_line | optimizer | 1 | 1 | 0 | NA | NA | [] | no finalized audit/report; 14/14 run records visible |
| pilot_init | fixed-adam | cam | primary | 1 | 1 | 0 | NA | NA | [] | no finalized audit/report; 14/14 run records visible |
| pilot_init | hard-cmaes | cam | primary | 2 | 2 | 0 | NA | NA | [] | no finalized audit/report; 14/14 run records visible |
| pilot_init | hard-cmaes | random | optimizer | 2 | 2 | 0 | NA | NA | [] | no finalized audit/report; 14/14 run records visible |
| pilot_init | hard-cmaes | straight_line | optimizer | 1 | 1 | 0 | NA | NA | [] | no finalized audit/report; 14/14 run records visible |
| pilot_init | nominal-cam | cam | primary | 1 | 1 | 0 | NA | NA | [] | no finalized audit/report; 14/14 run records visible |
| pilot_init | smooth-cmaes | cam | primary | 2 | 2 | 0 | NA | NA | [] | no finalized audit/report; 14/14 run records visible |
| campaign | NA | NA | NA | NA | NA | NA | NA | NA | NA | declared input directory is absent |

CAM perturbation is exploratory and excluded from the init-axis comparison. A CAM semantic anchor is never replaced by a perturbation or inferred parent. Collapse is pending when the declared hard-loss, parent or 0.125 mm verdict evidence is missing.

## Selection vs 0.125 mm verdict and seed soft/hard gap

| dataset | run_id | init | dialect | feasible_at_025 | geometry_pass_at_0125 | agree | seed_soft_objective | seed_hard_objective | soft_minus_hard | no_improvement_over_init |
|---|---|---|---|---|---|---|---|---|---|---|
| pilot_1 | annealed-adam-s0-d41244bcb53e-e4f2c066a1bd | cam | rs274 | NA | false | NA | NA | NA | NA | NA |
| pilot_1 | annealed-adam-s0-d41244bcb53e-e4f2c066a1bd | cam | haas | NA | false | NA | NA | NA | NA | NA |
| pilot_1 | fixed-adam-s0-d41244bcb53e-e4f2c066a1bd | cam | rs274 | NA | false | NA | NA | NA | NA | NA |
| pilot_1 | fixed-adam-s0-d41244bcb53e-e4f2c066a1bd | cam | haas | NA | false | NA | NA | NA | NA | NA |
| pilot_1 | hard-cmaes-s0-d41244bcb53e-e4f2c066a1bd | cam | rs274 | NA | false | NA | NA | NA | NA | NA |
| pilot_1 | hard-cmaes-s0-d41244bcb53e-e4f2c066a1bd | cam | haas | NA | false | NA | NA | NA | NA | NA |
| pilot_1 | hard-cmaes-s1-d41244bcb53e-e4f2c066a1bd | cam | rs274 | NA | false | NA | NA | NA | NA | NA |
| pilot_1 | hard-cmaes-s1-d41244bcb53e-e4f2c066a1bd | cam | haas | NA | false | NA | NA | NA | NA | NA |
| pilot_1 | nominal-cam-s0-d41244bcb53e-e4f2c066a1bd | cam | rs274 | NA | true | NA | NA | NA | NA | NA |
| pilot_1 | nominal-cam-s0-d41244bcb53e-e4f2c066a1bd | cam | haas | NA | true | NA | NA | NA | NA | NA |
| pilot_1 | smooth-cmaes-s0-d41244bcb53e-e4f2c066a1bd | cam | rs274 | NA | false | NA | NA | NA | NA | NA |
| pilot_1 | smooth-cmaes-s0-d41244bcb53e-e4f2c066a1bd | cam | haas | NA | false | NA | NA | NA | NA | NA |
| pilot_1 | smooth-cmaes-s1-d41244bcb53e-e4f2c066a1bd | cam | rs274 | NA | false | NA | NA | NA | NA | NA |
| pilot_1 | smooth-cmaes-s1-d41244bcb53e-e4f2c066a1bd | cam | haas | NA | false | NA | NA | NA | NA | NA |
| pilot_2 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pilot_init | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| campaign | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

The 0.25 mm hard-selection flag and the independently posted 0.125 mm geometry verdict are separate roles; missing selection evidence is excluded rather than counted as agreement. The signed soft−hard gap at the matched seed θ is a model discrepancy, not a safety verdict.

## Resource costs

| dataset | run_id | role | candidate_evaluations | forward_calls | backward_calls | validation_calls | wall_time_s | phase_wall_time_s | process_peak_rss_bytes | memory_scope |
|---|---|---|---|---|---|---|---|---|---|---|
| pilot_1 | annealed-adam-s0-d41244bcb53e-e4f2c066a1bd | optimizer_total | 4 | 6 | 4 | 2 | 127.027980969 | NA | 3693989888 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_1 | annealed-adam-s0-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 157.625003256 | 447320064 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_1 | annealed-adam-s0-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 157.924927902 | 447389696 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_1 | fixed-adam-s0-d41244bcb53e-e4f2c066a1bd | optimizer_total | 4 | 6 | 4 | 2 | 136.767113386 | NA | 3704840192 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_1 | fixed-adam-s0-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 160.073462767 | 446513152 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_1 | fixed-adam-s0-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 159.272100091 | 446603264 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_1 | hard-cmaes-s0-d41244bcb53e-e4f2c066a1bd | optimizer_total | 4 | 6 | 0 | 2 | 2.85618747608 | NA | 3684737024 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_1 | hard-cmaes-s0-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 155.857238983 | 451203072 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_1 | hard-cmaes-s0-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 154.861209407 | 448770048 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_1 | hard-cmaes-s1-d41244bcb53e-e4f2c066a1bd | optimizer_total | 4 | 6 | 0 | 2 | 2.82005200605 | NA | 3678638080 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_1 | hard-cmaes-s1-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 157.818146559 | 451186688 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_1 | hard-cmaes-s1-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 159.263200753 | 449658880 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_1 | nominal-cam-s0-d41244bcb53e-e4f2c066a1bd | optimizer_total | 1 | 1 | 0 | 0 | 0.960285466048 | NA | 3678404608 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_1 | nominal-cam-s0-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 153.339742976 | 447852544 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_1 | nominal-cam-s0-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 154.085572024 | 450887680 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_1 | smooth-cmaes-s0-d41244bcb53e-e4f2c066a1bd | optimizer_total | 4 | 6 | 0 | 2 | 16.7368950251 | NA | 3678519296 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_1 | smooth-cmaes-s0-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 156.964848753 | 447381504 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_1 | smooth-cmaes-s0-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 157.809631679 | 447426560 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_1 | smooth-cmaes-s1-d41244bcb53e-e4f2c066a1bd | optimizer_total | 4 | 6 | 0 | 2 | 17.746536292 | NA | 3683065856 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_1 | smooth-cmaes-s1-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 152.915704346 | 447340544 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_1 | smooth-cmaes-s1-d41244bcb53e-e4f2c066a1bd | screen | NA | NA | NA | NA | NA | 154.815749236 | 447541248 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_2 | nominal-cam-s0-d41244bcb53e-3f814b17b81f | optimizer_total | 1 | 1 | 0 | 0 | 3.36529356998 | NA | 3673796608 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_2 | nominal-cam-s0-d41244bcb53e-3f814b17b81f | screen | NA | NA | NA | NA | NA | 350.265234752 | 451481600 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_2 | nominal-cam-s0-d41244bcb53e-3f814b17b81f | screen | NA | NA | NA | NA | NA | 336.757415245 | 451203072 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | annealed-adam-s0-d41244bcb53e-0d4b5694333c | optimizer_total | 25 | 31 | 24 | 6 | 824.816063129 | NA | 3700146176 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | annealed-adam-s0-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 203.407620171 | 451645440 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | annealed-adam-s0-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 208.005890138 | 449437696 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | annealed-adam-s0-d41244bcb53e-0d4b5694333c-cam_perturbed | optimizer_total | 25 | 31 | 24 | 6 | 814.921717544 | NA | 3698917376 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | annealed-adam-s0-d41244bcb53e-0d4b5694333c-cam_perturbed | screen | NA | NA | NA | NA | NA | 194.751729328 | 446775296 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | annealed-adam-s0-d41244bcb53e-0d4b5694333c-cam_perturbed | screen | NA | NA | NA | NA | NA | 192.250332076 | 446836736 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | annealed-adam-s0-d41244bcb53e-0d4b5694333c-random | optimizer_total | 25 | 31 | 24 | 6 | 794.488943885 | NA | 3700928512 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | annealed-adam-s0-d41244bcb53e-0d4b5694333c-random | screen | NA | NA | NA | NA | NA | 19.7346805311 | 447119360 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | annealed-adam-s0-d41244bcb53e-0d4b5694333c-random | screen | NA | NA | NA | NA | NA | 19.784701359 | 446562304 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | annealed-adam-s0-d41244bcb53e-0d4b5694333c-straight_line | optimizer_total | 25 | 31 | 24 | 6 | 805.228747379 | NA | 3681513472 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | annealed-adam-s0-d41244bcb53e-0d4b5694333c-straight_line | screen | NA | NA | NA | NA | NA | 9.73071819299 | 416534528 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | annealed-adam-s0-d41244bcb53e-0d4b5694333c-straight_line | screen | NA | NA | NA | NA | NA | 9.23124354205 | 416542720 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | annealed-adam-s1-d41244bcb53e-0d4b5694333c-random | optimizer_total | 25 | 31 | 24 | 6 | 801.241214243 | NA | 3703074816 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | annealed-adam-s1-d41244bcb53e-0d4b5694333c-random | screen | NA | NA | NA | NA | NA | 28.19727695 | 443658240 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | annealed-adam-s1-d41244bcb53e-0d4b5694333c-random | screen | NA | NA | NA | NA | NA | 28.1029362719 | 442314752 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | fixed-adam-s0-d41244bcb53e-0d4b5694333c | optimizer_total | 25 | 31 | 24 | 6 | 975.233934159 | NA | 3697782784 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | fixed-adam-s0-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 198.139727412 | 442810368 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | fixed-adam-s0-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 196.499227812 | 442109952 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | hard-cmaes-s0-d41244bcb53e-0d4b5694333c | optimizer_total | 25 | 29 | 0 | 4 | 64.3355102569 | NA | 3679313920 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | hard-cmaes-s0-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 208.44625414 | 450535424 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | hard-cmaes-s0-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 201.239705024 | 451051520 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | hard-cmaes-s0-d41244bcb53e-0d4b5694333c-random | optimizer_total | 25 | 29 | 0 | 4 | 64.694747468 | NA | 3689897984 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | hard-cmaes-s0-d41244bcb53e-0d4b5694333c-random | screen | NA | NA | NA | NA | NA | 24.089258878 | 452366336 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | hard-cmaes-s0-d41244bcb53e-0d4b5694333c-random | screen | NA | NA | NA | NA | NA | 26.885864359 | 452218880 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | hard-cmaes-s0-d41244bcb53e-0d4b5694333c-straight_line | optimizer_total | 25 | 29 | 0 | 4 | 64.508453373 | NA | 3672248320 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | hard-cmaes-s0-d41244bcb53e-0d4b5694333c-straight_line | screen | NA | NA | NA | NA | NA | 9.37858161691 | 416022528 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | hard-cmaes-s0-d41244bcb53e-0d4b5694333c-straight_line | screen | NA | NA | NA | NA | NA | 8.72894239402 | 416038912 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | hard-cmaes-s1-d41244bcb53e-0d4b5694333c | optimizer_total | 25 | 29 | 0 | 4 | 60.7681983321 | NA | 3677298688 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | hard-cmaes-s1-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 179.509207835 | 450478080 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | hard-cmaes-s1-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 182.0656623 | 452022272 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | hard-cmaes-s1-d41244bcb53e-0d4b5694333c-random | optimizer_total | 25 | 29 | 0 | 4 | 65.97159198 | NA | 3673014272 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | hard-cmaes-s1-d41244bcb53e-0d4b5694333c-random | screen | NA | NA | NA | NA | NA | 22.4357810559 | 447188992 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | hard-cmaes-s1-d41244bcb53e-0d4b5694333c-random | screen | NA | NA | NA | NA | NA | 22.8368295571 | 448294912 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | nominal-cam-s0-d41244bcb53e-0d4b5694333c | optimizer_total | 1 | 1 | 0 | 0 | 3.65489621798 | NA | 3674664960 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | nominal-cam-s0-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 334.506773751 | 451325952 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | nominal-cam-s0-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 346.148974702 | 451534848 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | smooth-cmaes-s0-d41244bcb53e-0d4b5694333c | optimizer_total | 25 | 29 | 0 | 4 | 163.173365303 | NA | 3684151296 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | smooth-cmaes-s0-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 181.190262712 | 452239360 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | smooth-cmaes-s0-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 178.993361669 | 451477504 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | smooth-cmaes-s1-d41244bcb53e-0d4b5694333c | optimizer_total | 25 | 29 | 0 | 4 | 160.884917483 | NA | 3683942400 | process high-water and Python traced allocations; not per-candidate device memory |
| pilot_init | smooth-cmaes-s1-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 171.379226868 | 451649536 | per-child process peak RSS; shared parent and all other system work excluded |
| pilot_init | smooth-cmaes-s1-d41244bcb53e-0d4b5694333c | screen | NA | NA | NA | NA | NA | 174.225211888 | 448315392 | per-child process peak RSS; shared parent and all other system work excluded |

Optimizer process peaks cover the whole optimizer process, not one candidate. Posted-worker rows retain their process-local role. Elapsed process durations, candidate ledger sums and phase walls are different quantities; concurrency and other system work are not converted into a universal speedup or aggregate-memory certificate.


## T3.3 AD/FD/randomized-hard alignment

| case_id | path_id | objective_mode | k | status | ad_fd_cosine | ad_fd_relative_difference_norm | ad_primary_cosine | ad_reference_cosine | ad_primary_relative_difference_norm | ad_reference_relative_difference_norm | primary_standard_error_rms | reference_standard_error_rms | actual_candidate_evaluations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | cam | geometry | 20 | complete | 0.999999308618 | 0.0012003853014 | 0.470381904125 | 0.401720831453 | 13.5184845005 | 8.00522312009 | 0.124069688199 | 0.0600433367949 | 178 |
| 1 | cam | geometry | 40 | complete | 0.999999985071 | 0.000440490957361 | 0.427499575269 | 0.379816964052 | 0.979491435 | 0.988925125426 | 0.124069688199 | 0.0600433367949 | 18 |
| 2 | cam | geometry | 80 | complete | 0.999999991168 | 0.000290141709442 | 0.426757767047 | 0.378290901922 | 0.962375903644 | 0.979381649684 | 0.124069688199 | 0.0600433367949 | 18 |
| 3 | cam | geometry | 160 | complete | 0.999999994793 | 0.000552823195821 | 0.42358490618 | 0.375258048097 | 0.942366775122 | 0.967488089492 | 0.124069688199 | 0.0600433367949 | 18 |
| 4 | cam-perturb-17 | geometry | 20 | complete | 0.957435842689 | 0.291030817247 | 0.274171535156 | 0.2005009862 | 5.93877431311 | 2.37563763732 | 1.13633082735 | 0.412304176968 | 178 |
| 5 | cam-perturb-17 | geometry | 40 | complete | 0.999415085738 | 0.131395499301 | 0.527026134867 | 0.655259534753 | 1.09590677809 | 0.777818352179 | 1.13633082735 | 0.412304176968 | 18 |
| 6 | cam-perturb-17 | geometry | 80 | complete | 0.999995144 | 0.187607122601 | 0.535840373352 | 0.681709547668 | 0.950630958562 | 0.793333773317 | 1.13633082735 | 0.412304176968 | 18 |
| 7 | cam-perturb-17 | geometry | 160 | complete | 0.99926979471 | 0.282464026853 | 0.543902288082 | 0.683180769392 | 0.972972396612 | 0.783419442741 | 1.13633082735 | 0.412304176968 | 18 |
| 8 | cam-perturb-29 | geometry | 20 | complete | 0.999477184466 | 0.0644286996686 | -0.397261874464 | -0.774945333792 | 3.644116576 | 2.91166110335 | 1.20038276905 | 0.661180550898 | 178 |
| 9 | cam-perturb-29 | geometry | 40 | complete | 0.999998679966 | 0.013622756273 | -0.177801132034 | -0.494660869546 | 2.44971733031 | 2.05228317657 | 1.20038276905 | 0.661180550898 | 18 |
| 10 | cam-perturb-29 | geometry | 80 | complete | 0.998706514891 | 0.163139932713 | -0.196637461712 | -0.430462641382 | 3.20753430233 | 2.48757442213 | 1.20038276905 | 0.661180550898 | 18 |
| 11 | cam-perturb-29 | geometry | 160 | complete | 0.997370355078 | 0.186850745616 | -0.183698831326 | -0.357941989964 | 3.91933455026 | 2.89752987873 | 1.20038276905 | 0.661180550898 | 18 |
| 12 | delta-line | geometry | 20 | complete | 0.999999999001 | 0.00121231532181 | NA | NA | NA | NA | 0 | 0 | 178 |
| 13 | delta-line | geometry | 40 | complete | 0.99999999925 | 0.00107546087828 | NA | NA | NA | NA | 0 | 0 | 18 |
| 14 | delta-line | geometry | 80 | complete | 0.999999999594 | 0.000174586924796 | NA | NA | NA | NA | 0 | 0 | 18 |
| 15 | delta-line | geometry | 160 | complete | NA | NA | NA | NA | NA | NA | 0 | 0 | 18 |
| 16 | cam | zero_weights | 40 | pending | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 17 | cam | study_efficiency | 40 | pending | NA | NA | NA | NA | NA | NA | NA | NA | NA |

These cases measure alignment, not gradient equality or unbiasedness. Randomized hard smoothing is a different surrogate; k-independent hard probes are reused here (not fresh samples at every k). Standard errors are reported, not hidden. FD alignment is on the declared coordinate subset and epsilon values. Declared-but-absent cases remain pending. Positive alignment/cosine does not establish machining correctness.

- **cam / k=20 / geometry**: AD/primary-randomized-hard cosine=0.470381904125; relative difference=13.5184845005. AD/reference cosine=0.401720831453. The differing surrogates remain materially distinct.
- **cam / k=40 / geometry**: AD/primary-randomized-hard cosine=0.427499575269; relative difference=0.979491435. AD/reference cosine=0.379816964052. The differing surrogates remain materially distinct.
- **cam / k=80 / geometry**: AD/primary-randomized-hard cosine=0.426757767047; relative difference=0.962375903644. AD/reference cosine=0.378290901922. The differing surrogates remain materially distinct.
- **cam / k=160 / geometry**: AD/primary-randomized-hard cosine=0.42358490618; relative difference=0.942366775122. AD/reference cosine=0.375258048097. The differing surrogates remain materially distinct.
- **cam-perturb-17 / k=20 / geometry**: AD/primary-randomized-hard cosine=0.274171535156; relative difference=5.93877431311. AD/reference cosine=0.2005009862. The differing surrogates remain materially distinct.
- **cam-perturb-17 / k=40 / geometry**: AD/primary-randomized-hard cosine=0.527026134867; relative difference=1.09590677809. AD/reference cosine=0.655259534753. The differing surrogates remain materially distinct.
- **cam-perturb-17 / k=80 / geometry**: AD/primary-randomized-hard cosine=0.535840373352; relative difference=0.950630958562. AD/reference cosine=0.681709547668. The differing surrogates remain materially distinct.
- **cam-perturb-17 / k=160 / geometry**: AD/primary-randomized-hard cosine=0.543902288082; relative difference=0.972972396612. AD/reference cosine=0.683180769392. The differing surrogates remain materially distinct.
- **cam-perturb-29 / k=20 / geometry**: AD/primary-randomized-hard cosine=-0.397261874464; relative difference=3.644116576. AD/reference cosine=-0.774945333792. The differing surrogates remain materially distinct.
- **cam-perturb-29 / k=40 / geometry**: AD/primary-randomized-hard cosine=-0.177801132034; relative difference=2.44971733031. AD/reference cosine=-0.494660869546. The differing surrogates remain materially distinct.
- **cam-perturb-29 / k=80 / geometry**: AD/primary-randomized-hard cosine=-0.196637461712; relative difference=3.20753430233. AD/reference cosine=-0.430462641382. The differing surrogates remain materially distinct.
- **cam-perturb-29 / k=160 / geometry**: AD/primary-randomized-hard cosine=-0.183698831326; relative difference=3.91933455026. AD/reference cosine=-0.357941989964. The differing surrogates remain materially distinct.

## Limitations and scope

- **CEM is unavailable**: hard CMA-ES is only a zero-order representative and must not be relabelled CEM.
- **Titan evidence is partial**; it is not re-executed or validated by this package.
- **CNC-Net is native-route only here**: no executable-NC performance evidence is inferred. CAMotics remains supplementary, not a silent correctness oracle.
- All optimizer/pilot/gradient values are simulation/CPU studies, not physical CNC runs, tool-monitoring calibrations, collision-free certificates or trial-ready directions.
- Efficiency weights are provisional where the source records declare them provisional (including 0.01/0.01 in D-11 pilot work); study-specific weights are likewise not globally frozen protocol values.
- The smooth time/air channels use the documented native surrogate; the n/reference factor restores total-time scale but does not assert exact native-to-hard timing equality.
- Pilot 1 is retained pre-D-09/D-10 evidence with its own geometry-only setup and failure modes. Later pilot inputs are not silently substituted for it, and pilot/campaign defaults must not be conflated.
- Selection at 0.25 mm and fine 0.125 mm posted verdicts remain separate. Unsupported/failed/missing outcomes are never silently admitted as success or automatic geometry passes.
- Verdict-only campaign inputs have no additional screen comparison. Empty diagnostic sets do not establish zero disagreement or manufacture a screen-agreement statistic.
- Test tasks are independent units; solver seeds are averaged within a task. Sparse paired counts and null intervals remain explicit. A pass on an isolated dev task does not establish suite-wide performance.
- Unmeasured resources and missing comparisons stay NA. Process-local memory does not prove global peak-memory sufficiency at higher concurrency.
- Pending pilot/campaign cases are not completed experiments. No claim of final protocol freezing, published-study completeness or manuscript acceptance is made.
