# Fixes: removing oracle information from the joint KF + QRD-RLS simulations

## Problem
Several scripts gave the estimator the quantities it is supposed to estimate, so the reported accuracy was not valid:

| Script | Leak |
|---|---|
| `GenerateFig4.m`, `GenerateFig5.m`, `ReproduceResults.m` | DOA regression used the **true** source signals (`Gamma_mat*X`); no Kalman filter was run |
| `GenerateFigs2And3.m` | KF used the **true** manifold matrix A(theta) and the **true** AR matrix F; the estimated DOA / AR coefficients were never fed back |
| `TestOnSpeechSignal.m` | KF used the true DOA; F came from Yule-Walker on the clean signal and was never updated (*not changed here*) |

## What changed
`JointKF_Corrected.m` is now the single estimator used by `GenerateFigs2And3.m`, `GenerateFig4.m`, `GenerateFig5.m` and `ParameterExperiments.m`. It uses only estimated quantities: the KF runs with A(theta_hat) and F(a_hat) from the previous interval, and the AR / DOA regressions use the KF state estimates. Settings are in `Defaults.m`.

Model and numerical fixes:
- newest-first state ordering (matches the companion matrix F); non-overlapping windows; the KF is continuous across intervals
- Hermitian measurement covariance (`I`) instead of `exp(1j*pi/4)*I`; process noise only on the first entry of each AR block; analytic AR power instead of a `randn(1,1e8)` Monte Carlo
- T matrix middle entry is sqrt(2) (T is unitary)
- cos/sin row pairing fixed: row `M+1-m` pairs with row `2M+2-m` and is divided by `d_m`; DOA from `atan2`; ambiguity of larger spacings resolved with the smallest spacing and all pairs averaged
- `givens_rotation.m` updated rows i and j from already-modified values (output not triangular, QRD-RLS off from exact weighted LS by ~1e-3); `QRD_RLS_Iteration.m` also overwrote the rotated matrix. Now matches exact weighted LS to ~1e-15
- separate RLS state per row group; no shared state across subarrays
- the AR-order-3 experiment crashed (`R1` is 3x3, initial vector had 2 entries); the initial AR guess now has the AR order's length
- Figs 4/5 use element positions lambda/2 and lambda (the paper's geometry). The old code used `D = (lambda/2)*ones(M,1)`, which places all sensors of a side at one position
- each window is processed 25 times (KF with updated theta_hat, DOA RLS state restored each pass); forgetting factor 0.995 (0.99 for Figs 4/5)

## Results with estimated quantities only
Seed 20 (Figs 2/3 and parameter experiments) and seed 40 (Figs 4/5). RMSE in degrees for the two sources, after a 20% burn-in:

| Case | DOA RMSE | Notes |
|---|---|---|
| Fig 2/3 (M=1, 30 dB) | 0.73 / 2.30 | AR RMSE 0.02-0.04; source 2 settles about 2 deg high |
| Fig 4 (N=3, 10 dB) | 1.93 / 4.47 | static source settles about 4 deg off; moving source about 2-3 deg off |
| Fig 5 (N=30, 30 dB) | 0.20 / 0.01 | tracks both sources |
| iSNR = 1 dB | 8.1 / 63.1 | fails |
| M = 4 | 0.30 / 1.74 | |
| AR order 3, M = 2 | 0.10 / 0.46 | |

Before the fixes the same scripts showed errors of roughly 0.03-0.3 deg, which came from the leaks.

## Limitations
- With M=1 and d=lambda/4 the two source signals are poorly separable from the KF; the estimator is biased at 30 dB and fails at low SNR. Larger aperture helps (Fig 5, M=4).
- Fig 4 (3 samples per window) is still several degrees off.
- Tuned on seed 1 only; Figs 4/5 results vary by seed (one other seed gave 3.3 / 2.0 for Fig 4).
- Not updated: `TestOnSpeechSignal.m` (the `.wav` files are not in the repo; it still uses the true DOA), `ReproduceResults.m`, `Report.pdf`, `Fig2.fig`, `Fig3.fig`. These show the old, leaky results and carry a warning header or need regenerating.
- The report's conclusions on robustness to SNR do not hold for the corrected estimator.
