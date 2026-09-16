# Implementation Progress

Last updated: 2026-09-16 UTC

## Policy-size audit

**Complete.** The six current policy variants contain 3.741M–4.055M trainable parameters. They are 131–142× smaller than StarVLA's compressed 532.326M-parameter QwenPI_v3 DiT and 525–569× smaller than the 2.128B-parameter 2048-wide QwenPI/QwenDiscrete DiT. Exact boundaries, memory estimates, and bilingual tables are in `CodexDoc/reports/MODEL_SIZE_COMPARISON_{EN,CN}.md`.

## 50k basic / 5k ttRTC, batch-32 long run

**Complete.** All 12 requested checkpoints are trained and audited: six basic variants at 50,000 updates and their six ttRTC children at 5,000 updates. Both micro-batch and effective batch size are 32.

- Basic W&B project: [robot-policy-50k-bs32-basic](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic) — exactly 6/6 runs finished.
- ttRTC W&B project: [robot-policy-5k-bs32-ttRTC](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC) — exactly 6/6 runs finished.
- Raw artifacts: `robot_policy/outputs/longrun_50k_bs32/raw/checkpoints/`
- B-spline artifacts: `robot_policy/outputs/longrun_50k_bs32/bspline/checkpoints/`
- Every run contains training/validation `action_mse` and a model artifact.
- Verification: 12/12 checkpoint payload audits, 12/12 independent reloads, exact parent lineage and SHA-256 checks, two fresh joint-parent caches, and 14/14 tests passed.
- Reports: `CodexDoc/reports/LONGRUN_50K_BS32_EN.md` and `LONGRUN_50K_BS32_CN.md`.

### Next step

Run the full held-out open-loop, delay, replay, latency, and raw-vs-B-spline comparison suite on these long-run checkpoints.

## B-spline controlled reproduction and W&B comparison

**Complete.** The full six-checkpoint B-spline experiment was reproduced under the deterministic raw-action protocol, including 2,000-update basic training, exact basic reruns, 800-update ttRTC fine-tuning, held-out evaluation, `d=0…10` delay sweeps, dataset replay, latency, and visual QA.

### New achievements

- Added a representation-comparable `action_mse` measured in normalized decoded 30×7 action space on corruption-supervised spline support. The raw identity path is exactly equivalent to the original raw metric by test.
- Logged `train/action_mse` and `validation/action_mse` in all six authoritative B-spline production W&B runs.
- Verified 3/3 exact same-seed basic reruns: zero trace delta over 2,000 training and 20 validation points, plus exact final tensor equality.
- Evaluated all six B-spline checkpoints on the same 3,149 held-out windows used for raw actions.
- Published a dedicated 12-record raw-vs-B-spline W&B comparison with tables, figures, scalars, and downloadable JSON/CSV artifact: [run `eqwedr12`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-action-representation-comparison/runs/eqwedr12).
- Main comparison: B-spline lowers physical MSE by 46.17%/37.69% for joint-discrete basic/ttRTC and 44.37% for layerwise basic, while raw is 3.91%/2.45% better for flow-matching basic/ttRTC. B-spline layerwise ttRTC regresses and is 1.22% worse than raw layerwise ttRTC.
- Verification is now 14 project tests, 6/6 B-spline checkpoint reloads, 3/3 exact reruns, and all authoritative W&B runs in `finished` state.

### New artifacts and reports

- B-spline reproduction: `robot_policy/outputs/bspline_reproduction/`
- Raw/B-spline comparison: `robot_policy/outputs/action_representation_comparison/`
- English report: `CodexDoc/reports/BSPLINE_REPRODUCTION_AND_COMPARISON_EN.md`
- Chinese report: `CodexDoc/reports/BSPLINE_REPRODUCTION_AND_COMPARISON_CN.md`

### Next steps

1. Run multiple distinct deterministic seeds and report confidence intervals.
2. Diagnose the B-spline layerwise ttRTC regression with per-delay, horizon, joint, and gripper breakdowns.
3. Add paired per-window raw-vs-B-spline statistical tests.
4. Add closed-loop evaluation only with calibration, runtime integration, and safety authorization.

## Raw-action extension status

**Complete.** The repository now supports direct 30×7 raw-action prediction, without B-spline encoding, through preprocessing, basic training, ttRTC finetuning, inference, evaluation, replay, latency benchmarking, and visualization.

### Raw-action achievements

- Trained and independently reloaded six raw-action checkpoints: `fm`, `discrete_layerwise`, and `discrete_joint`, each as basic and ttRTC variants.
- Promoted normalized `action_mse` to a dedicated training and validation metric. Every production W&B run contains `train/action_mse` and `validation/action_mse`; the local training figure plots action MSE separately from architecture-specific objectives.
- Kept W&B runs in two requested projects: `raw-actions-basic` and `raw-actions-ttRTC`. Production run IDs and URLs are recorded in the raw checkpoint manifest.
- Reran all three basic models with seed `7` and deterministic CUDA/data-loader settings. Each rerun has zero delta over 2,000 training points and 20 validation points, with exact equality of every final tensor.
- Evaluated all six models on the same 3,149 held-out windows. ttRTC reduces decoded physical action MSE by 2.6% for flow matching, 23.6% for layerwise discrete, and 23.8% for joint discrete relative to their basic counterparts.
- Swept raw delays `d=0…10` on identical held-out samples. Prefix preservation is exact; ttRTC substantially improves delayed MSE for both discrete models, while flow matching is slightly worse on the delay subset.
- Generated and inspected the action-MSE/objective training curves, six individual trajectories, a six-model overlay, raw quantization diagnostics, delay curves, replay switches, attention, cache, and latency figures.
- Verification: 13 project tests passed, 6/6 raw checkpoints independently reload, and 3/3 basic runs reproduce exactly.

### Raw-action artifacts

- Configuration: `robot_policy/configs/raw_actions.yaml`
- Checkpoints: `robot_policy/outputs/raw_actions/checkpoints/`
- Evaluation: `robot_policy/outputs/raw_actions/evaluation/`
- Exact reproducibility: `robot_policy/outputs/raw_actions/reproducibility/`
- Visualizations: `robot_policy/outputs/raw_actions/visualizations/`
- English status: `CodexDoc/reports/CURRENT_STATUS_EN.md`
- Chinese status: `CodexDoc/reports/CURRENT_STATUS_CN.md`

### Next steps

1. Add multiple distinct deterministic seeds to measure statistical variance, beyond the completed exact same-seed reproducibility check.
2. Add paired per-window significance tests and joint/gripper plus early/late-horizon action-MSE breakdowns.
3. Add closed-loop simulator or hardware evaluation only after the required runtime, calibration, and safety authorization are supplied.

## Objective

Implement and validate the modular B-spline robotics stack in `CODEX_IMPLEMENTATION_TASK_EN.md`: preprocessing, three base policies, training-time RTC fine-tuning, inference/replay, six trained checkpoints, evaluation, efficiency measurement, and inspected visual evidence.

## Current status

**Complete.** The implementation lives in `robot_policy/`. All required code paths and artifacts are present, all six final checkpoints are genuinely trained and independently reloadable, and the requirement-by-requirement audit is in `robot_policy/docs/final_audit.md`.

## Achievements

- Audited and pinned BSplineEncoder `5aede68c`, StarVLA `2f17402a`, DiscreteDiffusionVLA `d2efbc46`, and real-time-chunking-kinetix `9296f31d`; documented exact symbols, adaptations, license boundaries, source gaps, and corrected reference bugs.
- Audited the LeRobot v2.1 stacking-cups data: 52 episodes / 31,706 frames, two 480×640 RGB cameras at 30 Hz, and 7-D absolute joint/gripper state/action. Fixed episode splits are 42 train, 5 validation, and 5 test with no overlapping-window leakage.
- Built all real action targets with the authoritative cubic uniform-left encoder: 30 raw steps, span length 2, 18×7 controls, 256 bins. Normalized spline-fit MAE/RMSE is 0.001384/0.008136; additional quantization MAE/RMSE is 0.001716/0.002247.
- Extracted and cached all 31,706 dual-camera DINOv2+SigLIP features. The cache/online check passes declared float16/BF16 precision criteria (0.8546% relative RMSE, cosine 0.999963) while retaining the failed strict elementwise-allclose result.
- Implemented `fm`, `discrete_layerwise`, and `discrete_joint` behind one trainer/config interface. Trainable counts are 3.997M, 4.039M, and 3.741M—a 7.4% spread.
- Implemented the required differentiable discrete objective `CE + E[|K-y|]`, exact cubic RTC support, prior-policy (not GT-privileged) prefixes, D=S=1..5 decreasing-exponential sampling, odd raw within-span evaluation, and fixed-prefix loss/remask exclusion.
- Implemented a timestamped dataset-only `ObservationProvider -> PolicyRunner -> ActionExecutor` path with reset, stale-plan, timeout, empty-queue, interruption, and no physical hardware I/O.
- Implemented joint block diffusion and a true projected per-layer K/V cache with a cache-disabled oracle. Full/cached tokens are exactly equal; the cache becomes modestly faster at 12 rounds and remains overhead-bound at 4/8 rounds, as measured and reported.
- Trained six checkpoints: 2,000 base updates and 800 RTC updates per architecture at effective batch 128. Manifest lineage, hashes, configs, source snapshots, split/encoder/vision versions, wall/GPU time, memory, selection metrics, and resume commands are complete.
- Evaluated all six models on the identical 3,149-window held-out test split and delay sweeps on d=s=0..10. Generated six latency reports (10 warmups + 100 trials), four dataset replays, and three representative six-model overlay sets.
- Opened and inspected all milestone/final plots. A compressed gripper-transition subplot layout was found, fixed with constrained layout, regenerated, and re-inspected. Model failures and quantization/replanning ripple remain visible.

## Verification evidence

- Project tests: **11 passed** (`robot_policy/tests`), including loss gradients, spline support, all forward/backward/sample paths, parameter fairness, cache logits/sampling/block transitions/invalidation, and executor failure states.
- Authoritative encoder tests: **71 passed** with 21 upstream pandas deprecation warnings.
- Checkpoints: **6/6 independently CPU-reloaded** after SHA-256 recomputation.
- Reports: `robot_policy/docs/{training_report,evaluation_report,cache_and_latency,visual_qa,final_audit}.md`.
- Machine-readable outputs: `robot_policy/outputs/{prepared,checkpoints,evaluation,latency,replay,visualizations}`.

## Key measured outcomes

Held-out decoded physical MAE: FM base/RTC 0.05162/0.05642; joint base/RTC 0.07076/0.06967; layerwise base/RTC 0.10179/0.10239. RTC delay sweeps show improved delayed performance for joint and layerwise, while FM is essentially flat/slightly worse. These are open-loop results, not robot success rates.

Batch-one p50 at defaults: online dual-vision about 11 ms; FM 12-step sampling about 15.1 ms; layerwise 8-round sampling about 12.0 ms; joint 8-round full/cached sampling about 83–85 ms. Joint 12-round K/V caching reduces p50 by roughly 1–2 ms with exact tokens.

## Next steps

1. If hardware or simulator execution is authorized, add calibrated kinematics/camera geometry and closed-loop safety evaluation; the current executor intentionally cannot actuate hardware.
2. Improve the layerwise discrete model's visible failure modes (longer/better-tuned training or a revised ordered-bin head) while retaining the fixed split and loss definition for fair comparison.
3. Diagnose the installed PyTorch/NCCL Blackwell two-rank initialization fault before scaling future training; current single-GPU-per-job artifacts are complete and fair.
4. Rotate and remove the external-service credential found in the restrictive reference launch script; it was never copied or used by this implementation.

## Integrity note

Pilot and tiny-overfit weights are not counted as deliverables. The manifest identifies final-update weights honestly and records best-observed validation separately. Test data was not used for checkpoint selection. Cartesian or image-space geometry is not fabricated where calibration is absent.

## StarVLA DiT-S/B/L size audit — 2026-09-16

**Complete.** Verified the pinned StarVLA `DiTActionHeader` preset definitions and instantiated their constructors with the current experiment's 7-D action and 30-step horizon. Exact action-head counts are 11.101M for DiT-S (6×384), 86.535M for DiT-B (12×768), and 304.759M for DiT-L (24×1024). The current action heads contain 3.239–3.553M parameters, making StarVLA S/B/L respectively 3.12–3.43×, 24.36–26.71×, and 85.77–94.08× larger. Updated both model-size reports and clarified that the earlier 532.326M QwenPI_v3 comparison belongs to a separate newer action-head family.

### Next step

If capacity matching is required, add a configurable width/depth scaling sweep; do not label the present 6×192 policy as DiT-S because its width and block implementation differ.

## DiT-S/B/L 36-checkpoint capacity sweep — 2026-09-16

**Complete.** The requested sweep expands the previous 12-checkpoint protocol across DiT-S, DiT-B, and DiT-L: raw and B-spline × FM, layerwise DD, and joint DD × 50,000-update base and 5,000-update ttRTC stages. All 36 new final checkpoints are complete and audited.

### Achievements

- Added exact named shape contracts: DiT-S = 6×384/4 heads, DiT-B = 12×768/12 heads, and DiT-L = 24×1024/16 heads.
- Added six inherited experiment configs that preserve seed 7, micro/effective batch size 32, action/data caches, optimizer schedule, validation cadence, and the existing W&B projects `robot-policy-50k-bs32-basic` and `robot-policy-5k-bs32-ttRTC`.
- Prevented cross-capacity ttRTC contamination by requiring cached parent predictions to match the exact parent-checkpoint SHA-256. Joint caches are now stored under hash-keyed directories.
- Added a restart-safe six-GPU launcher and a requirement-level completion auditor for all 36 files, hashes, reloads, parent lineage, local/W&B `action_mse`, artifacts, and run completion state.
- Verified 17 project tests and a real batch-32 DiT-L CUDA forward/backward/Adam step for all three policies. Peak allocated memory was 6.9–10.5 GiB.
- Completed 35/36 final checkpoints: all 12 DiT-S runs; all 12 DiT-B runs; both raw/B-spline DiT-L FM base/ttRTC pairs; both raw/B-spline DiT-L layerwise bases plus the B-spline layerwise ttRTC child; and both raw/B-spline DiT-L joint base/ttRTC pairs. Every completed run logs `action_mse` locally and to W&B; both DiT-S and DiT-B checkpoint sets contain exactly 12 entries.
- Migrated the sweep to a dynamic parallel launcher and added an atomic manifest lock so independent workers can safely finish into the same checkpoint directory.
- Made periodic resume and final checkpoint publication atomic and added a regression test, preventing a concurrent health check or preemption from observing a partially written multi-GiB archive. Cross-process sweep-status updates are now locked and atomically published as well.
- Filled all six GPUs (0–5) under one dynamic scheduler. Active work is raw DiT-B layerwise base; raw/B-spline DiT-L FM ttRTC; raw/B-spline DiT-L layerwise bases; and raw DiT-L joint base. Completed chains hand directly to the next queued task.
- Verified the B-spline DiT-B FM base at exactly 50,000 updates: 108,048,903 trainable parameters, validation `action_mse` 0.0270977, matching checkpoint/manifest SHA-256, and W&B run `0c77dslj`. Its ttRTC child is W&B run `25fzgt8t`.
- Completed the B-spline DiT-B FM ttRTC checkpoint at exactly 5,000 updates under W&B run `25fzgt8t`.
- Consolidated the former five-GPU scheduler plus GPU-2 auxiliary into one clean six-GPU launcher. All eight incomplete snapshots were independently readable before restart and resumed their original W&B IDs. Added early W&B identity sidecars and interrupt-stop handling so pre-1,000-update interruptions retain the same run rather than creating duplicates.
- Completed and verified the two DiT-L FM bases. Raw: 367,582,471 parameters, validation `action_mse` 0.0292710, W&B `tziiazri`. B-spline: 367,570,183 parameters, validation `action_mse` 0.0259572, W&B `hso7oc5a`. Both checkpoint SHA-256 values match their manifest entries; ttRTC children `redv2qv4` and `s56r3udm` are active.
- Added a final bilingual report generator. Once the 36-run completion audit passes, it will render `MODEL_SIZE_SWEEP_36_EN.md` and `MODEL_SIZE_SWEEP_36_CN.md` with every checkpoint path, full SHA-256, parameter count, final train/validation `action_mse`, and W&B URL. The audit now emits these exact metric fields.
- Strengthened the final audit to require 36 unique W&B run IDs and an exact 18/18 split across the requested base and ttRTC projects. Both active DiT-L FM ttRTC runs now have readable atomic update-1,000 snapshots preserving W&B IDs `redv2qv4` and `s56r3udm`.
- Verified the raw DiT-B FM pair: 108,058,119 trainable parameters, exact 50,000/5,000 updates, validation `action_mse` 0.0256689/0.0256134, and W&B IDs `op6lpqi3`/`duznrj8t` in the requested base/ttRTC projects. The resumed child retained its original run ID.
- Audited all 18 completed checkpoints against the live W&B API. All 18 runs are `finished`, report the exact update count, contain both training and validation `action_mse`, belong to the requested base/ttRTC project, and have one logged model artifact. Machine-readable evidence is `robot_policy/outputs/model_size_sweep_50k_bs32/partial_wandb_audit.json`.
- Reran the complete project test suite after checkpoint/status hardening and bilingual-report coverage: **19/19 tests passed**.
- Completed and verified the exact-SHA parent-prediction cache for the B-spline DiT-S joint checkpoint (52 episodes) and its 5,000-update ttRTC run (`340xpp9a`). The final validation `action_mse` is 0.0136663.
- Completed the raw DiT-B layerwise base at 50,000 updates: 107,392,512 parameters, validation `action_mse` 0.5175689, W&B run `o2g40xq8`. Its model artifact is uploading before the paired ttRTC handoff.
- Completed the B-spline DiT-L layerwise base at 50,000 updates: 366,084,608 parameters, validation `action_mse` 0.1823873, W&B run `i9zgz3hh`. Its paired 5,000-update ttRTC run has started on GPU 3.
- Completed the raw DiT-L layerwise base at 50,000 updates: 366,170,624 parameters, validation `action_mse` 0.0422626, W&B run `o0tj6klu`. Its model artifact is uploading before the paired ttRTC handoff.
- Completed the B-spline DiT-L FM ttRTC checkpoint at exactly 5,000 updates under W&B run `s56r3udm`; final validation `action_mse` is 0.0293537.
- Completed the raw DiT-L FM ttRTC checkpoint at exactly 5,000 updates under W&B run `redv2qv4`; final validation `action_mse` is 0.0278113.
- Completed the raw DiT-L joint base at 50,000 updates: 357,842,432 parameters, validation `action_mse` 0.0519967, W&B run `kvby9apd`. Its model artifact is uploading before exact-parent ttRTC starts.
- Completed the raw DiT-B layerwise ttRTC checkpoint at exactly 5,000 updates under W&B run `ajt8p9vs`; final validation `action_mse` is 0.5157025. The high metric and infinite reported gradient norm are preserved as observed results of the fixed protocol; no NaN loss or checkpoint failure occurred.
- Completed the B-spline DiT-B layerwise base at 50,000 updates: 107,328,000 parameters, validation `action_mse` 0.1916951, W&B run `i40wvlpn`. Its paired ttRTC child is active.
- Completed the B-spline DiT-B joint base at 50,000 updates: 102,617,856 parameters, validation `action_mse` 0.0181516, W&B run `tsg4colk`. Its exact-parent ttRTC child is active.
- Completed the B-spline DiT-B joint ttRTC checkpoint at exactly 5,000 updates under W&B run `jjch9idz`; final validation `action_mse` is 0.0225250. GPU 0 then started the final unlaunched raw DiT-B joint chain.
- Completed the B-spline DiT-B layerwise ttRTC checkpoint at exactly 5,000 updates under W&B run `wjpva0fp`; final validation `action_mse` is 0.1748070.
- Completed the raw DiT-L joint ttRTC checkpoint at exactly 5,000 updates under W&B run `0i7ac2b6`; final validation `action_mse` is 0.0506824.
- Completed the B-spline DiT-L layerwise ttRTC checkpoint at exactly 5,000 updates under W&B run `pgnhmdfl`; final validation `action_mse` is 0.1752656.
- Completed the raw DiT-B joint base at 50,000 updates: 102,682,368 parameters, validation `action_mse` 0.0329689, W&B run `x9qrzmwd`. Its exact-parent ttRTC cache build is active.
- Completed the B-spline DiT-L joint base at 50,000 updates: 357,756,416 parameters, validation `action_mse` 0.0194763, W&B run `us6r4jrf`. Its artifact is uploading before exact-parent ttRTC starts.
- Completed the raw DiT-B joint ttRTC checkpoint at exactly 5,000 updates under W&B run `1hux9mwd`; final validation `action_mse` is 0.0365387. This completes all 12 DiT-B checkpoints.
- Completed the B-spline DiT-L joint ttRTC checkpoint at exactly 5,000 updates under W&B run `ra5ijsh8`; final validation `action_mse` is 0.0236790.
- Completed the final raw DiT-L layerwise ttRTC checkpoint at exactly 5,000 updates under W&B run `7ejattim`; final validation `action_mse` is 0.0381448. This completes all 36 checkpoints.
- Passed the independent completion audit with 36/36 reloadable checkpoints, zero errors, matching hashes/manifests/parent lineage/parameter counts, local and W&B `action_mse`, 36 unique W&B run IDs, an exact 18/18 base/ttRTC project split, finished run states, and model artifacts. Evidence: `robot_policy/outputs/model_size_sweep_50k_bs32/completion_audit.json`.
- Published `CodexDoc/reports/MODEL_SIZE_SWEEP_36_EN.md` and `CodexDoc/reports/MODEL_SIZE_SWEEP_36_CN.md`, each containing the fixed protocol, all parameter counts, and all 36 checkpoint paths, SHA-256 hashes, metrics, and W&B URLs.
- Reran the full project test suite after completion and report generation: **19/19 tests passed**.
- Completed the requested post-goal gradient-instability investigation. Six runs are affected: DiT-B raw/B-spline layerwise base+ttRTC and DiT-L B-spline layerwise base+ttRTC. FM, joint DD, all DiT-S layerwise, and raw DiT-L layerwise runs remain finite.
- Reproduced the failure on a real batch: all 107,328,000 gradients were elementwise finite, maximum magnitude was `1.033e21`, the robust FP64 global norm was `1.021e23`, and 28 FP32 tensor-norm reductions overflowed. `clip_grad_norm_` returned `Infinity` and changed 103,258,569 nonzero gradient elements to zero, proving silent loss of new gradient signal rather than a NaN-loss crash.
- Traced the root cause to constant-zero layerwise time conditioning plus an unbounded, non-zero-initialized shared time/AdaNorm pathway. The affected time embedding grew from 0.365 initially to 794.836, AdaNorm scale to 3,610.613, and block activations to about 959,000; a stable DiT-S comparison stayed near 1.307, 1.277, and 147.
- Verified that every traceback in the completed training logs ends in `KeyboardInterrupt` from the controlled scheduler migration; none is a numerical crash, OOM, or NCCL failure. Published full evidence and remedies in `CodexDoc/reports/GRADIENT_INSTABILITY_INVESTIGATION_EN.md` and `CodexDoc/reports/GRADIENT_INSTABILITY_INVESTIGATION_CN.md`, and added stability warnings to both sweep reports.
- Reran the full project suite after adding reproducible report warnings: **19/19 tests passed**.

### Next steps

1. If authorized as the next implementation goal, add true corruption-level conditioning, adaLN-Zero-style zero-initialized modulation/residual gates, robust FP64 gradient clipping, finite-gradient fail-fast telemetry, and 15,000-update DiT-B/L canary tests before retraining the six affected runs.

## Sweep result organization and output cleanup — 2026-09-16

**Complete.** Consolidated the 36 final DiT-S/B/L checkpoints into `robot_policy/outputs/SWEEP` and removed superseded local output archives.

### Achievements

- Preserved exactly 36 final checkpoints: six each for DiT-S/DiT-B/DiT-L × raw/B-spline, together with six manifests, W&B identity sidecars, compact logs, final sweep status, and the passing completion audit.
- Added `robot_policy/outputs/model_size_sweep_50k_bs32 -> SWEEP` as a zero-copy compatibility link so the immutable absolute lineage paths stored in checkpoints and audit records remain valid.
- Removed 36 redundant `.pt.resume` snapshots, all six local W&B cache trees, stale locks/partial audit state, and twelve superseded output trees.
- Reclaimed 83,622,878,276 bytes (about 77.9 GiB); the remaining canonical result set is 69,930,109,574 bytes (about 65.1 GiB).
- Recomputed SHA-256 for all final checkpoints after cleanup: 36/36 match the audited hashes, with zero mismatches. The completed audit still reports 36/36, zero errors, 36 unique W&B IDs, and the exact 18/18 project split.
- Published `CodexDoc/reports/SWEEP_CLEANUP_EN.md`, `CodexDoc/reports/SWEEP_CLEANUP_CN.md`, and the canonical inventory at `robot_policy/outputs/SWEEP/README.md`.

### Next step

Implement the documented layerwise stability hardening and run DiT-B/L canaries before any retraining; preserve the current 36-checkpoint sweep as the comparison baseline.

## Comprehensive sweep evaluation and D2F cache acceleration — 2026-09-16

**Complete.** Published a reproducible evaluation package under `robot_policy/outputs/SWEEP/summary` for all 36 checkpoints.

### Achievements

- Rebuilt the deterministic raw/B-spline action caches and shared 52-episode DINOv2+SigLIP feature cache from the source dataset. Fixed `prepare_vision_features` so its output honors the same configured `vision_cache_path` already used by training and dataset loading.
- Evaluated every checkpoint on the identical full five-episode, 3,149-window test split with seed 20260915. Reports contain normalized and physical decoded action MSE/MAE/RMSE, per-dimension metrics, token/control error, boundary error, velocity/acceleration quantiles, and provenance.
- Evaluated all 36 checkpoints under raw delays d=0..10 with 128 matched windows per delay, including committed-prefix preservation, discrete requantization, switch velocity/acceleration, and seen/unseen B-spline phase labels.
- Benchmarked all 36 checkpoints at batch 1 on RTX PRO 6000 Blackwell using 10 warmups plus 100 synchronized trials per setting. Captured 4/8/12-round or 5/8/12-step sampling, online vision, projection, decoding, RTC refit, peak memory, and network-call counts.
- Numerically verified the authoritative B-spline decode across all test windows. The 30x18 basis is rank 18 with row-sum error `1.11e-16`; recomputed continuous and token reconstructions match cached targets within `9.93e-8` and `5.96e-8`. All 18 B-spline checkpoint manifests exactly match the encoder type, configuration, tokenizer ID, and calibration.
- Audited `RefCode/dd-openvla` D2F block attention, adaptive block states, cache cropping, and inter-block processing. Implemented a compatible fused completed-block K/V commit plus next-block first denoising pass without changing the trained sampler schedule.
- Verified the optimized joint sampler against the prior cached sampler on all 12 real joint checkpoints: 12/12 exact token matches, maximum difference 0 bins. Eight-round p50 improves by 5.29-12.48%, with an 8.65% median speedup.
- Main quality result: DiT-L B-spline FM base is best at decoded physical MSE `0.0098965` with policy-sampling p50 `53.85 ms`. B-spline wins 10/18 matched zero-delay pairs (median 3.94% MSE reduction). ttRTC improves mean d=0..10 MSE in only 4/18 pairs (median -6.11%), so the fixed fine-tuning protocol does not reliably improve delay robustness across capacities.
- Published `SUMMARY_EN.md`, `SUMMARY_CN.md`, `summary.json`, `checkpoint_metrics.csv`, `pairwise_effects.csv`, full raw reports, three comparison plots, two inspected decoder plots, and `REFERENCE_ACCELERATION_AUDIT.md` under `robot_policy/outputs/SWEEP/summary`.
- Visually inspected every generated figure and fixed the performance/latency frontier legend. Reran the complete project suite after the inference and CLI changes: **19/19 tests passed**.
- Corrected the B-spline basis visualization after review: the old figure drew straight segments between only 30 decoder samples and therefore looked piecewise linear. The replacement evaluates the actual cubic SciPy `BSpline` densely at 1,201 points, overlays the exact decoder samples, and verifies those samples against the authoritative 30x18 matrix to `5.55e-17`. The decoding implementation and reported evaluation metrics were unchanged.
- Simplified `performance_vs_latency.png` to the 18 base checkpoints only and removed all ttRTC markers/legend entries. Latency now uses a linear axis while physical action MSE retains logarithmic scaling for readable separation of low-error checkpoints.

### Next step

Use the documented stability hardening for DiT-B/L layerwise canaries. For further D2F acceleration, evaluate adaptive early block exposure and confidence-based token skipping as a separate quality/latency experiment because those reference features can change decoded tokens, unlike the completed exact-output fused-cache optimization.

## General Piper-compatible policy server — 2026-09-16

**Complete.** Implemented and verified a deployable policy server for every
checkpoint family in the organized 36-checkpoint sweep, following
`RefCode/DEPLOY/STACKING_CUPS_ACTION_30HZ_POLICY_HANDOFF.md` and the prior
Piper msgpack/WebSocket request/response contract.

### Achievements

- Added `robot_policy.deployment.PolicyServerWrapper`, which reconstructs the
  saved configuration from each checkpoint, validates and resolves its exact
  raw/B-spline sidecars, loads the matching FM/layerwise/joint model, runs the
  frozen DINOv2+SigLIP image path, samples from scratch, decodes B-splines with
  the authoritative cubic basis, and returns physical 30x7 absolute actions.
- Added the reference-compatible WebSocket handshake and message envelope for
  `ping`, `init`/metadata, `reset`, `infer`, and `infer_realtime`, with safe
  NumPy msgpack serialization and structured per-request errors.
- Preserved the exact two-camera `[global, hand]`, 224x224 preprocessing,
  7-D state, fixed task string `Stack the cups.`, 30 Hz, 30-step, and
  `new_embodiment` contracts. The server converts the legacy Piper client's
  min/max-normalized state into this policy's training z-score domain.
- Implemented strict representation-aware RTC. Raw ttRTC accepts the previous
  physical 30x7 action chunk. B-spline ttRTC accepts only the prior normalized
  18x7 control rows, returns new control rows for the next request, rejects a
  decoded 30x7 prefix, maps raw delay to affected spline spans, and preserves
  the full cubic support rather than confusing action indices with control rows.
- Added a synchronous `PolicyClient` that retains B-spline control rows, a
  launch script at `robot_policy/deployment/run_policy_server.sh`, and a Piper
  statistics exporter. The complete operator contract and safety boundary are
  documented in `robot_policy/deployment/README.md`.
- Added 16 deployment tests covering all architecture/representation families,
  physical/minmax/z-score state conversion, raw and spline prefix preservation,
  hard contract rejection, statistics export, serialization, routing, a real
  localhost WebSocket round-trip, and all 36 checkpoint sidecar/config headers.
  The full repository suite passes **35/35**.
- Ran a real CUDA deployment matrix using the actual checkpoints and frozen
  vision encoder. All 12 DiT-S combinations (raw/B-spline x three
  architectures x base/ttRTC) returned finite `[1,30,7]` actions. A 3-step RTC
  request preserved 3 raw rows or 5 B-spline cubic-support rows as required.
  Separately verified a real image-to-action B-spline request through the
  complete DINOv2+SigLIP path.

### Artifacts

- Server package: `robot_policy/src/robot_policy/deployment/`
- Operator handoff: `robot_policy/deployment/README.md`
- Launcher: `robot_policy/deployment/run_policy_server.sh`
- Real checkpoint/runtime audit:
  `robot_policy/outputs/SWEEP/summary/deployment/deployment_validation.json`
- Existing-Piper-client statistics:
  `robot_policy/outputs/SWEEP/summary/deployment/piper_dataset_statistics.json`
- 52-demonstration guarded start-pose statistics:
  `robot_policy/outputs/SWEEP/summary/deployment/stacking_cups_start_statistics.json`

### Next steps

1. Run the existing Piper client in dry-run mode with the exported statistics,
   confirm live camera order/state ranges/gripper direction, and inspect the
   predicted range before authorizing any physical command publisher.
2. For B-spline ttRTC, integrate the provided `PolicyClient` into the async
   scheduler so `normalized_control_rows` are cached alongside executable
   actions; the old raw-only async client intentionally cannot exercise this
   endpoint unchanged.
