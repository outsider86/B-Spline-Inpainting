# Implementation Progress

Last updated: 2026-09-22 UTC

## Future FM checkpoint cadence — 2026-09-22

For subsequent training, retain complete validation every epoch but persist
validation-best weights only once per 10-epoch interval. Keep the exact
within-interval best EMA state in memory and atomically flush it at epochs
10/20/30/...; do not reduce validation itself to a 10-epoch cadence. This
preserves exact selected epochs while removing repeated ~341 MiB writes.

## V4 FM three-dataset models finalized — 2026-09-22

**All six raw/B-spline h2 Flow-Matching runs are stopped and all GPUs are
free.** Complete validation ran every loader epoch. Local `base.pt` files now
contain the best observed validation EMA weights; W&B model artifact upload is
disabled.

- stacking_cup: raw epoch 41 / MSE 0.0229889; B-spline epoch 50 / 0.0214426.
- classify_blocks: raw epoch 64 / MSE 0.0311488; B-spline epoch 16 / 0.0304322.
- hanging_mug: raw update 33,360 / MSE 0.0150083; B-spline update 41,283 /
  0.0152258.
- All six bases clean-load. The two classify bases are explicitly
  inference-only because training was stopped before an optimizer snapshot;
  this does not affect deployment or evaluation.

### Next step

Run open-loop and GT-prefix RTC evaluation on the six finalized best models.
See `CodexDoc/reports/FM_V4_FINAL_MODELS_{EN,CN}.md`.

## V4 FM multi-dataset 600-epoch runs — active — 2026-09-22

**Six raw/B-spline h2 Flow-Matching runs are active on GPUs 0–5, with GPU 6
free.** The earlier stacking-cup runs with validation every 10 epochs were
stopped and preserved. Fresh stacking runs and all four classify-blocks /
hanging-mug runs now execute complete validation after every loader epoch,
giving 600 dense W&B validation points per run. Validation is never sampled.

- stacking_cup: 55/6/0 episodes, 40,748/4,537 train/val windows, 636
  updates/epoch, 381,600 updates total; GPUs 2/3.
- classify_blocks: 45/5/0 episodes, 88,315/9,673 windows, 1,379
  updates/epoch, 827,400 total; GPUs 0/1.
- hanging_mug: **complete at epoch 100**. Raw `base.pt` selects update 33,360
  (validation action MSE 0.0150083); B-spline selects update 41,283
  (0.0152258). Both epoch-100 snapshots and validation-best bases clean-load.
  GPUs 4/5 are free.
- All runs use batch/effective batch 64/64, two timesteps × two cameras,
  scratch ResNet-18 vision encoders, no test split, and exact snapshots every
  100 epochs through epoch 600.

The first complete validation passed for all six variants. All observed losses and gradient norms are finite,
with no skipped optimizer steps. The user will actively inspect convergence
and may request stopping at epoch 100. See
`CodexDoc/reports/FM_V4_600E_MULTI_DATASET_STATUS_{EN,CN}.md`.

### Next step

Monitor the remaining four dense validation curves and stop them at epoch 100
if requested; otherwise continue to their next scheduled checkpoints. Model
checkpoints remain local in scratch; W&B artifact upload is disabled.

## BSP scratch-vision Flow Matching v4 — authoritative batch-4 rerun — 2026-09-22

**PiGDM VJP diagnostic update:** the completed batch-64 base-FM checkpoints do
not exhibit meaningful RTC suffix conditioning. On a paired four-example
motion-rich validation cohort, PiGDM differs from naive `GT prefix + scratch
suffix` stitching by only 0.00160 physical RMS for raw and 0.00312 for
B-spline. The cross-VJP gain from condition error into mutable controls is only
0.00925/0.01918. The implementation matches the Kinetix equation and its
linear reference test; the failure is weak endpoint-Jacobian propagation, not
an obvious formula transcription bug. Full traces and direct-stitch comparison
plots are documented in `CodexDoc/reports/FM_PIGDM_VJP_DIAGNOSTIC_{EN,CN}.md`.

The clean batch-4 rerun was stopped once `validation/samples=3149` confirmed
that the historical discrepancy came from the old 256-window validation cap.
The existing completed batch-4 and batch-64 raw/B-spline checkpoints were then
evaluated over all 28,557 train and 3,149 validation windows.

- Batch 64 remains better in absolute validation open-loop MSE: 0.009458 raw
  and 0.009367 B-spline, versus 0.011329/0.011308 for batch 4.
- Batch 64 nevertheless has a much larger validation/train gap: 25.72× raw and
  22.26× B-spline, versus 1.87×/1.80× for batch 4.
- Prefix-6 GT-conditioned RTC suffix MSE is 0.010515 raw / 0.010025 B-spline
  for batch 64 and 0.011510/0.010283 for batch 4.
- Train and validation figures use the same motion-rich GT batches for raw and
  B-spline. Their blue RTC trajectory is exact GT prefix plus generated suffix;
  all saved artifacts pass bitwise prefix equality and zero committed-prefix
  MSE.
- Complete metrics and figures are in
  `robot_policy/outputs/BSP_UNET_V4/summary/authoritative_fullsplit/`.

### Next step

Treat batch 64 as the stronger open-loop checkpoint but do not deploy its base
PiGDM RTC path as effective conditioning. Test guidance/step sensitivity and a
ttRTC-finetuned checkpoint against the explicit naive-stitch baseline.

## BSP scratch-vision U-Net v3 — implementation complete, 16-checkpoint run starting — 2026-09-19

**Implementation and preflight complete; training in progress.** The new v3
policy path reproduces the reference BSP observation and temporal-backbone
design while retaining this repository's raw/B-spline codecs and RTC contract.

- Confirmed from source that each reference camera owns an independent
  `pretrained=False` ResNet-18, and that the image encoders are included in the
  policy optimizer and trained jointly from scratch.
- Added independent per-camera scratch ResNet-18 + GroupNorm, 32-keypoint
  SpatialSoftmax, `Linear(64,64)+ReLU`, and direct normalized-state
  concatenation. One- and consecutive-two-frame inputs produce 135-D or 270-D
  global conditions for this two-camera, 7-state dataset. DINOv2/SigLIP are not
  used by these models.
- Added the reference-sized `[256,512,1024]` conditional temporal U-Net in two
  policy families: continuous flow matching and 256-bin discrete token
  diffusion. The discrete policy shares monotonic block corruption and
  iterative MaskGIT-style unmasking and generates from a completely masked
  sequence.
- Both continuous and discrete RTC reuse the exact hard mask. For B-splines,
  only the union of control rows supporting the affected spans is immutable.
- Added reference optimizer/EMA settings: batch 64, AdamW `1e-4`, betas
  `(0.95,0.999)`, weight decay `1e-6`, 500-step warmup, cosine decay, and BSP
  EMA warmup with power `0.75` and maximum `0.9999`.
- Full-size GPU smoke passed: FM has 89,254,855 parameters and discrete DD has
  90,055,360; forward, backward, generation from scratch, EMA checkpoint save,
  clean reload, and deployment all pass. The full suite passes **62/62 tests**.
- Built and length-validated the shared RGB cache: 52 episodes, 31,706 frames,
  two cameras, uint8 CHW at 84x84, approximately 1.3 GB.
- Added a six-GPU queue for all eight base plus eight RTC checkpoints, and an
  evaluation runner for held-out generation-from-scratch accuracy, latency,
  train/test RTC trajectories, and the requested base-only linear-latency / log
  action-MSE visualization.

### Active experiment matrix

`{FM, discrete DD} × {raw, B-spline} × {1 frame, 2 frames} × {base, RTC}` =
16 checkpoints. Base training starts at 50,000 updates; RTC uses 5,000 updates.
Six GPUs are kept occupied through base training, exact-parent prediction-cache
generation, and RTC fine-tuning.

### Next step

Monitor convergence and gradient health at every 500-update validation point.
Extend beyond 50k only if the from-scratch validation action MSE has not
converged. After all 16 checkpoints finish, run the complete evaluation,
deployment audit, bilingual result summary, and upload to `NewModel/v3`.

## Exact B-spline RTC hard mask and policy-architecture audit — 2026-09-19

**Complete.** Audited the B-spline RTC condition
path end-to-end and confirmed that continuousRTC (FM) and discreteRTC (joint DD)
share the exact hard support mask. For `D` affected cubic spans, only control
rows `[0,D+3)` are fixed. FM reapplies those rows before and after every Euler
step; joint DD keeps the corresponding tokens immutable through unmasking.

- Replaced remaining implicit spline constants in training, inference,
  evaluation, and latency paths with checkpoint/config-derived geometry.
- Added deployment metadata that explicitly declares hard-mask type and exact
  affected-span support scope.
- Added basis-level and policy-level regressions: the mask must equal the
  authoritative basis support, and values outside it must have zero effect on
  same-seed FM/joint-DD loss and generation.
- The complete project suite passes **50/50 tests**, including deployment
  metadata checks for the hard-mask contract.
- Audited `RefCode/bspline-policy` at policy-architecture scope. Its documented
  default is a globally conditioned temporal 1D U-Net trained with epsilon
  diffusion and sampled with DDIM; its optional Transformer is a separate
  causal encoder-decoder path. Our FM policy is a non-causal alternating
  cross/self-attention DiT trained as a continuous velocity field.
- Expanded the reference audit down to its exact image and whole-policy input
  tensors: independent non-pretrained ResNet-18 + 32-keypoint SpatialSoftmax
  produces one 64-D vector per camera per observation step; low-dimensional
  state is concatenated directly, and the default U-Net flattens two observation
  steps into one 144/272/440-D task-dependent global FiLM condition.
- Reports:
  `CodexDoc/reports/RTC_BSPLINE_HARD_MASK_AUDIT_{EN,CN}.md` and
  `CodexDoc/reports/BSPLINE_POLICY_VS_FM_ARCH_CN.md`.

### Next step

Use the architecture audit to design a parameter-matched U-Net/DDIM versus
DiT/FM ablation on the same observation and action interfaces. Keep FM and
joint DD as the only active policy families and DiT-S/B as the active sizes.

## NewModel v1/v2 publication and RTC deployment verification — 2026-09-18

**Complete.** Reorganized the Hugging Face dataset release at
[`DiscreteRTC/dRTC/NewModel`](https://huggingface.co/datasets/DiscreteRTC/dRTC/tree/main/NewModel)
into explicit version folders and verified the deployed RTC interface through
real GPU-backed msgpack/WebSocket requests.

- `NewModel/v1` is a byte-for-byte server-side copy of the previous release:
  77 files and 36 checkpoints, including historical DiT-L and layerwise DD.
  All copied sizes and LFS SHA-256 values match before the old unversioned
  copies were removed.
- `NewModel/v2` contains 75 files and exactly 16 active checkpoints:
  DiT-S/B × raw/B-spline × FM/joint DD × base/ttRTC. All 16 remote LFS hashes
  exactly match the local checkpoints; v2 contains no DiT-L or layerwise file.
- v2 also includes four configs, eight codec/normalization sidecars, bilingual
  summaries, evaluation tables and plots, deployment matrices, decoder audit,
  W&B identity, and RTC deployment documentation.
- A raw FM ttRTC checkpoint completed a real WebSocket `infer_realtime` call
  with a physical `[1,30,7]` previous action chunk and produced finite
  `[1,30,7]` output while preserving three delayed action rows.
- A B-spline joint-DD ttRTC checkpoint completed the same wire-level call with
  normalized `[1,18,7]` previous control rows and produced finite `[1,30,7]`
  output, mapping delay 3 to two spans and five fixed cubic-support rows.
- Hardened the transport boundary to copy read-only msgpack NumPy views before
  torch conversion, and added automatic `vN/sidecars/{raw,bspline}` discovery.
  The project suite now passes 48/48 tests.

### Next step

Keep robot execution disabled while running the Piper live client in dry-run
mode against v2. Verify real camera ordering, state coordinates, gripper
direction, timing, start-pose guard, action limits, and emergency-stop behavior
before considering any closed-loop command publication.

## Active 16-checkpoint DiT-S/B comparison complete — 2026-09-18

**Complete.** The post-DiT-L active matrix now contains exactly 16 audited
checkpoints: DiT-S/B × raw/B-spline × FM/joint DD × base/ttRTC. The last
DiT-B B-spline joint-DD ttRTC child completed from its exact parent; all four
evaluation families, both 8-case deployment matrices, the 45-test suite, and
the local plus remote-W&B completion audits pass.

- Best held-out from-scratch physical action MSE across all 16 checkpoints is
  `0.00818665` from DiT-B/raw/FM/ttRTC. The corresponding base result is
  `0.00820306` at `68.15 ms` policy-sampling p50.
- Across the eight base policies used in the requested accuracy/latency plot,
  DiT-S/B B-spline FM form the practical low-latency frontier at
  `0.00850626 / 24.01 ms` and `0.00833015 / 37.31 ms`; DiT-B raw FM reaches
  `0.00820306 / 68.15 ms` for the lowest base MSE.
- B-spline improves both joint-DD base models: physical MSE falls by 18.20%
  for DiT-S and 11.74% for DiT-B, while sampling latency falls by 46.38% and
  43.70%, respectively. Every joint-DD model remains slower and less accurate
  than its capacity-matched FM result on this open-loop benchmark.
- The requested base-only visualization uses a linear latency x-axis and log
  from-scratch physical-MSE y-axis, with no ttRTC points:
  `robot_policy/outputs/FULL_VISION_512/summary/latency_vs_generation_from_scratch.png`.
- The 16-row W&B aggregate comparison is finished at run `wt2pf4rp`; bilingual
  reports are `CodexDoc/reports/FULL_VISION_512_16_{EN,CN}.md`.

### Next step

Use the audited tables to investigate why joint DD trails FM despite B-spline's
consistent joint-policy gain, prioritizing unmasking-round quality/latency,
token error by action dimension, and held-out episode-level failure clusters.
Keep DiT-L and layerwise DD excluded from all new training and evaluation.

## DiT-L retired from active research — 2026-09-18

**Complete.** Stopped both active full-vision DiT-L joint-DD base-training
groups (raw and B-spline) and the two dependent 24-checkpoint gate waiters.
GPUs 1–4 were released; the unrelated active job on GPU 0 was not touched.
No DiT-L files were deleted.

DiT-L now follows the same legacy/load-only policy as layerwise DD. Active
training, open-loop evaluation, RTC evaluation, inference-RTC evaluation, and
latency evaluation reject DiT-L. Active sweep/evaluation/deployment defaults
now contain only DiT-S/B, reducing the full-vision target from 24 to 16
checkpoints. Existing DiT-L configs and artifacts remain loadable for historical
reproducibility.

### Next step

Complete the remaining DiT-B FM/joint-DD work and generate the 16-checkpoint
DiT-S/B comparison; do not restart DiT-L or layerwise-DD jobs.

## Shared RTCEVAL comparison axes — 2026-09-18

**Complete.** RTCEVAL comparison figures now use an explicit shared log physical-MSE y-axis within each DiT size across raw/B-spline, FM/joint-DD, and base/ttRTC. The same limits are also applied to every corresponding per-checkpoint `metrics_vs_prefix.png`, and are recorded in each `report.json` under `metric_plot_axis_limits`. The trajectory panels were independently verified to already use one identical seven-channel physical-action min/max contract computed across the full dataset for both raw and B-spline.

### Next step

Regenerate the shared axes automatically whenever each remaining DiT-B checkpoint becomes ready; the restart-safe RTCEVAL summarizer now performs this refresh.

## DiT-S B-spline FM training-split inference-RTC diagnostic — 2026-09-18

**Complete.** Re-evaluated `dit_s/bspline/checkpoints/fm_base.pt` on 64 deterministic, episode-balanced complete chunks from the training split, without overwriting the held-out test result. The protocol is otherwise identical: 512 vision tokens, ground-truth prefixes of 2/4/6/8/10 raw actions, representation-native B-spline fixed-prefix inpainting from scratch, and full-dataset physical min/max axes.

- Training outputs: `robot_policy/outputs/FULL_VISION_512/summary/inference_rtc_train/dit_s/bspline_fm_base/`
- At the plotted six-step prefix, training suffix physical action MSE is `0.0005762174`, versus `0.0088039580` on test (test is 15.28× higher); MAE is `0.0111050` versus `0.0407958`.
- Across all five prefixes, training MSE is 15.28–24.14× lower than test. The training trajectories visually track the targets closely, so the poor held-out examples primarily expose a generalization/split-distribution gap rather than failure to fit training trajectories.
- This remains an oracle-prefix inference diagnostic, not the from-scratch validation metric used for checkpoint selection.

### Next step

Add paired train/validation/test, per-episode and per-joint breakdowns before attributing the gap to any specific task or state distribution; keep the held-out test split authoritative for generalization claims.

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

## Hugging Face checkpoint publication — 2026-09-16

**Complete.** Published the organized sweep to the public dataset repository
[`DiscreteRTC/dRTC`](https://huggingface.co/datasets/DiscreteRTC/dRTC/tree/main/NewModel)
under `NewModel/`.

### Achievements

- Uploaded all **36 checkpoint `.pt` files** plus 41 manifests, sidecars,
  summaries, and deployment-validation artifacts: **77 files total**.
- Published **69,897,068,420 checkpoint bytes (65.10 GiB)** while preserving
  the `dit_s`/`dit_b`/`dit_l`, `raw`/`bspline`, and checkpoint-family layout.
- Independently enumerated the remote repository after upload: remote and local
  inventories match exactly at 77/77 files, with no missing or extra paths and
  no size mismatches.
- Compared every remote checkpoint's LFS SHA-256 against the six authoritative
  local checkpoint manifests: **36/36 hashes match**.
- Verified Hugging Face commits
  `449f81788a73a10fcbdba992d17c0e9443f692af` and
  `d3cc53fb9f704480b87521be9acede7a97965080`.

### Next step

Use the immutable LFS hashes in the uploaded checkpoint manifests when
downloading models for deployment; no further publication work is required.

## Ground-truth-prefix inference RTC evaluation — 2026-09-17

**Complete.** Added and executed a representation-correct conditional
inpainting evaluation over all 12 DiT-S checkpoint variants.

### Achievements

- Added the reusable `evaluate_inference_rtc` command. It loads the current
  observation plus a ground-truth action-chunk prefix, freezes that condition,
  and generates the remaining chunk from scratch for FM, layerwise DD, and
  joint DD checkpoints.
- Raw actions freeze exactly the requested prefix. B-splines freeze the full
  cubic support, `ceil(d/2)+3` controls, so raw action steps are never confused
  with control-point indices.
- Added a four-GPU 12-way runner covering raw/B-spline × three architectures ×
  base/ttRTC, with deterministic episode-balanced sampling and prefix lengths
  2/4/6/8/10.
- Evaluated 64 identical complete test chunks per variant. All 12 variants
  produced finite suffix metrics; every conditioned control was preserved
  exactly (`fixed_control_max_abs = 0`) across all 60 variant/prefix cases.
- Saved exactly 12 directly named folders under `robot_policy/outputs/RTCEVAL`,
  each with aggregate JSON, 320 per-sample rows, source trajectory arrays, a
  five-example seven-channel trajectory grid, and a prefix-length metric plot.
- Added root `SUMMARY.md`, `summary.json`, `summary.csv`, and
  `inpainting_comparison.png`. At the six-action primary prefix, B-spline FM
  ttRTC is best at physical suffix MSE `0.00807522`.
- Regenerated all trajectory grids with the true physical dataset min/max as a
  shared y-axis per action channel, computed across all 52 prepared episodes.
  Predictions outside those bounds are explicitly shown with red boundary
  triangles and counted in each report rather than silently clipped.
- Added four targeted tests for representation masks, balanced sample
  selection, and exact prefix preservation across every architecture. The full
  repository suite passes **39/39**.

### Next step

Use the same evaluator for DiT-B/L only if a capacity-wide conditional
inpainting comparison is needed; the requested DiT-S matrix is complete.

## DiscreteDiffusionVLA vision-token audit — 2026-09-17

**Complete (read-only architecture audit).** Traced the local `dd-openvla`
vision path through preprocessing, fused feature extraction, projection,
multimodal insertion, D2F training attention, and cached D2F inference.

### Achievements

- Confirmed that the default 224px DinoSigLIP encoder takes second-to-last
  DINOv2 and SigLIP patch features and concatenates them by feature channel,
  retaining one 16x16 grid (256 tokens) per image rather than doubling the
  token count or globally pooling it.
- Confirmed that a token-wise MLP projects every fused patch into the LLM
  hidden dimension. The resulting tokens are inserted immediately after BOS;
  their labels are `IGNORE_INDEX`, and language/action queries condition on
  them through self-attention.
- Confirmed multi-camera behavior: complete 256-token grids are concatenated
  along the sequence dimension. Although the Python dataclass defaults to one
  image, the maintained D2F train/eval launchers default to two images and the
  verified LIBERO path therefore uses 512 vision tokens. The documented ALOHA
  setup uses three images and therefore 768 vision tokens.
- Confirmed that D2F inference stores the visual/text condition prefix in the
  KV cache and reuses it while action blocks are unmasked, so the vision
  encoder and condition prefix are not recomputed on every refinement step.
- Compared the reference path with the current policy: both use the same
  second-to-last DinoSigLIP feature fusion, but the current policy compresses
  each camera to 4x4 (16) tokens and adds camera/spatial embeddings. With two
  cameras it uses 32 visual tokens plus one state token, versus 512 visual
  tokens for a two-camera 224px dd-openvla input.
- Identified a mask discrepancy for follow-up: during D2F training, inserted
  vision-token queries see BOS and other vision tokens but not later prompt
  tokens; `_d2f_prediction` makes the entire vision+prompt condition prefix
  bidirectional before caching it.

### Next step

Run a controlled D2F mask-ablation only if adopting the reference decoder:
make the condition-prefix visibility identical in training and inference, then
measure action MSE and latency against the current pooled 32-token condition.

## Active policy scope: FM + joint DD — 2026-09-17

**Complete.** Following the verified capacity, stability, action-MSE, and
latency experiments, future training and evaluation now focus on flow matching
(`fm`) and joint discrete diffusion (`discrete_joint`) only.

### Achievements

- Removed `discrete_layerwise` from the public base-training, RTC-finetuning,
  open-loop, RTC, inference-RTC, latency, comparison, checkpoint-finalization,
  deployment-validation, and sweep scheduling paths.
- Reduced future raw/B-spline x base/ttRTC matrices from 12 variants to eight,
  and future DiT-S/B/L capacity sweeps from 36 checkpoints to 24.
- Kept the layerwise implementation and config ID loadable for reproducibility
  of the existing 36-checkpoint archive. No historical checkpoints, summaries,
  W&B records, or published Hugging Face artifacts were deleted or rewritten.
- Centralized the distinction as `ACTIVE_ARCHITECTURES = ("fm",
  "discrete_joint")` and `LEGACY_ARCHITECTURES = ("discrete_layerwise",)`.
- Corrected the D2F vision-token audit: maintained LIBERO launchers use two
  224px images (512 tokens), while the documented three-camera ALOHA setup uses
  768 tokens; DINO/SigLIP are fused by feature channel, not token count.

### Next step

Use only FM and joint DD in new runs and reports. Treat layerwise checkpoints
as frozen historical evidence; do not resume, fine-tune, or include them in
new aggregate comparisons.

## Full-vision DiT-S/B/L rerun — 2026-09-17 (historical log; superseded)

**Superseded on 2026-09-18 by the DiT-L retirement decision recorded at the
top of this file.** The entries below preserve the chronological experiment
log. They are not current instructions: active work now covers the 16 DiT-S/B
checkpoints only, and DiT-L must not be restarted or newly evaluated.

### Achievements so far

- Replaced teacher-assisted validation loss with deterministic from-scratch
  generation validation. The sampler receives only vision and state; target
  actions are used only after generation to compute normalized decoded action
  MSE, control MSE, and discrete token accuracy.
- Created six capacity/representation configs preserving 50,000 base updates,
  5,000 ttRTC updates, and effective batch 32 while changing each camera from
  16 pooled tokens to the full 16x16 grid (256 tokens), for 512 vision tokens.
- Prepared fresh raw and B-spline action caches with the same split and
  normalization contract.
- Built and audited the full vision cache across six GPUs: 52 episodes, 31,706
  frames, shape `[frame, 2, 256, 2176]`, float16, 70,648,076,800 bytes.
- Verified an end-to-end two-update DiT-S FM/joint smoke run. Both checkpoint
  histories contain `generation_action_mse == action_mse == loss`; even when an
  RTC parent object is supplied, validation does not condition on a prefix.
- Started four online-W&B DiT-S base runs (raw/B-spline x FM/joint DD). The
  restart-safe launcher will cache exact parent generations before each ttRTC
  stage.
- Verified the complete test suite after the validation/cache/launcher changes
  (43 tests passed) and confirmed finite gradients in every active run.
- Started the DiT-B raw FM and joint-DD chains on the two remaining GPUs while
  DiT-S occupies GPUs 0-3. Capacity-scoped launcher locks keep S/B/L outputs
  disjoint while process locks protect the shared status and checkpoint
  manifests; all six GPUs are now in use.
- Added a dedicated W&B evaluation publisher for the final 24-checkpoint table,
  ranking, pairwise effects, validation-vs-test plot, and latency frontier.
- Detected and contained a late DiT-S raw-FM stability failure: the first
  abnormal pre-clip gradient appeared at update 37,630 (>17 versus a healthy
  historical maximum near 5.5), then repeated finite BF16 backward spikes let
  Adam moments degrade the model even though ordinary norm clipping remained
  enabled. Final generation MSE regressed from about 0.009 to 0.732.
- Added an optimizer guard that rejects non-finite steps for every architecture
  and FM-only finite spikes above a pre-clip norm of 10. Joint DD retains its
  naturally larger finite norms. Added explicit unit coverage; 44 tests pass.
- Moved the unstable checkpoint, recovery file, W&B sidecar, log, and incomplete
  hash-keyed parent cache into `outputs/FULL_VISION_512/quarantine` for forensic
  retention. Nothing was deleted. DiT-S raw FM restarted cleanly; unaffected
  chains resumed from atomic recovery checkpoints.
- Rejected those operational resumes from the controlled comparison after
  auditing the legacy recovery payload: it restores model and optimizer but not
  RNG/DataLoader position. Their partial files were quarantined and their W&B
  runs tagged `excluded-from-comparison`; controlled replacements start at
  update 0.
- Calibrated the FM guard across historical DiT-S/B/L traces. Legitimate B/L
  warm-up norms reach 17--22, but every healthy run after update 1,000 stays
  below 2.4. The final rule therefore rejects FM norms above 10 only after
  update 1,000, while rejecting non-finite gradients at every update. Clean
  DiT-B and DiT-L replacements are now running with this rule.
- Empirically verified the final guard in the deterministic DiT-S raw-FM
  failure window: it rejected isolated steps 37,595, 37,834, and 37,964 with
  pre-clip norms 149.5, 13.6, and 52.4. Surrounding applied gradients stayed
  near 0.1--0.6 and generation validation remained healthy at 0.0087--0.0104
  rather than regressing toward the quarantined run's 0.732.
- Refined the final post-warm-up threshold from 10 to 3 after the guarded trace
  showed that the instability begins through smaller 4.3/6.6 precursors before
  the large spikes. This remains above the audited healthy S/B/L maximum of
  2.33. The threshold-10 calibration artifacts and W&B runs are quarantined;
  all final FM checkpoints start from update 0 with threshold 3.
- Completed the controlled DiT-S raw-FM base run at 50,000 updates. Its final
  from-scratch validation action MSE is `0.00761137` (best `0.00565124`). The
  guard rejected 5,621 anomalous finite updates; the largest applied
  post-warm-up norm was `2.99934`, and every recorded gradient remained finite.
  The exact checkpoint-hash parent cache is complete for all 52 episodes.
- Found a distinct ttRTC instability when the base FM learning rate (`3e-4`)
  was reused for conditional finetuning: divergence began within 200 updates.
  Added an explicit `rtc_learning_rate` and set all six full-vision configs to
  `3e-5`. Failed calibration artifacts and their W&B runs are retained only
  under `quarantine` and excluded from the comparison.
- Isolated the remaining DiT-S FM instability to the optimized CUDA SDP
  backward kernel. For the identical finite parent, batch, seed, and loss, the
  default kernel produced NaNs or parameter gradients near `1e20`; the math
  SDP kernel produced a fully finite global gradient norm of `6.14` in FP32
  (`10.41` under BF16). FM training now forces math SDP, while no-grad
  evaluation and deployment keep the accelerated inference kernel.
- Calibrated the corrected ttRTC path for 500 deterministic BF16 updates. All
  500 gradients were finite (maximum `98.51`), no optimizer step was rejected,
  and from-scratch validation action MSE improved from the parent's
  `0.00761137` to `0.00645953`. ttRTC uses a separate finite-spike ceiling of
  `100` from update 1; base FM retains the audited threshold `3` after update
  1,000. Guard-only and pre-math-SDP runs are quarantined and excluded from W&B
  comparisons.
- Completed the corrected raw-FM ttRTC children for DiT-S and DiT-B. DiT-S
  finished 5,000 updates with final/best generation MSE
  `0.00825399`/`0.00668315`, four guarded outliers, and no non-finite
  gradients. DiT-B finished with `0.00700098`/`0.00662437`, maximum gradient
  norm `0.41416`, and zero guarded steps.
- Completed both DiT-L FM base checkpoints at 50,000 updates. Raw and B-spline
  final generation MSE are `0.00661672` and `0.00659340`; their exact
  checkpoint-hash parent caches are currently generating concurrently on GPUs
  1 and 3.
- Completed the controlled DiT-B raw-FM base checkpoint. Its final
  from-scratch validation action MSE is `0.00680764`; the matching parent cache
  and ttRTC child will be produced by the updated launcher after the older
  in-memory launcher exits.
- Started clean DiT-S raw joint-DD, DiT-B raw joint-DD, and DiT-L raw-FM,
  raw-joint-DD, and B-spline-FM bases concurrently. All six GPUs are occupied;
  no operationally resumed checkpoint will enter the comparison.
- Completed all four DiT-S FM checkpoints and all four DiT-L FM checkpoints
  (raw/B-spline x base/ttRTC), plus both DiT-B raw-FM checkpoints. Exact
  checkpoint-hash parent caches were used for every completed ttRTC child.
- Completed the DiT-S/raw and DiT-B/raw joint-DD base checkpoints. Their full
  open-loop generation evaluations and identical oracle-prefix/suffix
  inference RTCEVAL diagnostics are complete. DiT-B/raw also has its latency
  and 128-sample delay-RTC reports; completed per-checkpoint reports remain
  restart-safe for the final 24-checkpoint aggregation.
- Kept GPU 1 as an opportunistic evaluation lane while all other devices train
  or generate exact parent caches. Every newly completed checkpoint will enter
  the same RTCEVAL immediately when a device becomes free, without waiting for
  the entire sweep.
- Reached 14/24 final checkpoints. Every one of those 14 now has all four
  checkpoint-level result records: open-loop generation, inference latency,
  128-sample delay RTC, and oracle-prefix/suffix inference RTCEVAL. This
  includes the completed DiT-S/raw joint-DD ttRTC child and DiT-B/B-spline FM
  base.
- Moved the episode-atomic DiT-B/B-spline FM parent-cache continuation from the
  congested GPU 0 lane to newly free GPU 3 after preserving 41/52 completed
  episodes. The filtered launcher will finish the cache and start its ttRTC
  child there, while GPU 0 continues the two joint-DD bases.
- Added capacity-scoped completion auditing (`--sizes`) so the original DiT-S
  objective can be proven independently of the later DiT-B/L extension. The
  current DiT-S audit verifies the 52-episode/31,706-frame 2x256-token vision
  cache and all six existing checkpoints; its only failures are the two
  genuinely unfinished B-spline joint-DD artifacts. The full test suite remains
  44/44 passing.
- Strengthened that audit to prove all four required result families for every
  checkpoint, not merely count files: full test-split open-loop generation,
  batch-1 latency, 128-sample delay RTC, and 64-sample oracle-prefix inference
  RTCEVAL. It verifies checkpoint identity, representation/training type,
  512-token inference metadata, fixed-prefix preservation, and the exact
  `{2,4,6,8,10}` prefix schedule. All 24 evaluation checks across the six ready
  DiT-S checkpoints pass; only checkpoints 7 and 8 remain absent.
- Made deployment validation inventory-scoped by capacity family. Once the two
  remaining DiT-S artifacts land, the real GPU runtime matrix can enforce an
  exact eight-checkpoint DiT-S inventory even while the independently requested
  DiT-B/L expansion is still training. This prevents unrelated later-family
  progress from weakening or delaying the original DiT-S completion gate.
- Completed the DiT-B/raw joint-DD ttRTC child (final from-scratch validation
  MSE `0.01760805`) and the DiT-B/B-spline FM ttRTC child (`0.00657846`). Both
  received inference RTCEVAL immediately on the free evaluation GPU, followed
  by full 3,149-chunk open-loop scoring, cache-aware/batch-1 latency, and
  128-sample delay RTC. The sweep now has 16/24 atomic checkpoints and exactly
  16 reports in each of the four per-checkpoint evaluation families.
- Re-ran the DiT-S audit against the remote W&B API. All six ready checkpoints
  have finished remote runs and published the required generation metric; the
  audit still reports only the two expected missing S/B-spline joint-DD files.
- Completed the final DiT-S/B-spline joint-DD base and exact-parent ttRTC child,
  bringing the original capacity family to 8/8 checkpoints. Their final
  from-scratch validation action MSE values are `0.02412884` and `0.02554141`.
- Completed all four evaluation families for both new checkpoints. DiT-S now
  has 8/8 full-test open-loop reports, batch-1 latency reports, 128-sample
  delay-RTC reports, and 64-sample inference-RTCEVAL reports. The real GPU
  deployment matrix passes all eight runtime cases.
- Passed both local and remote-W&B DiT-S completion audits with zero errors:
  exact 8-checkpoint inventory, 52 episodes/31,706 frames, 256 tokens per
  camera (512 total), generation-only validation, exact parents, finite
  gradient traces, manifests, four evaluations per checkpoint, and finished
  W&B metrics. The full test suite remains 44/44 passing.
- Added final bilingual DiT-S reports at
  `CodexDoc/reports/FULL_VISION_512_DIT_S_EN.md` and
  `CodexDoc/reports/FULL_VISION_512_DIT_S_CN.md`. The broader DiT-B/L
  replication remains active for the requested 24-checkpoint comparison.

### Historical next steps (cancelled by the 2026-09-18 scope change)

1. Complete and audit the eight-checkpoint DiT-B pipeline.
2. Analyze the final 16-checkpoint DiT-S/B comparison. Preserve partial and
   completed DiT-L artifacts solely as historical evidence.

## 2026-09-19: BSP scratch-vision U-Net v3

- Implemented the reference BSP observation stack: independent scratch
  ResNet-18+GroupNorm per camera, 32-keypoint SpatialSoftmax, direct state
  concatenation, and the `[256,512,1024]` FiLM temporal U-Net. DINOv2/SigLIP
  are absent from this family.
- Completed the original raw/B-spline × one/two-frame × FM/discrete-DD ×
  base/RTC matrix (16/16 checkpoints). FM converged normally; no variant had a
  non-finite gradient or skipped optimizer update.
- Diagnosed the DD generation gap. The original 50k runs optimized partial
  teacher corruption while deployment begins from all MASK tokens, and
  checkpoint selection used only 256 of 3,077 validation samples. Teacher loss
  kept improving while full-sequence rollout degraded.
- Ran controlled cosine, D2F, 25%-full-MASK, and 50%-full-MASK diagnostics.
  The verified correction is 50% full-MASK examples plus 50% original D2F
  block corruption, complete validation, and publication of the best
  generation-from-scratch EMA weights. Corrected base training stops at 10k
  because the full-validation curves converge well before 50k; RTC remains 5k.
- The 50%-full diagnostic test MSE values are `0.04773`/`0.04977` for raw h1/h2
  and `0.06547`/`0.06460` for B-spline h1/h2. The archived 50k B-spline D2F
  baselines were `0.08192`/`0.07738`.
- Audited DD-OpenVLA categorical/Gumbel decoding. Its changes were small and
  inconsistent across full validation, so deterministic argmax/confidence
  MaskGIT remains the default; the convolutional U-Net has no transformer KV
  cache, and reports now state that accurately.
- Verified the real two-frame deployment wrapper on GPU for continuous raw and
  B-spline RTC. Raw delay 4 fixes exactly four action rows. B-spline delay 4
  maps to two spans and fixes only their five-row cubic support union.
- Corrected implementation passed 65/65 tests and a full-size GPU
  train/save/select/reload smoke. Commit `1450899` is pushed. Four corrected DD
  base+RTC queues are running on GPUs 0--3 with online W&B; the eight frozen FM
  checkpoints are uploading to `DiscreteRTC/dRTC/NewModel/v3`.

### Next

1. Finish the four corrected DD base and RTC children and audit selected
   updates, exact parent hashes, and W&B completion.
2. Run all 16 final open-loop, latency, train/test RTC trajectory, and real
   deployment checks.
3. Publish the final bilingual reports and complete the `NewModel/v3` upload.

## 2026-09-21: DINOv2 joint discrete diffusion, h1/h2

- Added the approved eight-checkpoint matrix: raw/B-spline representation x
  h1/h2 observation history x base/ttRTC. The policy remains categorical joint
  discrete diffusion; both action representations share the same mixed
  full-MASK/block-corruption training and iterative block-unmasking sampler.
- Replaced the previous 512-token visual input with frozen DINOv2 patch tokens
  followed by independent learned 32-query resamplers for the global and hand
  cameras. Each timestep therefore contributes 64 visual tokens plus one state
  token: 65 observation tokens for h1 and 130 for h2.
- Built and verified the shared scratch-resident DINOv2 cache for all 52
  episodes and 31,706 frames. Each cached frame contains two cameras x 256 raw
  DINOv2 patch tokens x 1,024 channels; the 31 GiB cache is under
  `robot_policy/outputs/DINO_DD_JOINT_H12/cache/dinov2_patch16` and no dataset or
  model cache was moved into the home directory.
- Added h2-safe dataset and deployment handling, dynamic token/feature metadata,
  DINO-only extraction, trainable per-camera resampling, and explicit 50%
  full-MASK training so generation from scratch is represented directly during
  optimization. Existing checkpoint defaults remain backward compatible.
- Passed 74 selected repository tests, four base GPU smoke trainings, four RTC
  GPU smoke trainings, checkpoint save/reload, and static diff checks.
- Started the four 50k base runs concurrently with online W&B on GPUs 0--3.
  GPUs 4--6 remain unreserved for other agents. Each lane automatically builds
  its exact-checkpoint parent cache and starts its 5k ttRTC child after base
  completion. Early updates are finite with no skipped optimizer steps.

### Next

1. Monitor full-validation generation-from-scratch action MSE and gradient
   health through base convergence; keep the best generation checkpoint.
2. Verify each exact parent cache and complete the four ttRTC children.
3. Run train/test open-loop and prefix-inpainting RTC evaluations, latency, and
   deployment checks for all eight final checkpoints, then publish the final
   comparison and checkpoint bundle.

### 2026-09-21 runtime update

- Enforced a hard four-GPU ceiling for this pipeline: training is bound only to
  GPUs 0--3 and GPUs 4--6 remain available to other agents. The waiting
  evaluation pipeline is also configured for GPUs 0--3 and consumes no GPU
  until training completes.
- Switched to batch 64 / accumulation 1 at the same effective batch size and
  enabled complete 3,077-window validation every 500 updates. Authoritative
  continuation W&B runs are `84mf4xkm`, `e8e5u39c`, `44ah3k06`, and
  `gr8i2pn6` for raw h1, raw h2, B-spline h1, and B-spline h2 respectively.
- Latest monitoring reached beyond 12.5k updates for both raw bases and 15.5k
  for both B-spline bases. Gradients and losses remain finite with zero skipped
  optimizer updates. Scratch has approximately 11 TiB free; the current
  experiment occupies about 34.9 GiB, almost entirely the shared vision cache.
- Re-audited DD-OpenVLA block/inter-block training and decoding. The joint-DD
  implementation preserves its block-causal information boundary and completed
  block KV-cache semantics. It intentionally keeps hard-token supervision and
  50% full-MASK exposure instead of adopting D2F's teacher-distillation loss.
- Restarted the evaluation waiter after it observed a transient stale failure
  state during a deliberate batch-size resume. It now tracks the active
  launcher and will generate all four required evaluation families for each of
  the eight final checkpoints after `COMPLETE.json` is published.
- At the user's request on 2026-09-21 23:30 UTC, stopped all four training
  processes and the evaluation waiter. No GPU process remains. Atomic resume
  points are raw h1 17.5k, raw h2 15k, B-spline h1 20k, and B-spline h2 20k;
  checkpoints, optimizer states, W&B identities, caches, and logs remain in
  scratch. Runtime status is `stopped_by_user`, not an algorithmic failure.

## 2026-09-22: V5 FM state + previous-command training

- Added an explicit `data.state_dim` architecture/deployment contract. V4 and
  older checkpoints default to 7D; V5 uses the cleanup datasets' native 14D
  rows: measured robot state `[0:7]` plus previous command `[7:14]`.
- Kept the otherwise identical h2 BSP FM setup: two timestamps, global+hand
  camera at each timestamp (four images), independent scratch ResNet-18 vision
  encoders, batch/effective batch 64, raw and B-spline action variants, and no
  test split. The observation condition is now 284D instead of 270D.
- Audited and prepared all three cleanup datasets. Classify uses 45/5 episodes
  and 86,561/9,493 train/validation windows; hanging uses 55/6 and
  24,150/2,738; stacking uses 55/6 and 37,780/4,245. All RGB caches are complete
  and all action sidecars contain 14D state normalization and 7D action
  normalization.
- Added exact best-model checkpointing: validation covers the complete
  validation split every loader epoch; the exact best EMA weights remain in
  memory and a standalone deployment-loadable `base.best_epoch_NNN.pt` is
  written at epochs 10,20,...,100. This does not reduce selection to only the
  ten-epoch boundary models. No W&B model artifacts are uploaded.
- Passed 51 relevant regression tests plus a full-size 10-update GPU smoke.
  The smoke proved training, `base.best_epoch_010.pt` save/load, 14D deployment
  input, and `[B,30,7]` action output, then its temporary files were deleted.
- Deleted the interrupted V5 training checkpoints/logs/W&B runtime files at
  the user's request while preserving verified prepared/RGB caches. Restarted
  all six authoritative runs from update zero on GPUs 0--5 in the shared W&B
  project `robot-policy-bsp-unet-v5-state-command-100e`. Runtime truth is in
  `robot_policy/outputs/BSP_UNET_V5_STATUS.json`.
- Hanging-mug raw and B-spline completed all 100 epochs. Their selected
  validation-best `base.pt` files use epoch 100 (MSE 0.0135608550) and epoch 68
  (MSE 0.0130568118), respectively. At the user's request, all hanging-mug
  periodic/rolling model checkpoints were deleted after verification; only the
  two best `base.pt` models plus small W&B/manifest audit metadata remain.
- Stacking-cup raw and B-spline also completed all 100 epochs. Their final
  `base.pt` files select epoch 98 (MSE 0.0185145810) and epoch 49
  (MSE 0.0188139177), respectively. The ten periodic best checkpoints and
  rolling best-weight sidecars were deleted after verification, leaving only
  the two best `base.pt` models and small audit metadata.
- At the user's request, the remaining classify-blocks runs were stopped at
  atomic checkpoint boundaries 70 (raw) and 30 (B-spline). The exact best
  validation states through those boundaries were promoted to `base.pt`: raw
  epoch 45 with MSE 0.0283979928, and B-spline epoch 24 with MSE 0.0286925847.
  Both load independently with the V5 14D/284D contract. All other classify
  model/recovery checkpoints were deleted. The finalized V5 inventory is
  exactly six `base.pt` model files and no active GPU process.

### Next

1. Let all six jobs reach 100 epochs and verify all ten epoch-numbered best
   checkpoints plus the final `base.pt` for each run.
2. Record selected epochs and full-validation generation action MSE for all
   six models in the bilingual V5 report.
3. Remove completed optimizer recovery files while retaining only requested
   model checkpoints and local audit metadata.
# Finalized: FM V5.1 current-timestep observations (2026-09-22)

- All V5.1 training is stopped and no training process remains. Five runs
  completed 100 epochs; the user stopped the remaining classify-blocks
  B-spline run after its epoch-80 checkpoint was written.
- V5.1 uses exactly two current-timestep camera images (`observation_horizon=1`) while retaining the V5 14D measured-state + previous-command vector.
- Training stays in the existing V5 W&B project with run names prefixed by `v5.1`; no W&B artifacts are uploaded.
- V5 and V5.1 now share one policy-server executable selected by `CKPT` + `POLICY_CONFIG`; the JSON/model contract is validated before serving.
- Exactly six deployable `base.pt` models remain. All resume files, periodic
  snapshots, and rolling best-weight sidecars were removed. The stopped run's
  `base.pt` contains the global validation winner through epoch 80: selected
  update 56,784 with validation/action_mse 0.0268160413.
- Final status: [CN](reports/FM_V5_1_CURRENT_IMAGES_STATUS_CN.md) / [EN](reports/FM_V5_1_CURRENT_IMAGES_STATUS_EN.md).

## 2026-09-23: V4 / V5 / V5.1 policy organization and ttRTC preparation

- Added three canonical output indexes: `robot_policy/outputs/V4`, `V5`, and
  `V51`. Each contains the same `task/{raw,bspline}` topology for
  classify-blocks, hanging-mug, and stacking-cup.
- The 18 `base.pt` entries are symbolic links to the authoritative existing
  checkpoints. This avoids duplicating multi-GB files and preserves all legacy
  experiment paths, caches, logs, and W&B provenance.
- Audited all 18 model contracts: V4 is h2 with 7D measured state; V5 is h2
  with 14D measured state + previous command; V5.1 is h1 with the same 14D
  state. Every representation matches its `raw` or `bspline` leaf and no link
  is broken.
- Reserved `ttrtc.pt` in each leaf as the canonical output of the requested
  five-epoch ttRTC stage.

- Rebuilt the missing V4Full stacking-cup raw/B-spline action caches and the
  complete 61-episode, 45,285-frame RGB cache from the original V4 dataset.
- Completed all 18 requested five-epoch ttRTC runs. Every job used a fresh
  optimizer, fixed the simulated-delay prefix at flow time one, sampled the
  suffix flow time, and applied loss only on the mutable suffix.
- Each job ran full validation once per epoch and published the best of five as
  `ttrtc.pt`. All 18 parent hashes, model loads, manifests, and server JSON
  contracts passed the strict audit; all runs had zero skipped optimizer
  updates and no training artifacts remain.
- Results: [summary](../robot_policy/outputs/TTRTC_5E_SUMMARY.md) and
  [Chinese report](reports/FM_V4_V5_V51_TTRTC_5E_CN.md).
- Uploaded all 18 `ttrtc.pt` files to their matching directories in
  `DiscreteRTC/dRTC/NewModel/{v4Full,V5Full,V5_1Full}`. Remote file sizes and
  LFS SHA-256 values match the local checkpoints 18/18; the resulting dataset
  revision is `c47d9f5cb4394d1ce86ea4bca54f2965db44df1a`.
- Corrected direct FM ttRTC inference to preserve the training-time
  asynchronous time map at every Euler step: prefix rows use `t=1`, suffix
  rows use the current integration time, and prefix values remain hard
  clamped. This removes PiGDM/VJP while retaining multi-step flow integration;
  targeted RTC and deployment regression tests pass.

### Next

Run a matched GT-prefix RTC suffix evaluation for all 18 base/ttRTC pairs. The
generation-from-scratch validation metric used for checkpoint selection is not
an RTC suffix-quality measurement.

## 2026-09-23: V6 ActionJoint h1 project started

- Created the canonical project root at `robot_policy/output/NEW`, organized as
  `task/state_variant/{raw,bspline}` for `hanging_mug`, `stacking_cup`, and
  `classify_blocks`.
- The six jobs in each task are the Cartesian product of three observation
  states (`LastCommand_Joint` 7D, `State_Joint` 7D, and
  `State_LastCommand_Joint` 14D) and raw/B-spline action representations. All
  use the BSP-UNet flow-matching architecture, h1 observations, and exactly two
  current-timestep camera images.
- The task order is fixed to hanging mug, stacking cup, then classify blocks.
  Each task uses six independent GPUs concurrently. Base training runs for 100
  loader epochs and performs complete generation-from-scratch validation only
  at epochs 10,20,...,100; the selected base then receives five epochs of
  training-time RTC fine-tuning with a fresh optimizer.
- Added a byte-level equivalence gate before cache sharing. One RGB84 cache is
  shared per task only when all MP4s plus action/timestamp/frame alignment have
  matching SHA-256 digests across the three state variants. Action caches stay
  isolated because state normalization differs.
- Configured exactly one W&B project per task (base and ttRTC runs share that
  project), with metrics/summary logging only and no W&B model artifacts.
- Full repository tests pass (one historical optional test skipped). Runtime
  truth is recorded in `robot_policy/output/NEW/V6_STATUS.json`.
- Completed all 18 action/state caches and all three shared RGB84 caches. The
  complete frame counts are 26,661 (hanging), 41,715 (stacking), and 95,881
  (classify); the prepared project currently occupies about 7.3 GiB.
- Launched the first six authoritative hanging-mug base runs on GPUs 0--5.
  Their W&B IDs are `mkssi3dq`, `bj8nmrmm`, `hram8hi0`, `y3nwhyyw`,
  `48ys7oz6`, and `aej5vxkr`. Epoch-10 full validation covered all 2,722
  validation windows for every run and produced action MSE values in
  0.01347--0.01514; all six epoch-10 best snapshots were written successfully.
- Completed all six hanging-mug base runs and all six five-epoch ttRTC runs.
  The launcher promoted the validation winners to the canonical `base.pt` and
  `ttrtc.pt` paths and removed every resume/periodic/best-epoch training
  artifact. All 12 runs are online in the task-specific hanging-mug W&B
  project and had zero skipped optimizer updates.
- Audited and uploaded the completed hanging-mug subtree early, without
  interrupting stacking-cup training. The task-level audit verified 12/12
  loadable models with no missing metadata or unwanted training checkpoints;
  Hugging Face contains all 12 models plus 24 server/training JSON files under
  `NewModel/V6/hanging_mug`. Remote LFS SHA-256 matches 12/12 at dataset
  revision `c827f2b04799be020897b60159279d6c1825a1b2`. The final all-task upload
  will still re-audit and verify the complete V6 tree.
- Started the six stacking-cup base runs on GPUs 0--5 in
  `robot-policy-bsp-unet-v6-stacking-cup`. Full validations at epochs
  10/20/30/40/50 each covered all 4,211 validation windows. All six models
  refreshed their best at epoch 50, where generation action MSE ranges from
  0.0180482 to 0.0198387. The six training streams remain finite with zero
  optimizer skips and continue toward epoch 60.
- Completed all six stacking-cup base runs and all six five-epoch ttRTC runs.
  Base selection epochs were 70/100 (LastCommand raw/B-spline), 100/70
  (State raw/B-spline), and 60/100 (State+LastCommand raw/B-spline). ttRTC
  selection used the best of five complete validation passes. All 12 runs had
  zero optimizer skips, and all resume/periodic/best-epoch artifacts were
  removed after final publication.
- Audited and uploaded the completed stacking-cup subtree while the next task
  trained. The task-level audit verified 12/12 loadable models with complete
  server/training JSON and no unwanted checkpoints. Hugging Face contains the
  12 models plus 24 JSON files under `NewModel/V6/stacking_cup`; remote LFS
  SHA-256 matches 12/12 at dataset revision
  `b59cfae040213ed1028bc32dc954c10f1efaf4e3`.
- Completed all six classify-blocks base runs and their five-epoch ttRTC
  children. Base selection epochs are 90/70 (LastCommand raw/B-spline), 70/80
  (State raw/B-spline), and 60/20 (State+LastCommand raw/B-spline). ttRTC
  selection epochs are 5/5, 5/5, and 5/4 in the same order. Every base
  validation covered all 9,467 validation samples, every run had zero skipped
  optimizer updates, and all intermediate model/recovery checkpoints were
  removed.
- The final independent completion audit loaded all 36 models and checked the
  7D joint-action output, 7D/14D state contracts, h1 two-camera input, cubic
  uniform-left B-spline metadata, 100+5 epoch budgets, full-validation sample
  counts, task-specific online W&B projects, parent checkpoint linkage, and
  local/published SHA-256 values. The result is 36/36 with zero errors.
- Uploaded the complete V6 tree to
  `DiscreteRTC/dRTC/NewModel/V6`: 36 models plus 72 server/training JSON files.
  Remote presence and model LFS SHA-256 verification are 108/108 and 36/36,
  respectively, with no mismatch. Final dataset revision:
  `08a54d5c34cda652cf29f8451afd94ec0228c5a6`.

### Next

The V6 ActionJoint training and publication goal is complete. The next
research step is a separate deployment/open-loop/GT-prefix RTC evaluation of
the published models; it is intentionally not part of this training goal.

## 2026-09-23: V7 ActionEE h1 project started

- Added the V7 project at `robot_policy/output/NEW/V7Full`, with the topology
  `task/State_EE/{raw,bspline}` for hanging mug, stacking cup, and classify
  blocks. The six base jobs will run concurrently on GPUs 0--5, followed by
  the six corresponding ttRTC jobs.
- Audited every parquet episode before launch. All three datasets contain
  finite 8D states and 7D actions with consecutive frame indices and strictly
  increasing timestamps. The semantic contract is measured TCP position +
  quaternion-xyzw + gripper for state, and local delta translation + local
  delta rotation-vector + next gripper for action.
- V7 retains the validated V6 protocol: BSP-UNet flow matching, h1 with exactly
  the current global/hand images, raw and cubic uniform-left B-spline action
  representations, 100 base epochs with full validation every 10 epochs, and
  five ttRTC epochs with full validation every epoch.
- All 12 training runs use the single new online W&B project
  `robot-policy-bsp-unet-v7-ee`; W&B artifacts remain disabled. Final models
  will be audited, cleaned to exactly six `base.pt` plus six `ttrtc.pt`, and
  uploaded to `DiscreteRTC/dRTC/NewModel/V7Full`.
- Added a reusable V7 launcher while preserving the existing V6 behavior.
  Config resolution and the complete repository test suite pass (one optional
  historical test skipped).

### Next

1. Build the six isolated action caches and three per-task shared RGB84 caches.
2. Complete the concurrent base and ttRTC stages, retain validation winners,
   run the strict 12-model audit, and verify the Hugging Face upload hashes.

## 2026-09-23: V7 live training and classify epoch-30 handoff

- Completed all ActionEE preparation caches and launched the six base lanes in
  parallel on GPUs 0--5. Hanging-mug raw and B-spline reached epoch 100 with
  zero skipped optimizer steps; their selected full-validation action MSE is
  0.05006838 and 0.04653973, respectively. Stacking and classify remain in the
  same uninterrupted launcher process.
- Froze both classify-blocks epoch-30 validation winners for an early
  collaborator handoff without changing the final 100+5 epoch objective. The
  raw snapshot has validation action MSE 0.05845407 and SHA-256
  `354adcb058e4a0018a3f3a1b8a1978b0f693324affde9ff3e661c5d19c464b59`;
  the B-spline snapshot has 0.05668508 and SHA-256
  `64686eadc8106f0d6b0c805bee46d100e4a1435f76a4876851cf62d6d70ef661`.
- Uploaded those two explicitly non-final `base30.pt` snapshots and their
  server/training metadata to `NewModel/V7Full/classify_blocks/State_EE`.
  Remote LFS hashes match. The model upload commit is
  `ca7b530bc3c0b8b5f72c76608d15eab213bf66db`; the strict deployment-sidecar
  schema was then remotely read back and verified at revision
  `372590ac87f39885a3d6be791d3ad4ff03d12395`.

### Next

1. Let the uninterrupted stacking/classify base runs reach epoch 100.
2. Complete all six five-epoch ttRTC runs, audit the 12 final checkpoints,
   clean intermediate training artifacts, and publish/verify the final tree.

## 2026-09-24: V7 ActionEE h1 project completed

- Completed all six 100-epoch base runs and all six five-epoch ttRTC runs.
  Every base validation covered the complete split every 10 epochs; every
  ttRTC validation covered it every epoch. Validation sample counts are 2,723
  for hanging mug, 4,212 for stacking cup, and 9,469 for classify blocks.
- Promoted the validation winner in each run to the canonical `base.pt` or
  `ttrtc.pt`. The final tree contains exactly 12 model checkpoints and no
  resume, periodic, or best-epoch training checkpoint. All 12 runs had zero
  skipped optimizer updates and finite gradient diagnostics.
- Independently loaded all 12 models and verified 8D EE+gripper state, 7D
  delta-EE+gripper action, h1 with exactly global/hand current-frame images,
  and the 87,333,831-parameter BSP-UNet architecture. Raw predicts 30x7
  actions directly; B-spline predicts 18x7 cubic uniform-left control points
  and decodes them to a 30x7 trajectory.
- Verified all six ttRTC parent paths and base SHA-256 values. Also tested all
  six task/representation combinations in a simulated downloaded layout where
  the workstation's absolute prepared path is absent; the portable
  `State_EE/sidecars/{raw,bspline}` fallback resolves and validates correctly.
- All 12 runs are finished in the single W&B project
  `robot-policy-bsp-unet-v7-ee`. User/model artifacts are zero. W&B created
  one non-deletable system-managed `wandb-history` parquet for run `8gqr77ky`;
  the API explicitly rejects deletion of this backend-owned artifact.
- Published the final release to `DiscreteRTC/dRTC/NewModel/V7Full`: 12
  models, 24 server/training JSON files, and 12 portable encoder/normalization
  sidecars. Remote presence, 12 model LFS SHA-256 values, and 12 sidecar blob
  hashes all match at revision
  `9e334da2f2a8aaca002f410b18cc38ae9dc3cc18`.
- The complete repository test suite passes; one historical optional test is
  skipped. Authoritative evidence is in `V7_STATUS.json`, `V7_AUDIT.json`,
  `V7_HF_UPLOAD.json`, and `V7_INDEPENDENT_AUDIT.json` under the V7 output
  root.

### Final selection summary

| Task | Representation | Stage | Selected epoch | Validation action MSE |
|---|---|---|---:|---:|
| hanging_mug | raw | base | 100 | 0.05006838 |
| hanging_mug | raw | ttRTC | 5 | 0.04991525 |
| hanging_mug | B-spline | base | 100 | 0.04653973 |
| hanging_mug | B-spline | ttRTC | 5 | 0.04652055 |
| stacking_cup | raw | base | 100 | 0.05624402 |
| stacking_cup | raw | ttRTC | 5 | 0.05647647 |
| stacking_cup | B-spline | base | 70 | 0.05345627 |
| stacking_cup | B-spline | ttRTC | 5 | 0.05361407 |
| classify_blocks | raw | base | 90 | 0.05737160 |
| classify_blocks | raw | ttRTC | 5 | 0.05701298 |
| classify_blocks | B-spline | base | 70 | 0.05542328 |
| classify_blocks | B-spline | ttRTC | 4 | 0.05543821 |

### Next

The V7 ActionEE training, validation, portable packaging, and publication goal
is complete. LastCommand and State+LastCommand variants remain intentionally
out of scope until a separate decision is made.
