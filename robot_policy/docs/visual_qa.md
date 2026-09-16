# Visual QA record

All listed images were opened and inspected, not merely generated.

## Observation preprocessing

`outputs/visualizations/m1_vision_alignment.png` shows synchronized global and hand-camera frames, the common 224×224 geometric transform, and per-camera 4×4 fused-feature PCA maps. The global view is the expected overhead scene with arm and cups; the hand view is the expected close gripper/cup view. No camera swap, mirroring, or accidental crop was found. The source aspect ratio is intentionally mapped by a direct bicubic resize, matching the audited reference geometry. DINOv2 and SigLIP use separate checkpoint normalization after that common geometric transform.

`outputs/visualizations/m1_state_distributions.png` shows all seven raw state channels together with train-stat-normalized channels. Joint channels are centered at useful scale; the gripper remains visibly multimodal rather than being smoothed into a Gaussian-looking display.

The cached-vs-online check is retained in `outputs/visualizations/m1_cache_check.json`. Cross-process BF16 feature extraction plus float16 storage is not strict elementwise deterministic: mean absolute error is 0.01649, maximum is 0.59691, strict elementwise allclose fails, relative RMS error is 0.8546%, and cosine similarity is 0.999963. The declared storage-precision criteria (relative RMS below 1%, cosine above 0.999) pass, while the stricter failure remains visible. Frozen backbone parameters require no gradients and their sampled before/after fingerprint is unchanged. On a real two-sample batch, the trainable vision projector and state tokenizer have nonzero gradient norms 6.6465 and 0.10248; see `m1_gradient_check.json`.

## B-spline representation

`outputs/visualizations/m2_spline_reconstruction.png` uses a real test-episode chunk and plots every available action dimension without display smoothing. The continuous spline fit tracks the normalized source closely. Independent per-control quantization causes small visible oscillations, including around saturated joint/gripper values. The recorded aggregate normalized errors are fit MAE/RMSE 0.001384/0.008136 and additional quantization MAE/RMSE 0.001716/0.002247. The fit maximum 0.47694 and quantization maximum 0.08291 are explicitly retained; the largest fit outliers occur around gripper transitions. Cartesian XYZ was not plotted because the dataset provides absolute joints and gripper only and contains no calibrated forward-kinematics transform.

`outputs/visualizations/m5_delay_support.png` shows D=S from 1 through 5. The image correctly grows from four through eight committed controls, with the expected three-control overlap between adjacent cubic spans.

## Final-policy inspection

The six-model overlays were inspected on the same three diagnostic types: episode 8/frame 200 (ordinary near-stationary window), episode 34/frame 240 (largest test-set gripper change neighborhood), and episode 18/frame 714 (largest test-set joint-step neighborhood). The first gripper-transition overlay exposed a layout compression artifact in one subplot column; the plotting code was changed to constrained layout and the image was regenerated and re-opened.

- FM is visually the closest on most smooth joint channels and is the only family that approximately times the large gripper transition, though it overshoots during the transition.
- Joint discrete follows several fast-motion dimensions well and is consistently closer than layerwise discrete in the aggregate, but misses the representative gripper transition and some constant channels.
- Layerwise discrete has clear failure cases, including nearly constant wrong predictions and severe RTC deviations. These are visible in the plots and consistent with its worse held-out errors; they were not hidden by axis clipping or smoothing.
- The training-curve image shows genuine convergence for all base models and the shorter RTC continuations, with the different FM/discrete objective scales clearly separated on a log axis.
- The joint attention image has the intended observation-only top-left block and staircase action visibility with no future-block leakage. The cache-lifecycle image agrees with the implementation: observation/completed blocks cache, active block recomputes.
- The delay plot shows exact-span and shaded unseen within-span points without discontinuous odd-delay collapse. RTC improves the delayed joint/layerwise curves but not FM universally.
- The replay-switch plot contains all 120 commands with plan-switch markers. Large joint changes are smooth through switches; the small repeated gripper ripple is a real discrete quantization/replanning artifact and is deliberately retained.

Final inspected evidence is under `outputs/visualizations/{normal,gripper_transition,fast_motion,final}`. No 3-D or image-space geometric path is fabricated because calibration is unavailable.
