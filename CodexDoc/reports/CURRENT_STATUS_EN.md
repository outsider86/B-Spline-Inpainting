# Current Project Status — Raw-Action Baseline and ttRTC

**Report date:** 2026-09-16  
**Overall status:** Complete for the requested raw-action offline training, exact base reruns, evaluation, visualization, and W&B tracking scope.

## Executive summary

The modular robotics codebase now supports a second action representation that predicts the original 30-step × 7-dimensional action sequence directly, without B-spline encoding. The same three policy families were trained as both a basic model and a training-time real-time chunking (ttRTC) model, producing six independently reloadable checkpoints.

`action_mse` is now a first-class training and validation diagnostic. It is logged for every training update and every scheduled validation in W&B, plotted separately from each architecture's optimization objective, and reported alongside decoded physical action MSE during evaluation. The two quantities are deliberately kept distinct: training/validation `action_mse` is measured in normalized action space under the training corruption process, while evaluation MSE is measured after decoding in the dataset's physical action coordinates.

All three basic models were rerun with seed `7` under deterministic CUDA settings. Each rerun exactly reproduced all 2,000 training-trace points, all 20 validation points, and every final model tensor.

## Delivered scope

- Added representation-aware preprocessing, datasets, policy heads, RTC prefix handling, training, inference, evaluation, replay, latency benchmarking, and visualization.
- Raw targets contain 210 values (`30 × 7`) and do not pass through the B-spline encoder.
- The discrete models use 256 ordered bins per raw-action scalar; the flow-matching model predicts continuous normalized raw actions.
- Raw ttRTC delays cover every action delay `d = 1…10`; already committed raw actions are fixed exactly and excluded from remasking/loss.
- Added strict online W&B tracking with separate projects for basic and ttRTC runs.
- Added deterministic seeding for Python, NumPy, PyTorch, CUDA, the data loader, cuDNN, and cuBLAS.
- Added exact parent-prediction caching for joint-discrete ttRTC; cached tokens were checked against online parent inference with zero difference.

## Data and preprocessing

- Source: LeRobot v2.1 stacking-cups dataset.
- Size: 52 episodes / 31,706 frames, dual RGB cameras at 30 Hz, 7-D absolute state and action.
- Leakage-safe split: 42 train / 5 validation / 5 test episodes, split seed `20260915`.
- Raw target: 30 future actions × 7 dimensions.
- Representation fit error: exactly zero in the prepared-data audit; the tiny `2.88e-8` evaluation value is float32 roundoff after normalization/de-normalization.
- Optional 256-bin quantization error: normalized MAE `0.001661`, RMSE `0.002082` over prepared data.
- The previously generated frozen dual-camera DINOv2 + SigLIP feature cache is reused; action targets are newly prepared and independent of the B-spline representation.

## Training and action-MSE tracking

All basic models used 2,000 optimizer updates; all ttRTC models used 800 updates. Effective batch size is 128, optimizer is AdamW, learning rate is `3e-4`, precision is BF16, and the deterministic seed is `7`.

| Architecture | Stage | Final validation objective | Final normalized validation `action_mse` | Wall time | Production W&B run |
|---|---:|---:|---:|---:|---|
| Flow matching | Basic | 0.051752 | **0.019163** | 152.6 s | `ac6lvqdh` |
| Flow matching | ttRTC | 0.051271 | **0.018422** | 104.8 s | `afyozrhr` |
| Layerwise discrete | Basic | 15.460252 | **0.028369** | 152.5 s | `1ntp78q4` |
| Layerwise discrete | ttRTC | 15.063881 | **0.030174** | 249.3 s | `su659dtx` |
| Joint discrete | Basic | 14.224628 | **0.022709** | 154.1 s | `3mowd0an` |
| Joint discrete | ttRTC | 13.604142 | **0.023398** | 84.9 s | `3dripcb6` |

The discrete optimization objective is `CE + E|K-y|`; its numerical scale is therefore not comparable with flow-matching loss or with action MSE. The dedicated `action_mse` curve is the appropriate cross-architecture training diagnostic. Validation `action_mse` improves for flow matching but is slightly worse after ttRTC for the two discrete policies. This does not contradict the rollout evaluation below: the validation diagnostic is calculated at stochastic training corruptions, whereas open-loop evaluation runs each complete sampler.

### W&B projects

- Basic production and reproducibility runs: [raw-actions-basic](https://wandb.ai/401910710-university-of-california-berkeley/raw-actions-basic)
- ttRTC production runs: [raw-actions-ttRTC](https://wandb.ai/401910710-university-of-california-berkeley/raw-actions-ttRTC)

All six production runs were verified as `finished`, and each contains both `train/action_mse` and `validation/action_mse`. Three exact reproduction runs are also recorded in the basic project. Earlier interrupted/superseded exploratory runs remain in W&B history; the production run IDs in the table and checkpoint manifest are authoritative.

## Exact reproducibility

| Architecture | Training points compared | Validation points compared | Maximum trace delta | Final tensors equal | Reproducible |
|---|---:|---:|---:|---:|---:|
| Flow matching | 2,000 | 20 | 0 | Yes | Yes |
| Layerwise discrete | 2,000 | 20 | 0 | Yes | Yes |
| Joint discrete | 2,000 | 20 | 0 | Yes | Yes |

The comparison also verifies the architecture, stage, update count, code snapshot, configuration excluding tracking metadata, seed, and deterministic-mode settings. Detailed reports and rerun checkpoints are under `robot_policy/outputs/raw_actions/reproducibility/`.

## Held-out open-loop evaluation

All six checkpoints were evaluated on the same 3,149 test windows. The primary comparison below is decoded physical action MSE.

| Architecture | Stage | Physical action MSE ↓ | MAE ↓ | RMSE ↓ | Token accuracy |
|---|---:|---:|---:|---:|---:|
| Flow matching | Basic | **0.013315** | 0.053154 | 0.115391 | N/A |
| Flow matching | ttRTC | **0.012975** | 0.051956 | 0.113910 | N/A |
| Layerwise discrete | Basic | **0.058202** | 0.124522 | 0.241251 | 0.1868 |
| Layerwise discrete | ttRTC | **0.044474** | 0.112396 | 0.210888 | 0.1739 |
| Joint discrete | Basic | **0.044148** | 0.110067 | 0.210115 | 0.1733 |
| Joint discrete | ttRTC | **0.033632** | 0.093647 | 0.183389 | 0.1710 |

Relative to its matching basic checkpoint, ttRTC lowers held-out physical action MSE by 2.6% for flow matching, 23.6% for layerwise discrete, and 23.8% for joint discrete. Flow matching remains the strongest model in absolute MSE. Token accuracy is not the primary quality metric because ordered bins can move to a neighboring token while reducing continuous action distance.

## Delay robustness and replay

Delay sweeps use the same 128 held-out samples at every raw delay `d = 0…10`. All delays `1…10` were represented during raw ttRTC training. Committed-prefix preservation error is exactly zero for all six checkpoints and every nonzero delay.

| Architecture | Stage | MSE at d=0 | MSE at d=10 | Mean MSE over d=0…10 |
|---|---:|---:|---:|---:|
| Flow matching | Basic | 0.006373 | 0.007995 | **0.007178** |
| Flow matching | ttRTC | 0.007436 | 0.007981 | **0.007949** |
| Layerwise discrete | Basic | 0.035404 | 0.038929 | **0.038985** |
| Layerwise discrete | ttRTC | 0.029111 | 0.029341 | **0.027454** |
| Joint discrete | Basic | 0.055547 | 0.064702 | **0.058423** |
| Joint discrete | ttRTC | 0.026105 | 0.040436 | **0.030745** |

ttRTC reduces mean delayed MSE by 29.6% for layerwise discrete and 47.4% for joint discrete. Flow-matching ttRTC is 10.7% worse on this smaller delay subset despite its 2.6% improvement over the full open-loop test set. This limitation is retained in the plots and report rather than averaged away.

Each ttRTC replay issued 120/120 commands, performed 40 plan switches at `d=3`, and had zero empty-queue events.

## Inference latency

Latency was measured at batch size 1 on an RTX PRO 6000 Blackwell after 10 warmups and across 100 CUDA-synchronized trials.

| Checkpoint | Default sampler | Sampling p50 |
|---|---:|---:|
| Flow matching basic | 12 steps | 14.91 ms |
| Flow matching ttRTC | 12 steps | 16.56 ms |
| Layerwise basic | 8 rounds | 12.55 ms |
| Layerwise ttRTC | 8 rounds | 12.82 ms |
| Joint basic, cached | 8 rounds | 173.72 ms |
| Joint ttRTC, cached | 8 rounds | 173.51 ms |

Online dual-camera vision adds about 10.7–11.3 ms. The raw joint model processes 210 action tokens, so it is materially slower than the earlier 126-token B-spline joint model. At 8 rounds, joint K/V caching improves p50 from 177.70 to 173.72 ms for basic and from 177.87 to 173.51 ms for ttRTC.

## Verification and artifact status

- Raw-action project tests: **13 passed**.
- Independent CPU checkpoint reload: **6/6 passed**.
- Exact basic-model reproducibility: **3/3 passed**.
- Open-loop evaluations: **6/6 complete** on identical samples.
- Delay reports and latency reports: **6/6 each complete**.
- Visual QA: action-MSE curves, objective curves, six individual trajectory plots, combined overlay, raw quantization, delay support, delay curves, latency, cache lifecycle, attention mask, and RTC replay were generated and inspected.

Primary artifacts:

- Checkpoints and manifest: `robot_policy/outputs/raw_actions/checkpoints/`
- Evaluation: `robot_policy/outputs/raw_actions/evaluation/`
- Exact reruns: `robot_policy/outputs/raw_actions/reproducibility/`
- Visualizations: `robot_policy/outputs/raw_actions/visualizations/`
- Latency and replay: `robot_policy/outputs/raw_actions/{latency,replay}/`
- Configuration: `robot_policy/configs/raw_actions.yaml`

## Known limitations

- These are open-loop dataset and replay measurements, not closed-loop robot success rates or a hardware-safety validation.
- Physical MSE aggregates six joint dimensions and one gripper dimension in their dataset units; calibrated Cartesian pose metrics cannot be produced without robot kinematics and calibration.
- A single seed was rerun exactly. This proves implementation-level repeatability for seed `7`, not statistical robustness across different initialization seeds.
- The representative trajectory plots intentionally expose difficult/failing cases; good aggregate action MSE does not guarantee every trajectory is satisfactory.

## Recommended next steps

1. Run at least three distinct deterministic seeds per method and report mean, standard deviation, and confidence intervals for physical action MSE.
2. Add a paired per-window significance analysis for basic versus ttRTC, using the already fixed test windows.
3. Break action MSE into joint-only, gripper-only, early-horizon, and late-horizon values so model improvements can be localized.
4. With explicit simulator or hardware authorization, add calibrated closed-loop task success, safety bounds, and recovery evaluation.

