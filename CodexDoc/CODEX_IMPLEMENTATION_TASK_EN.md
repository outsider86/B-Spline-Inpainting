# Implementation Task: Modular B-spline Robot Policies, Training-time RTC, and Efficient Inference

## 0. Objective and execution principles

You are the Codex agent responsible for implementing and validating this project. In the target Linux environment, build a concise, modular, reproducible robot policy repository and complete actual training and evaluation.

The core action representation already exists in `BSplineEncoder`. Understand and reuse it, connecting existing vision encoders, action models, RTC training, and inference implementations. Do not reinvent the B-spline mathematics, substitute a generic spline fitter for the existing encoder, or expand this into a large VLM project.

**The core workflow is: extract existing code → connect modules → provide functional scripts and interfaces. Write as little new code as possible, especially for core algorithms.** Before implementing each concrete feature, consult this specification and the relevant reference documentation, then use `rg` to locate implementations, call chains, configurations, and tests. Understand them before extracting or porting them. This is an ongoing workflow, not a one-time M0 audit followed by rewriting from memory.

Prefer direct reuse, then thin adapters, then extraction of necessary modules. Add a minimal new implementation only when the required functionality is genuinely missing, and document the sources searched, the gap, and the reason. Preserve algorithmic semantics when porting and validate against the reference. Do not sacrifice correctness checks or necessary interfaces merely to reduce code size.

Priorities:

1. Correct action representations, temporal alignment, RTC, and caching.
2. Three comparable, compact policy architectures.
3. Efficient, concise reuse with clear module boundaries.
4. Actual training and validation of six checkpoints.
5. Visualizations that are inspected, with reproducible evidence.

You may and are encouraged to use multiple agents to investigate independent reference codebases in parallel. Assign file ownership before implementation. The lead agent owns interface consistency, integration, and final acceptance. Continue useful independent work while resolving genuine uncertainties. Do not repeatedly request authorization for downloads, implementation, training, or evaluation already authorized here.

Do not present random initialization, a brief smoke run, copied checkpoints, or unverified weights as final training results. Do not describe open-loop metrics as closed-loop robot success rates.

## 1. Target paths and dataset

Project root:

```text
/scratch/wangpc/B-Spline-Inpainting
```

Required source code to read first:

```text
/scratch/wangpc/B-Spline-Inpainting/BSplineEncoder

/scratch/wangpc/B-Spline-Inpainting/RefCode/starVLA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py
/scratch/wangpc/B-Spline-Inpainting/RefCode/starVLA/starVLA/model/modules/action_model/LayerwiseDiscreteDiffusion_ActionHeader.py

/scratch/wangpc/B-Spline-Inpainting/RefCode/real-time-chunking-kinetix
/scratch/wangpc/B-Spline-Inpainting/RefCode/dd-openvla
```

Final training dataset:

```text
/scratch/wangpc/B-Spline-Inpainting/Data/stacking_cups_action_30hz
```

Vision reference: the SigLIP + DINOv2 ViT frontend from DiscreteDiffusionVLA. If unavailable locally, consult the official repository:

```text
https://github.com/Liang-ZX/DiscreteDiffusionVLA
```

These paths were supplied by the user; their existence has not been verified by this specification. First inspect the directories, applicable `AGENTS.md` files, Git status, hardware, and runtime environment. Report missing paths accurately. Do not claim to have inspected unavailable code. Preserve original data, reference repositories, existing artifacts, and uncommitted user changes.

Choose a separate implementation subdirectory under the project root, such as `robot_policy/`. Follow existing conventions if a suitable implementation directory already exists. Do not recreate, overwrite, or move the entire project root.

## 2. Confirmed decisions

These decisions are settled. Do not ask the user to reconfirm them. This section takes precedence over any broader wording elsewhere in this document.

### 2.1 Three architectures

| ID | Architecture | Observation interface |
|---|---|---|
| `fm` | Compact StarVLA Layerwise FM / DiT policy | Observations act as conditioning during denoising; follow StarVLA architecture and attention |
| `discrete_layerwise` | Compact StarVLA Layerwise Discrete Diffusion policy | Observations act as conditioning during denoising; follow StarVLA architecture and attention |
| `discrete_joint` | Joint-sequence masked discrete diffusion DiT | Concatenate observation tokens and masked action tokens; the entire compact DiT processes the sequence and progressively unmasks a subset of action positions |

Only the third architecture uses the joint token sequence. Its observation tokens do not use the separate conditioning interface of the first two architectures and are not action prediction targets.

Only `discrete_joint` requires the specialized attention, block diffusion, and KV cache design from `/scratch/wangpc/B-Spline-Inpainting/RefCode/dd-openvla`. The first two architectures follow StarVLA without additional specialized attention or KV caching. Cache correctness requirements below apply to the third architecture.

Reuse the two specified StarVLA action headers for the first two policies. For the third, reuse StarVLA DiT blocks and adapt the token/attention interface using the discrete reference. Do not count three sampling step counts as three architectures.

Write `docs/architecture_decision.md` with these decisions, source symbols, architectural differences, and the parameter matching plan. If source interfaces differ from the target design, document the necessary adaptation based on actual code. Do not infer implementation details from filenames or silently change the confirmed architectures.

### 2.2 Action and vision defaults

- Action chunk length: **30 raw action samples**.
- Spline type: **uniform left-clamped B-spline**.
- Knot span length: **2**.
- Number of action bins: **256**.
- FM predicts the **continuous representation before discretization**.
- Verify spline degree, control-point count, and fixed/predicted fields in `BSplineEncoder`; preserve the existing mathematical design.
- Use the current observation only: `observation_horizon=1`.
- Share vision weights across both cameras.
- Freeze SigLIP and DINOv2 globally; train the projector and state tokenizer.
- Caching frozen vision features is allowed.

The uniform left-clamped workflow **does not predict knots or knot spans**. They are fixed encoding configuration, not policy outputs, vocabulary targets, or loss terms. Whenever a concrete question arises, inspect the relevant workflow in `/scratch/wangpc/B-Spline-Inpainting/BSplineEncoder`. Verify the actual units and temporal mapping of the knot span parameter rather than guessing from its name.

### 2.3 RTC delay notation and setup

- B-spline uses uppercase `D` and `S`, referred to by the user as “DS,” measured in **knot spans**, with range **1–5**.
- Raw actions use lowercase `d` and `s`, referred to as “ds,” measured in **individual raw actions**, with range **1–10**.
- Always set **`D = S`** and **`d = s`**.
- `D/d` is inference delay. `S/s` is the amount advanced between planning iterations, i.e. the replanning interval, in the corresponding units.
- Do not conflate span units with raw action units. Apply the four-control-point support and overlap requirements in Section 8.

For FM training-time RTC, inspect the actual setup, delay sampling, and training procedure in:

```text
/scratch/wangpc/B-Spline-Inpainting/RefCode/real-time-chunking-kinetix
```

For training-time RTC in both discrete policies, inspect:

```text
/scratch/wangpc/B-Spline-Inpainting/RefCode/dd-openvla
```

Adapt those implementations within the specified ranges and equality constraints. Record the sampling distribution and configuration sources; do not independently assume uniform sampling. Where a reference configuration conflicts with explicit user requirements, preserve the user requirements and document the adaptation.

Use zero delay for correctness checks, without silently adding it to the specified training range. The raw action range defines units and experimental conventions; it does not add architectures or increase the six-checkpoint deliverable.

### 2.4 GPU usage, training fairness, and evaluation scope

Use as many **available, unoccupied GPUs** as possible. Inspect current usage before launching; do not use GPUs occupied by other processes or terminate other jobs. The user expects approximately four GPUs to be occupied and about six available, with more released later. Treat these as estimates and use actual measurements.

Additional GPUs may be used for subsequent runs or checkpoint resumes by increasing world size. Live elastic resizing during a run is not required.

**Keep the training setup consistent across variants.** Within each stage, hold the effective global batch size, data split, update budget, and common optimizer/schedule settings consistent. Use gradient accumulation or equivalent measures to preserve effective settings when GPU count changes. Explicitly document algorithmically necessary differences. Compare base runs consistently with each other, and RTC runs consistently with each other.

Final scope: six checkpoints, open-loop evaluation, dataset RTC replay, inference efficiency, and trajectory visualization. Simulator or physical robot closed-loop experiments are not required.

### 2.5 Discrete diffusion supervision: CE + expected token distance

Both discrete policies use the following objective in base training and RTC fine-tuning:

```text
loss = loss_ce + lambda_l1 * loss_l1
lambda_l1 = 1.0
```

The confirmed L1 definition is the **expected discrete absolute distance from the entire predicted action-vocabulary distribution to the ground-truth token position**, at each valid supervised position. Fixed knots/spans are excluded because this encoding workflow does not predict them.

For valid action bins `k = 0, ..., 255`, ground-truth bin `y_i`, and predicted probability `p_i(k)`:

```text
L1_i = sum_{k=0}^{255} p_i(k) * abs(k - y_i)
CE_i = -log p_i(y_i)
loss = mean_valid(CE_i) + 1.0 * mean_valid(L1_i)
```

This is `E[|K-y|]`, not `|E[K]-y|`, an argmax-token distance, or MAE over bin centers or decoded trajectories. Use raw bin-index distance; do not divide by 255 by default. The GT bin contributes zero distance, while every other bin contributes according to both its probability and distance. Keep the probability computation differentiable back to the logits.

Use the ordered bin positions defined by the encoder, not arbitrary global token IDs with vocabulary offsets. MASK/PAD and other special symbols are not numeric bins. If the reference output head includes them, adapt it to a valid probability distribution over the action vocabulary, using the same action distribution for CE and L1. Do not discard special-token probability without handling normalization.

Average CE and L1 separately over the same valid supervised positions, then add them at a 1:1 weight. Follow the relevant reference's masked/RTC supervision rules; exclude PAD, fixed boundaries, and non-predicted fields. Log both loss terms and gradient scales. Do not silently change the default weight because the scales differ. Report decoded trajectory errors separately without replacing the token-distance loss.

Acceptance checks:

- Distance is zero when probability is concentrated at the GT bin.
- Moving equal error probability to a more distant bin increases the loss.
- A symmetric distribution around the GT still has positive distance loss even when its mean equals the GT.
- Verify gradients, padding masks, vocabulary offsets, and integration into both discrete policies.

## 3. M0: Source audit and minimal interface design

### 3.1 Required inspection

Locate actual classes, functions, configurations, and call chains in each reference. Reading only READMEs is insufficient.

- `BSplineEncoder`: LeRobot reader, action chunk construction, left-clamped encoding/decoding, normalization, bin tokenization, boundary support, and control-point/span mapping.
- StarVLA: DiT blocks, time embeddings, layerwise conditioning, action input/output projections, FM objective, discrete mask schedule, samplers, and training configurations.
- `real-time-chunking-kinetix`: delay simulation, committed action prefixes, training objectives, asynchronous execution, and guidance/inpainting details.
- `dd-openvla`: discrete training-time RTC, block diffusion, remasking, attention masks, and KV cache creation/update/invalidation rules.
- DiscreteDiffusionVLA: exact dual-vision checkpoints, preprocessing, feature extraction layers, fusion dimensions, and handling of CLS/register tokens.

Independent agents may audit the encoder, StarVLA, RTC, and discrete inference while the lead agent inspects data and the environment. Avoid uncoordinated edits to the same core module.

### 3.2 Required outputs

- `docs/source_map.md`: module → source file/symbol/commit → reuse method → necessary changes.
- `docs/data_contract.md`: shapes, dtypes, units, time axes, and mask semantics.
- `docs/environment.md`: GPU models/count/memory, driver, CUDA, PyTorch, dependencies, and dataset version.
- `docs/architecture_decision.md`: the three architectures and parameter matching plan.
- Minimal repository structure, runnable configurations, and smoke-test commands.

Preserve licenses and attribution when extracting third-party code, and pin reference commits. Copy only necessary modules or create stable thin adapters. Avoid dependencies on the caller's working directory, scattered absolute paths, or copying entire VLM stacks.

## 4. Encoding Part 1: Unified observation tokenization

### 4.1 Inputs

- Two RGB camera streams.
- Current robot state.
- No language input, language tokenizer, or LLM/VLM backbone.
- Current frame only for the initial implementation: `observation_horizon=1`. Camera count and history length may remain explicit configuration fields.

Confirm both camera keys, state keys, dimensions, and physical meanings from dataset metadata. Do not guess fields. Verify timestamps and anomalous intervals despite the dataset's stated 30 Hz rate.

### 4.2 Vision frontend

Reuse the SigLIP + DINOv2 ViT fusion design from DiscreteDiffusionVLA:

1. Feed each RGB view into both pretrained vision encoders.
2. Extract dense patch features.
3. Fuse features according to the reference, then map them to the policy hidden dimension using a small projector.
4. Share the same vision weights across cameras; avoid unnecessary independent copies.
5. Freeze both backbones throughout base training and RTC fine-tuning: `requires_grad=False`, with evaluation mode maintained.
6. Keep the projector and small state tokenizer trainable.

Downloading the corresponding public weights is authorized. Record sources, revisions, extraction layers, and preprocessing. If access is restricted, report the specific limitation without silently substituting a model.

Use each checkpoint's own pixel normalization. Geometric preprocessing of the same camera image for the two encoders must remain spatially aligned. Equal token counts alone do not establish patch correspondence.

Frozen fused backbone features may be cached **before the trainable projector**. Cache keys must include episode/frame, camera, weight revision, preprocessing, and extraction layer. If using random image augmentation, explain how caching affects it; fixed cached features do not provide fresh pixel augmentation each epoch.

### 4.3 State tokenizer and unified output

Use a small MLP/linear tokenizer on normalized state to produce one or a few state tokens. Compute normalization statistics only on the training split. Define stable camera, modality, and spatial position information.

Suggested interface; adapt names to existing code:

```python
ObservationTokens(
    tokens=...,       # [B, N_obs_tokens, D_model]
    valid_mask=...,   # [B, N_obs_tokens]
    metadata=...,     # Camera/token layout, preprocessing version, etc.
)
```

### M1 acceptance and visualization

- Produce unified RGB + state tokens from a real dataset batch, with explicit shape, dtype, and device.
- Visualize original and preprocessed views, patch grids, and feature PCA/similarity maps; inspect alignment, mirroring, cropping, and camera swaps.
- Plot raw/normalized state distributions and tokenizer output scales.
- Verify unchanged backbone weights and absent gradients, with valid projector/state-tokenizer gradients.
- Compare cached and online extraction numerically using tolerances appropriate to storage precision.
- Open and inspect generated images. Report findings and fixes; file creation alone is not visual acceptance.
- Deliver real-batch sample artifacts, test commands, and results as an independent milestone.

## 5. Encoding Part 2: Reuse the core BSplineEncoder

Read `/scratch/wangpc/B-Spline-Inpainting/BSplineEncoder` thoroughly. The user's existing mathematical design is authoritative for action representation. Prefer its public interfaces. Any necessary bug fix must be minimal and supported by evidence.

Requirements:

- Construct each action chunk from a LeRobot episode aligned to its observation time.
- Use the existing uniform left-clamped B-spline encoding.
- Reuse existing bin tokenization for the discrete action representation.
- Use the corresponding continuous representation for FM and discrete representation for discrete diffusion.
- Identify predicted encoder outputs versus quantities fixed by state, boundary conditions, or metadata.
- Preserve all metadata needed for decoding, including phase, duration, and fixed knot information. Do not add fixed boundary quantities to generation targets.
- Never cross episode boundaries. Handle short episodes, terminal padding, invalid targets, and fitting failures explicitly.
- Verify and document the collection convention for `action[t]` versus `observation[t]`; do not assume alignment semantics.

Both representations should be available from the same sample:

```python
PolicyBatch(
    observations=...,
    continuous_action_target=...,
    discrete_action_target=...,
    action_valid_mask=...,
    boundary_metadata=...,
    episode_id=...,
    frame_index=...,
    timestamps=...,
)
```

This is an interface sketch, not a requirement to rewrite the encoder around this class.

### M2 acceptance and visualization

- Show original trajectory → encoding → decoded reconstruction.
- Separately report encoding error, additional quantization error, maximum errors, and quantiles.
- Plot each action dimension over time, end-effector XYZ where available, and boundary/span annotations.
- Verify left-clamped initial conditions and properties guaranteed by the encoder.
- Include episode endings, short chunks, stationary behavior, fast motion, and gripper changes.
- Inspect images; do not conceal errors with display smoothing.
- Split by episode before creating overlapping train/validation/test windows to prevent leakage.

## 6. Policy architecture: Separate policies and comparable DiTs

### 6.1 General requirements

- Implement the three confirmed architectures in a separate `policies/` module.
- Reuse StarVLA compact DiT blocks, time embeddings, objectives, and schedules.
- Do not introduce a large OFT/VLM/LLM, load language-model weights, or require language input.
- For `discrete_joint`, process observation tokens and masked action tokens in one sequence; do not replace this with observation-only cross-attention at each layer.
- Read action outputs from action positions only. Do not apply action classification loss to observation positions.
- Projectors should provide dimension/modality adaptation, not become another large model.

### 6.2 Joint-token interface

For `discrete_joint` only:

```text
[camera-0 tokens | camera-1 tokens | state tokens | corrupted action tokens]
                               |
                       small DiT blocks
                               |
                  action-position output head
```

Action inputs use discrete embeddings and a MASK token. The first two policies retain StarVLA input and conditioning interfaces. FM uses continuous representations and the correct FM objective. “Discrete diffusion” here means masked discrete diffusion, not continuous DDPM.

Specify and visualize the third policy's attention mask, including:

- Observation-to-observation visibility.
- Observation-to-action visibility.
- Action-to-observation visibility.
- Full/block action-to-action visibility.
- Padding, fixed-prefix, and current-generation-block semantics.

**A shared sequence does not imply fully bidirectional attention at every position.** If observation hidden states depend on changing action tokens, their KV generally cannot be reused directly across denoising rounds. Determine attention topology, training masks, and caching together from the reference. Do not silently alter the trained visibility pattern to enable caching.

### 6.3 Parameter counts and fairness

Keep hidden width, depth, vision frontend, state tokenizer, and output representation budgets as similar as practical. Separately report:

- Frozen vision parameters.
- Trainable projector/state parameters.
- Action backbone parameters.
- Input embedding/output head parameters.
- Total trainable and total parameters.

Do not hide action-backbone differences behind large frozen vision parameter counts. Aim for approximately 10% or less difference in total trainable parameters. If vocabulary sizes or other necessary differences make this unreasonable, explain and provide the closest suitable configurations. Similar parameter counts do not imply similar FLOPs, sequence lengths, or latency; report those as well.

### M3 acceptance

- Working forward, loss, backward, and sampling for all three architectures.
- Reasonable overfitting on a tiny subset, with predictions decodable into trajectories.
- Correct classification ranges, MASK/PAD exclusion, continuous target scales, normalization, and padding loss handling.
- Meaningful checks of masked-token perturbations, changed observations/conditions, and attention visibility.
- Architecture diagrams, parameter tables, token layouts/attention masks, loss curves, and reconstructed examples.
- Configuration-based policy selection rather than three duplicated training loops.
- Verified CE + expected token-distance supervision in both discrete policies, as defined in Section 2.5.

## 7. Base training: Three base checkpoints

Use the specified stacking-cups dataset. Fix episode splits, seeds, preprocessing, encoder configuration, and comparable training budgets. Reference StarVLA schedules after inspecting their model scale, optimizer, and effective-batch assumptions. Do not blindly copy hyperparameters designed for a 7B model.

Run a throughput/memory pilot, then select batch size, gradient accumulation, and update counts using available hardware, subject to Section 2.4. Record wall time, GPU-hours, update counts, samples seen, peak memory, and checkpoint selection metrics.

FM inference defaults to **12 Euler steps**, with **5/8/12-step** evaluation. Integration steps are independent of action representation length and control frequency. Follow the audited FM objective for training-time sampling.

For discrete diffusion, use the appropriate reference mask schedule and sampler. Record rounds, remasking rules, and block size where applicable. Compare actual network-call budgets and latency curves with FM, not just nominal step counts. Specialized block diffusion applies to `discrete_joint`.

### M4 acceptance

- Three genuinely trained base checkpoints.
- Full configuration, source commit, encoder/bin version, split, normalization statistics, metrics, and resume commands for each.
- Independent inference after loading from disk, without objects left over from the training process.
- No final-test-set use for checkpoint selection.

## 8. Training-time RTC fine-tuning

### 8.1 Shared requirements

Initialize each RTC run from its corresponding base checkpoint and fine-tune over simulated inference delays. Produce one RTC checkpoint per architecture covering the declared delay distribution; do not present a single-delay model as general RTC.

Reuse the reference mechanisms for previous-plan generation, committed/fixed prefixes, delay sampling, conditioning, and losses. Audit and report whether training prefixes come from ground truth, noisy approximations, or previous policy predictions. Do not create an unrealistically privileged inference path using information unavailable at deployment.

Record delays in milliseconds, 30 Hz raw action steps, spline parameter/span units, and token/control-point indices. These units are not interchangeable. Apply the confirmed ranges and `D=S`, `d=s` constraints in Section 2.3.

### 8.2 Critical B-spline requirement: At least four control points

**Simulating one span of inference delay requires support from at least four control points.** Compute the relevant support and left-clamped boundary behavior from the existing encoder's cubic B-spline implementation.

Do not equate a one-frame/one-token delay with freezing one control point. Also do not interpret four-point support as four new, disjoint control points for every additional delayed span: adjacent spans generally share support.

Build and test an explicit mapping:

```text
delay_ms
   -> execution time interval
   -> spline phase / affected spans
   -> required control-point support and boundary conditions
   -> committed / frozen / token masks used by the model
```

Verify at least:

- Zero delay, within-span delay, exact span boundaries, and multiple spans.
- The left-clamped start, window end, and short remaining intervals.
- Four-control-point support and overlap across adjacent spans.
- Which parameters must remain fixed to preserve committed trajectory intervals, beyond merely counting prefix tokens.
- Consistent fixed knots/spans during RTC; do not add knot prediction branches.
- An explicit policy for delays beyond prediction coverage; do not silently clamp them into a different delay.

If one control point does not correspond to one token, construct masks through the encoder's actual mapping.

### 8.3 FM RTC

Extract training-time delay simulation and the corresponding objective from `real-time-chunking-kinetix`. Adapt them to the continuous encoded representation, verifying preservation of the decoded committed trajectory.

### 8.4 Discrete RTC

Extract discrete RTC training masks, fixed-prefix handling, remasking, losses, and applicable block handling from `dd-openvla`. Apply the training-time mechanism to both discrete policies while retaining their confirmed architecture differences. Distinguish:

- Committed tokens that cannot change.
- Tokens currently being generated or recovered.
- Tokens outside the current block, where block generation applies.
- Padding and special boundary fields.

Remasking must not modify committed tokens. Training mask/block distributions must be compatible with the corresponding deployment path. Specialized block diffusion and KV caching remain exclusive to `discrete_joint`.

### M5 acceptance

- Three RTC fine-tuned checkpoints with explicit lineage to their bases.
- Performance curves by delay and evaluation of unseen delays.
- Visualizations of time axes, spans, control-point support, committed prefixes, and generated suffixes.
- Base/RTC trajectory overlays for identical episodes and delays.
- Checks of zero-delay degradation, changes to executed intervals, and position/velocity plus applicable acceleration errors at plan switches.
- Numerical invariant tests; visualizations supplement rather than replace them.

## 9. Inference-time RTC, block diffusion, and KV cache

### 9.1 Modular robot control interface

Provide three independently testable components:

```text
ObservationProvider -> PolicyRunner -> ActionExecutor
                            |
                    RTC state / action buffer
```

- Include a dataset replay/mock backend that runs without a robot.
- The executor consumes decoded time-indexed trajectories, not spline parameters or discrete tokens as robot commands.
- Timestamp camera data, state, and plans; distinguish observation age from inference delay.
- Decouple inference from high-frequency execution; maintain remaining actions, committed intervals, and plan-switch rules.
- Explicitly handle timeouts, empty queues, stale plans, episode resets, and interruptions.
- Do not connect to or actuate physical hardware without explicit hardware authorization and configuration. Replay satisfies the current scope.

### 9.2 FM

Implement the reference inference-time RTC mechanism. Support configurable 5/8/12 Euler steps and measure actual wall-clock delay. Reuse visual encoding within an observation update; place reusable conditioning computation outside the sampling loop where valid.

### 9.3 Discrete diffusion

For **`discrete_joint` only**, implement block diffusion and KV caching from `dd-openvla`, with an explicit cache lifecycle. `discrete_layerwise` retains the StarVLA inference architecture and does not require specialized caching.

Specify:

- Which prefixes/blocks are cacheable under the same observation.
- Which layers/positions become invalid after block commitment or remasking.
- How observation changes, state changes, and episode resets clear caches.
- Whether the trained attention topology permits each reuse.

**Caching must preserve model semantics.** In a fully bidirectional sequence, an unchanged token ID can still have a hidden state dependent on changing tokens. An unchanged ID alone is insufficient evidence for KV reuse.

Maintain a cache-disabled reference path. Fix random draws and sampling settings to compare logits, output tokens/trajectories, and latency with and without caching. Document appropriate numerical tolerances and explain differences. Label approximate caching explicitly rather than claiming exact equivalence.

### M6 acceptance

- Independent switches and tests for FM RTC, discrete RTC, and the third policy's block diffusion/KV cache.
- Cache correctness comparisons covering remasking, block transitions, observation changes, and resets.
- Inference/execution timelines, cache hit/invalidation visualizations, and trajectory switch plots.
- Measured throughput and latency, not speed inferred solely from theoretical step counts.

## 10. Final training and evaluation deliverables: Six checkpoints

| Architecture | Base | Training-time RTC fine-tuned |
|---|---|---|
| `fm` | Checkpoint 1 | Checkpoint 2 |
| `discrete_layerwise` | Checkpoint 3 | Checkpoint 4 |
| `discrete_joint` | Checkpoint 5 | Checkpoint 6 |

### 10.1 Open-loop evaluation

Evaluate all six checkpoints on identical held-out episodes, including:

- Encoded-space errors.
- Decoded per-dimension MAE/RMSE, maximum errors, and quantiles.
- Discrete token accuracy and quantization-error reference baselines where applicable.
- Boundary and committed-trajectory preservation errors.
- Position, rotation, and gripper metrics in meaningful physical units where available; avoid misleading aggregation across incompatible units.
- Trajectory continuity and velocity/acceleration anomaly statistics.
- RTC metrics across delays.

Explain that open-loop evaluation uses fixed recorded observations and cannot measure environmental feedback caused by policy actions. RTC replay is not a closed-loop robot success-rate evaluation.

### 10.2 Inference efficiency

Time image preprocessing, vision encoding, projector/state processing, the generative network, full sampling, action decoding, and RTC scheduling separately.

- Prioritize the online batch-size-one path; report training-feature-cache paths separately.
- Record GPU, precision, compilation settings, attention backend, token lengths, generation steps, and applicable block sizes.
- Use sufficient warmup and correct CUDA synchronization/events.
- Report p50/p95/p99 latency, peak memory, and actual network-call counts.
- Distinguish replanning rate, action-command frequency, and inference throughput.
- Compare FM at 5/8/12 steps; compare discrete rounds and, for the third policy, block/cache settings.

### 10.3 Trajectory visualization

Use the same representative samples across checkpoints, covering normal, difficult, and failure cases:

- Ground truth, encoder reconstruction, and policy prediction overlays.
- Per-dimension curves and 3D trajectories where available.
- RTC delay intervals, control-point support, committed prefixes, and switch locations.
- Base versus RTC, and cache on versus off where applicable.
- Synchronized original dual-camera views and predicted trajectories. Do not fabricate image-space geometric projections without calibration.

Actually inspect plots and representative video frames, record visual QA, and fix/revalidate issues discovered through visualization.

## 11. Suggested minimal repository structure

Adapt this to the reused code. Avoid building a large abstraction framework:

```text
robot_policy/
  configs/
    data/  policies/  training/  rtc/  inference/
  data/
    lerobot_adapter.py
    observation_cache.py
    action_targets.py
  encoders/
    vision.py
    state.py
    bspline_adapter.py
  policies/
    common/
    fm.py
    discrete_layerwise.py
    discrete_joint.py
  rtc/
    delay_mapping.py
    training.py
    inference.py
  inference/
    runner.py
    block_diffusion.py
    kv_cache.py
    executor.py
  evaluation/
    open_loop.py
    latency.py
    visualize.py
  scripts/
  tests/
  docs/
  outputs/
  README.md
```

Use the confirmed policy IDs. Share the trainer where practical; avoid duplicating RTC and representation-conversion logic inside each model.

Provide at least these script entry points:

```text
audit_data
prepare_observations
prepare_actions
smoke_test
train_base
finetune_rtc
evaluate_open_loop
benchmark_inference
visualize_trajectories
replay_rtc
```

Each needs a genuinely runnable example, configuration path, input/output documentation, and useful failure messages. Support checkpoint resume and configurable data/weight paths. Do not hardcode `/scratch/...` paths inside core models.

## 12. Final report and completion criteria

Deliver:

1. Complete modular code and a pinned or reproducible environment.
2. Source extraction/reuse mapping and attribution.
3. Independent observation-tokenization milestone artifacts.
4. BSplineEncoder/discrete representation integration and reconstruction validation.
5. Three architectures, parameter tables, and training configurations.
6. Six genuinely trained checkpoints and a manifest.
7. Training-time RTC and corresponding inference-time RTC for all three architectures.
8. Block diffusion and correctness-validated KV caching for `discrete_joint`.
9. Open-loop, latency, memory, and trajectory-visualization reports for all six models.
10. Reproduction instructions requiring one or a small number of commands.

For each model, `checkpoint_manifest.json` must include at least: architecture, base/RTC type, parent checkpoint, file path/hash, configuration, code commit, data split, encoder and vision-weight versions, training update count, and checkpoint selection metric.

Distinguish verified results, code-only implementations, incomplete work, and external blockers. If resources or weights are unavailable, report actual progress, remaining work, and resume commands. Do not claim completion by reducing experiments or copying weights.

**Completion means six loadable, evaluable, genuinely trained artifacts with the full validation evidence—not merely a code skeleton or design document.**
