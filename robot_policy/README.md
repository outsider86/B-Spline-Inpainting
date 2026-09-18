# Modular robot policies

This repository is the isolated implementation workspace for the stacking-cups task. It provides representation-aware preprocessing, two active policy architectures, lineage-correct RTC fine-tuning, checkpoint manifests, open-loop evaluation, latency measurement, visual QA, and a dataset-only timestamped executor. It supports both direct raw-action prediction and B-spline action encoding. It never actuates physical hardware.

The production inference server for active raw/B-spline, FM/joint-DD,
base/ttRTC, and DiT-S/B checkpoints is documented in
[`deployment/README.md`](deployment/README.md). It implements the existing
Piper msgpack/WebSocket interface and returns decoded, unnormalized 30x7
absolute action chunks. Historical DiT-L and layerwise checkpoints remain
loadable for reproducibility but are excluded from active training and
evaluation.

Published checkpoints are versioned in the
[`DiscreteRTC/dRTC` NewModel directory](https://huggingface.co/datasets/DiscreteRTC/dRTC/tree/main/NewModel):
`v1` preserves the prior 36-checkpoint archive and `v2` contains the active
16-checkpoint full-vision release.

## Architecture IDs

- `fm`: StarVLA-style layerwise-conditioned continuous flow matching.
- `discrete_joint`: joint `[observation | action]` block diffusion with exact prefix caching under a block-causal attention topology.

`discrete_layerwise` is a legacy, load-only architecture retained for the
existing checkpoint archive and historical reports. New training and evaluation
entry points intentionally accept only `fm` and `discrete_joint`.

DiT-L is also legacy/load-only for new research as of 2026-09-18. Existing
DiT-L checkpoints remain supported by the production loader, but active
training/evaluation and sweep defaults use DiT-S and DiT-B only.

All use current observations only, shared frozen DINOv2 + SigLIP weights across the global and hand cameras, and a trainable compact projector and state tokenizer. `configs/default.yaml` selects 18 × 7 uniform-left cubic B-spline controls; `configs/raw_actions.yaml` selects the direct 30 × 7 raw-action sequence. The raw path does not invoke the B-spline encoder.

## Setup

```bash
conda activate starVLA
cd /scratch/wangpc/B-Spline-Inpainting/robot_policy
pip install -e ../BSplineEncoder
pip install -e '.[test]'
export HF_HOME=/scratch/wangpc/B-Spline-Inpainting/.cache/huggingface
```

Every command supports `--config` and repeated `--set section.key=value`. Core modules contain no dataset or weight absolute paths.

## Reproduction

```bash
# Audit and train-only normalization/action targets.
audit_data --config configs/default.yaml
prepare_actions --config configs/default.yaml

# Run one worker per free GPU; workers own disjoint episode IDs.
CUDA_VISIBLE_DEVICES=4 prepare_observations --config configs/default.yaml --rank 0 --world-size 2 &
CUDA_VISIBLE_DEVICES=5 prepare_observations --config configs/default.yaml --rank 1 --world-size 2 &
wait

smoke_test --config configs/default.yaml
pytest

# Keep effective global batch=128. These can run concurrently on two GPUs.
CUDA_VISIBLE_DEVICES=4 train_base --config configs/default.yaml --architecture fm --output outputs/checkpoints/fm_base.pt
CUDA_VISIBLE_DEVICES=5 train_base --config configs/default.yaml --architecture discrete_joint --output outputs/checkpoints/discrete_joint_base.pt

CUDA_VISIBLE_DEVICES=4 finetune_rtc --config configs/default.yaml --architecture fm --parent outputs/checkpoints/fm_base.pt --output outputs/checkpoints/fm_rtc.pt
CUDA_VISIBLE_DEVICES=5 finetune_rtc --config configs/default.yaml --architecture discrete_joint --parent outputs/checkpoints/discrete_joint_base.pt --output outputs/checkpoints/discrete_joint_rtc.pt
```

`train_base` and `finetune_rtc` accept `--resume PATH` and `--updates N`. A base checkpoint always starts from random initialization; RTC always verifies and loads the matching base architecture. The default effective batch, split, update budget and common optimizer are identical within each stage.

Evaluate one checkpoint as follows; use the same held-out split and command for all six:

```bash
evaluate_open_loop --config configs/default.yaml --architecture fm --checkpoint outputs/checkpoints/fm_base.pt --output outputs/evaluation/fm_base.json
evaluate_rtc --config configs/default.yaml --architecture fm --checkpoint outputs/checkpoints/fm_rtc.pt --output outputs/evaluation/fm_rtc_delays.json
benchmark_inference --config configs/default.yaml --architecture fm --checkpoint outputs/checkpoints/fm_base.pt --output outputs/latency/fm_base.json
replay_rtc --config configs/default.yaml --architecture fm --checkpoint outputs/checkpoints/fm_rtc.pt --delay-spans 3 --output outputs/replay/fm_D3.json
replay_rtc --config configs/default.yaml --architecture fm --checkpoint outputs/checkpoints/fm_rtc.pt --delay-raw-actions 3 --output outputs/replay/fm_d3_within_span.json
visualize_trajectories --config configs/default.yaml --output outputs/visualizations
finalize_checkpoints --config configs/default.yaml --directory outputs/checkpoints
build_report --root outputs --output outputs/visualizations/final
```

The metrics are open-loop predictions under recorded observations. RTC replay exercises buffering, timing, plan switches, support masks and reset behavior, but is not a simulator/robot success rate.

### Ground-truth-prefix inference RTC

`evaluate_inference_rtc` is the controlled inpainting diagnostic. It supplies
the current observation and a ground-truth prefix from the same action chunk,
freezes that prefix in the checkpoint's native raw or B-spline representation,
and generates the remaining chunk from scratch. The 8-way DiT-S runner covers
raw/B-spline × FM/joint-DD × base/ttRTC:

```bash
python scripts/run_inference_rtc_eval.py \
  --gpus 0,1,2,3 \
  --prefixes 2,4,6,8,10 \
  --max-samples 64 \
  --batch-size 32
```

Results are written to `outputs/RTCEVAL`, with one directly named folder per
variant and a root `SUMMARY.md`, `summary.json`, `summary.csv`, and comparison
plot. See [the inference-RTC contract](docs/inference_rtc.md) for exact
conditioning semantics and the single-checkpoint command.

## Raw-action experiment

The completed raw-action experiment uses `configs/raw_actions.yaml`, deterministic seed `7`, and online W&B logging. Basic and ttRTC runs are separated into the `raw-actions-basic` and `raw-actions-ttRTC` projects. Both training and validation log a dedicated normalized `action_mse` metric in addition to each architecture's native objective.

```bash
prepare_actions --config configs/raw_actions.yaml

CUDA_VISIBLE_DEVICES=4 train_base --config configs/raw_actions.yaml --architecture fm --output outputs/raw_actions/checkpoints/fm_base.pt
CUDA_VISIBLE_DEVICES=4 finetune_rtc --config configs/raw_actions.yaml --architecture fm --parent outputs/raw_actions/checkpoints/fm_base.pt --output outputs/raw_actions/checkpoints/fm_ttrtc.pt

evaluate_open_loop --config configs/raw_actions.yaml --architecture fm --checkpoint outputs/raw_actions/checkpoints/fm_base.pt --output outputs/raw_actions/evaluation/fm_base.json
evaluate_rtc --config configs/raw_actions.yaml --architecture fm --checkpoint outputs/raw_actions/checkpoints/fm_ttrtc.pt --output outputs/raw_actions/evaluation/fm_ttrtc_delays.json
verify_reproducibility --first outputs/raw_actions/checkpoints/fm_base.pt --rerun outputs/raw_actions/reproducibility/fm_base_rerun.pt --output outputs/raw_actions/reproducibility/fm.json
```

Repeat the same commands with `discrete_joint`. The authoritative historical raw checkpoint manifest is `outputs/raw_actions/checkpoints/checkpoint_manifest.json`; the English and Chinese result summaries are in `../CodexDoc/reports/`.

## Controlled B-spline reproduction and comparison

`configs/bspline_reproduction.yaml` reruns the six B-spline models under the same deterministic protocol as the raw-action experiment. Its support-aware normalized decoded `action_mse` is directly comparable with raw actions and is logged as `train/action_mse` and `validation/action_mse`.

```bash
prepare_actions --config configs/bspline_reproduction.yaml

CUDA_VISIBLE_DEVICES=4 train_base --config configs/bspline_reproduction.yaml --architecture fm --output outputs/bspline_reproduction/checkpoints/fm_base.pt
CUDA_VISIBLE_DEVICES=4 finetune_rtc --config configs/bspline_reproduction.yaml --architecture fm --parent outputs/bspline_reproduction/checkpoints/fm_base.pt --output outputs/bspline_reproduction/checkpoints/fm_rtc.pt

compare_representations \
  --raw-root outputs/raw_actions \
  --bspline-root outputs/bspline_reproduction \
  --output outputs/action_representation_comparison
```

Repeat training, fine-tuning, evaluation, and reproducibility verification for `discrete_joint`. The authoritative historical B-spline manifest is `outputs/bspline_reproduction/checkpoints/checkpoint_manifest.json`. The dedicated comparison W&B project is `robot-policy-action-representation-comparison`; bilingual result reports are in `../CodexDoc/reports/BSPLINE_REPRODUCTION_AND_COMPARISON_{EN,CN}.md`.

## 50k/5k batch-32 checkpoints

The long-run configurations are `configs/longrun_50k_bs32_raw.yaml` and `configs/longrun_50k_bs32_bspline.yaml`. They use 50,000 basic updates, 5,000 ttRTC updates, and both micro/effective batch size 32. All six basic variants share the `robot-policy-50k-bs32-basic` W&B project; all six ttRTC variants share `robot-policy-5k-bs32-ttRTC`.

The completed checkpoints and authoritative manifests are under `outputs/longrun_50k_bs32/{raw,bspline}/checkpoints/`. See `../CodexDoc/reports/LONGRUN_50K_BS32_{EN,CN}.md` for run IDs, direct W&B links, final validation action MSE, and the completion audit.

## Contracts and evidence

- [Source map](docs/source_map.md)
- [Data contract](docs/data_contract.md)
- [Architecture decision](docs/architecture_decision.md)
- [Environment](docs/environment.md)
- [Training report](docs/training_report.md)
- [Evaluation report](docs/evaluation_report.md)
- [Cache and latency](docs/cache_and_latency.md)
- [Visual QA](docs/visual_qa.md)
- [Final requirement audit](docs/final_audit.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Live project progress](../CodexDoc/PROGRESS.md)

Outputs are intentionally outside package source. `outputs/prepared` contains split/statistics/encoder manifests and cached frozen features; `outputs/checkpoints/checkpoint_manifest.json` is the authoritative six-artifact lineage and hash record.
