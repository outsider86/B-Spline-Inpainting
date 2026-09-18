# Ground-truth-prefix inference RTC

This diagnostic measures conditional chunk inpainting. It is deliberately
different from `evaluate_rtc`, which shifts a previous model prediction to
simulate replanning delay. Here, each example loads the current recorded
observation and the current ground-truth 30-step action chunk, exposes a prefix,
and asks the checkpoint to generate the remaining suffix from scratch.

## Conditioning contract

- Raw representation: freeze exactly the first `d × 7` ground-truth action
  values or action-bin tokens.
- B-spline representation: map the `d` raw steps to affected spline spans and
  freeze the complete cubic support, `ceil(d / 2) + 3` control rows. This is the
  minimum representation-correct condition; freezing only `d` controls would
  confuse raw action indices with spline control indices.
- FM uses continuous ground-truth controls. Joint DD uses the prepared
  ground-truth action tokens. All other values are initialized and sampled by
  the normal checkpoint sampler.
- Base and ttRTC checkpoints receive the identical inputs, sample IDs, prefix
  lengths, and metrics. This isolates learned inpainting behavior.

The main metric is physical action MSE over suffix steps `[d, 30)`. Reports also
include normalized suffix MSE, reconstruction-aware suffix MSE, physical error
of the conditioned region relative to the raw dataset, fixed-control maximum
error, per-sample rows, and synchronized sampling latency.

## One checkpoint

```bash
evaluate_inference_rtc \
  --config configs/model_size_sweep/dit_s_raw.yaml \
  --set data.prepared_path=outputs/SWEEP/summary/cache/raw \
  --set data.vision_cache_path=outputs/SWEEP/summary/cache/vision \
  --architecture discrete_joint \
  --checkpoint outputs/SWEEP/dit_s/raw/checkpoints/discrete_joint_ttrtc.pt \
  --output-dir outputs/RTCEVAL/raw_discrete_joint_ttrtc \
  --prefixes 2,4,6,8,10 \
  --plot-prefix 6 \
  --max-samples 64
```

Each result folder contains:

- `report.json`: provenance and aggregate curves across prefix lengths.
- `per_sample_metrics.csv`: every sample/prefix measurement.
- `trajectory_examples.npz`: exact arrays behind the example visualization.
- `trajectory_examples.png`: condition action, ground-truth suffix, and
  predicted suffix for five episode-balanced test examples and all 7 channels.
  Every sample row uses the same true physical dataset min/max for its action
  channel, computed across every prepared episode. Red boundary triangles mark
  predictions outside that dataset range instead of silently hiding them.
- `metrics_vs_prefix.png`: suffix quality and prefix-preservation curves.

## Complete DiT-S matrix

```bash
python scripts/run_inference_rtc_eval.py --gpus 0,1,2,3 --force
```

The runner creates exactly eight directly named variant folders (raw/B-spline
× FM/joint-DD × base/ttRTC) under
`outputs/RTCEVAL` and then validates their availability while producing the
root summary artifacts. Results are open-loop conditional generation metrics;
they are not simulator or physical-robot success rates.
