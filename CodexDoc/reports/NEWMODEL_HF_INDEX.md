# NewModel releases

`NewModel` is versioned so published experiments remain reproducible.

- `v1/`: byte-for-byte archive of the previous 36-checkpoint release. It
  includes DiT-S/B/L and historical layerwise-DD artifacts.
- `v2/`: current active release with 16 checkpoints: DiT-S/B × raw/B-spline ×
  FM/joint-DD × base/ttRTC. DiT-L and layerwise DD are intentionally excluded.

Use `v2` for new research and deployment. Use `v1` only to reproduce the
previous sweep. Each version contains its own README, manifests, action-codec
sidecars, and validation evidence.
