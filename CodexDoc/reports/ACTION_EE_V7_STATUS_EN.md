# ActionEE V7 Status

V7 trains one observation-state variant (`State_EE`) over three tasks and two
action representations, yielding six base and six ttRTC checkpoints.

- State: 8D TCP position, quaternion-xyzw, and gripper.
- Action: 7D local delta translation, local delta rotation-vector, and next
  gripper.
- Observation: h1, exactly the current global and hand images.
- Policy: BSP-UNet flow matching with raw and cubic uniform-left B-spline
  action representations.
- Protocol: 100 base epochs with complete validation every 10 epochs, followed
  by five ttRTC epochs with complete validation every epoch.
- Tracking: all 12 runs share `robot-policy-bsp-unet-v7-ee`; no user/model
  artifacts are logged.

Every source episode has been audited. Hanging has 61 episodes / 26,698
frames, stacking has 61 / 41,752, and classify has 50 / 95,900. All files have
finite 8D states and 7D actions, consecutive frame indices, strictly increasing
timestamps, and metadata matching the EE/delta-EE semantic contract.

The six base jobs ran concurrently on GPUs 0--5. ttRTC jobs whose parents
finished early used the freed GPUs, and the same driver idempotently completed
the remaining jobs. Local outputs live under `robot_policy/output/NEW/V7Full`;
the final audited tree contains six `base.pt`, six `ttrtc.pt`, and load
metadata, and has been published to `DiscreteRTC/dRTC/NewModel/V7Full`.

## Final results (2026-09-24)

| Task | Representation | Stage | Selected epoch | Full-validation action MSE | W&B ID |
|---|---|---|---:|---:|---|
| hanging_mug | raw | base | 100 | 0.05006838 | `ey82cx7f` |
| hanging_mug | raw | ttRTC | 5 | 0.04991525 | `a9loddl2` |
| hanging_mug | B-spline | base | 100 | 0.04653973 | `6yn4b220` |
| hanging_mug | B-spline | ttRTC | 5 | 0.04652055 | `4768yan2` |
| stacking_cup | raw | base | 100 | 0.05624402 | `8gqr77ky` |
| stacking_cup | raw | ttRTC | 5 | 0.05647647 | `je27amjd` |
| stacking_cup | B-spline | base | 70 | 0.05345627 | `a6wbwnij` |
| stacking_cup | B-spline | ttRTC | 5 | 0.05361407 | `kx3qow0j` |
| classify_blocks | raw | base | 90 | 0.05737160 | `pkalnt69` |
| classify_blocks | raw | ttRTC | 5 | 0.05701298 | `bbp2e1c9` |
| classify_blocks | B-spline | base | 70 | 0.05542328 | `klf0hh7z` |
| classify_blocks | B-spline | ttRTC | 4 | 0.05543821 | `oheabq6u` |

Every base completed 100 epochs; the table reports the validation-selected
epoch. Base runs evaluated the complete validation split every 10 epochs, and
ttRTC runs did so every epoch. The complete splits contain 2,723 hanging,
4,212 stacking, and 9,469 classify samples. All 12 runs have zero optimizer
skips and finite gradient diagnostics.

## Completion audit and publication

- The local tree contains exactly six `base.pt` and six `ttrtc.pt` files, with
  no resume, step, best-weight, or best-epoch training checkpoint left over.
- All 12 models pass an independent strict CPU load. Each has 87,333,831
  parameters. Raw predicts a 30x7 action directly; B-spline predicts 18x7
  cubic uniform-left control points and decodes them to a 30x7 action.
- All six ttRTC parent paths and base SHA-256 values match.
- All six task/representation pairs pass a simulated downloaded-layout test
  with the workstation's absolute prepared path unavailable; the loader finds
  `State_EE/sidecars/{raw,bspline}` automatically.
- The single W&B project has exactly 12 V7 runs, all `finished`, and zero
  user/model artifacts. W&B created one non-deletable system-managed
  `wandb-history` parquet for `8gqr77ky`; it is backend-owned, not a model
  upload.
- The final Hugging Face release contains 12 models, 24 server/training JSON
  files, and 12 portable sidecars. Model LFS SHA-256, sidecar blob hashes, and
  remote presence all match at revision
  `9e334da2f2a8aaca002f410b18cc38ae9dc3cc18`.
- The two classify epoch-30 collaborator snapshots remain available and are
  explicitly marked non-final.
- The complete repository test suite passes, with one historical optional
  test skipped.

Authoritative evidence is stored in `V7_STATUS.json`, `V7_AUDIT.json`,
`V7_HF_UPLOAD.json`, and `V7_INDEPENDENT_AUDIT.json` under the V7 output root.
The V7 ActionEE goal is complete; LastCommand and State+LastCommand remain out
of scope pending a separate decision.
