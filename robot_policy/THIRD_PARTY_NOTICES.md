# Third-party notices

- BSplineEncoder, commit `5aede68cee997a2d501210ce3dfbc15144d44447`, MIT License. Used as a runtime dependency; no mathematical implementation was copied.
- StarVLA, commit `2f17402a5ccaa09907516ae5e542b0fa6ee5d155`, MIT License. Architecture and objectives were adapted. Its referenced DiT/action encoder files carry NVIDIA Apache-2.0 notices.
- real-time-chunking-kinetix, commit `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`, MIT License. Its RTC delay distribution and inpainting semantics informed the adapter.
- DiscreteDiffusionVLA and the local `dd-openvla` root have restrictive notices. No code was copied. Publicly observable topology and behavior were independently implemented through PyTorch/timm APIs. The vendored Transformers fork is not distributed by this package.

See `docs/source_map.md` for exact symbols and snapshot hashes.

