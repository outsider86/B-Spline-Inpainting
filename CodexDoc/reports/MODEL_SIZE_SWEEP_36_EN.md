# DiT-S/B/L 36-Checkpoint Capacity Sweep

**Report date:** 2026-09-16

## Outcome

The capacity sweep is complete: **36/36** final checkpoints passed the requirement-level local and W&B audit. The experiment covers DiT-S, DiT-B, and DiT-L across raw and B-spline actions, FM, layerwise discrete diffusion, and joint discrete diffusion, each with a 50,000-update base stage and 5,000-update ttRTC stage.

These are open-loop validation metrics, not closed-loop robot success rates. `action_mse` is the common cross-architecture metric; architecture-specific training objectives are not directly comparable.

## Numerical stability warning

Post-sweep gradient forensics found FP32 gradient-norm overflow in six layerwise-DD runs: DiT-B raw and B-spline base/ttRTC, and DiT-L B-spline base/ttRTC. Their forward losses remained finite, but clipping erased the new gradient signal after the overflow. These checkpoints are valid reproduction artifacts, not trustworthy converged capacity comparisons. DiT-S layerwise and raw DiT-L layerwise remained finite. See `CodexDoc/reports/GRADIENT_INSTABILITY_INVESTIGATION_EN.md` for evidence and corrective recommendations.

## Fixed protocol

- Seed: 7 with deterministic CUDA/data loading.
- Micro batch = effective batch = 32.
- Base training: exactly 50,000 updates in `robot-policy-50k-bs32-basic`.
- ttRTC fine-tuning: exactly 5,000 updates in `robot-policy-5k-bs32-ttRTC`.
- Shapes: DiT-S 6×384/4 heads; DiT-B 12×768/12 heads; DiT-L 24×1024/16 heads.
- Each ttRTC checkpoint has an exact local base parent; joint-DD parent-prediction caches are keyed by the parent SHA-256.

## Trainable policy sizes

| Size | Representation | Architecture | Parameters |
|---|---|---|---:|
| DiT-S | Raw | FM | 15,072,903 |
| DiT-S | Raw | Layerwise DD | 15,035,136 |
| DiT-S | Raw | Joint DD | 13,850,496 |
| DiT-S | B-spline | FM | 15,068,295 |
| DiT-S | B-spline | Layerwise DD | 15,002,880 |
| DiT-S | B-spline | Joint DD | 13,818,240 |
| DiT-B | Raw | FM | 108,058,119 |
| DiT-B | Raw | Layerwise DD | 107,392,512 |
| DiT-B | Raw | Joint DD | 102,682,368 |
| DiT-B | B-spline | FM | 108,048,903 |
| DiT-B | B-spline | Layerwise DD | 107,328,000 |
| DiT-B | B-spline | Joint DD | 102,617,856 |
| DiT-L | Raw | FM | 367,582,471 |
| DiT-L | Raw | Layerwise DD | 366,170,624 |
| DiT-L | Raw | Joint DD | 357,842,432 |
| DiT-L | B-spline | FM | 367,570,183 |
| DiT-L | B-spline | Layerwise DD | 366,084,608 |
| DiT-L | B-spline | Joint DD | 357,756,416 |

## Checkpoint results

| Size | Representation | Architecture | Stage | Parameters | Final train action MSE | Final validation action MSE | Checkpoint | SHA-256 | W&B |
|---|---|---|---|---:|---:|---:|---|---|---|
| DiT-S | Raw | FM | Base | 15,072,903 | 0.0010298328 | 0.023626779 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_s/raw/checkpoints/fm_base.pt` | `c045b0d086480ff3da1881c87c44a2b67ecf26f96957b922e6aa8bd71d45bd4c` | [i9p5hb9y](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/i9p5hb9y) |
| DiT-S | Raw | FM | ttRTC | 15,072,903 | 0.00073005189 | 0.02645032 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_s/raw/checkpoints/fm_ttrtc.pt` | `360e5814cf8c0590cc5036e67f02447ed921dcdea044c972e05f9982c49c0483` | [v9ubgo5v](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/v9ubgo5v) |
| DiT-S | Raw | Layerwise DD | Base | 15,035,136 | 0.002314968 | 0.0055600479 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_s/raw/checkpoints/discrete_layerwise_base.pt` | `eff9e989fe18cf912e789dbd64e8d6068efc1e5bdafc13e9abfce8aa1bb6ac3b` | [zh84gs3w](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/zh84gs3w) |
| DiT-S | Raw | Layerwise DD | ttRTC | 15,035,136 | 0.00025027746 | 0.0062398141 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_s/raw/checkpoints/discrete_layerwise_ttrtc.pt` | `f5fc888db8ba3fa2efdafc6f9b25410ea9ac28e759427289be508152c05dbfc1` | [2o2kthwz](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/2o2kthwz) |
| DiT-S | Raw | Joint DD | Base | 13,850,496 | 0.00074377732 | 0.012555373 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_s/raw/checkpoints/discrete_joint_base.pt` | `3e561c7fb2af01f0c437259db382bb5afccdf54d15c97baf7519e833e0eef8ce` | [ykwmgupy](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/ykwmgupy) |
| DiT-S | Raw | Joint DD | ttRTC | 13,850,496 | 1.4295054e-05 | 0.0069909944 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_s/raw/checkpoints/discrete_joint_ttrtc.pt` | `b49affff04576f229f27d2b2072610f1636765a90c06fa45c2288b95894f21e2` | [dgxz1ipu](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/dgxz1ipu) |
| DiT-S | B-spline | FM | Base | 15,068,295 | 0.00060674868 | 0.033935821 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_s/bspline/checkpoints/fm_base.pt` | `fd56a07b4c41b54eff0ae946d3dfef316773a2ff00ee50672ad2ba52ed377cf6` | [2rb15m2t](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/2rb15m2t) |
| DiT-S | B-spline | FM | ttRTC | 15,068,295 | 0.00097202568 | 0.0356551 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_s/bspline/checkpoints/fm_ttrtc.pt` | `0cdc64d690ecdec7ecda6263b49dab98484d7eebe978c411790c3a30339417b3` | [83pdq57s](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/83pdq57s) |
| DiT-S | B-spline | Layerwise DD | Base | 15,002,880 | 0.00031072076 | 0.0029131562 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_s/bspline/checkpoints/discrete_layerwise_base.pt` | `ff761f1243a9b7f211536738ed32201b0a9c02b98e2af2926adf0bfa64272079` | [ya0xdf4n](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/ya0xdf4n) |
| DiT-S | B-spline | Layerwise DD | ttRTC | 15,002,880 | 0.0020413231 | 0.0037665326 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_s/bspline/checkpoints/discrete_layerwise_ttrtc.pt` | `7b6879fe5c630c3516c33387adafbf210c1b983b9927fb071a35bd5bdf4eb504` | [q28aplkd](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/q28aplkd) |
| DiT-S | B-spline | Joint DD | Base | 13,818,240 | 0.0010353074 | 0.010927464 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_s/bspline/checkpoints/discrete_joint_base.pt` | `2a0f8091c81d257856eae10d17d47eff8ddbe3370300cac25b228f3781f222bc` | [0q4g7cq4](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/0q4g7cq4) |
| DiT-S | B-spline | Joint DD | ttRTC | 13,818,240 | 7.1351133e-05 | 0.013666284 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_s/bspline/checkpoints/discrete_joint_ttrtc.pt` | `e026370cbcbb6744e32f7d7fee5f2cf1c9aa60dc7dc91af8d02dc0d8dd8543cc` | [340xpp9a](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/340xpp9a) |
| DiT-B | Raw | FM | Base | 108,058,119 | 0.00095223304 | 0.025668904 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_b/raw/checkpoints/fm_base.pt` | `59480b3d2bf830947f7ca145b2d6677e2e1e580a1a9abe8447e9f3e7a5b4646a` | [op6lpqi3](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/op6lpqi3) |
| DiT-B | Raw | FM | ttRTC | 108,058,119 | 0.00038307122 | 0.025613394 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_b/raw/checkpoints/fm_ttrtc.pt` | `de187fa53a5c51e22d5296aca17d5f622c10bde234da836f8c27145968b51149` | [duznrj8t](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/duznrj8t) |
| DiT-B | Raw | Layerwise DD | Base | 107,392,512 | 0.52690947 | 0.5175689 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_b/raw/checkpoints/discrete_layerwise_base.pt` | `7fac744844f4b7d865e5e3b0b2bb6e59b260ee58baf1f2d83c2bf9fbe197356a` | [o2g40xq8](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/o2g40xq8) |
| DiT-B | Raw | Layerwise DD | ttRTC | 107,392,512 | 0.47236577 | 0.5157025 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_b/raw/checkpoints/discrete_layerwise_ttrtc.pt` | `35ffb57856657c89b29745b3f01c822a6d4e964e18b1f7edd0e1870dea0220fa` | [ajt8p9vs](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/ajt8p9vs) |
| DiT-B | Raw | Joint DD | Base | 102,682,368 | 0.0089171017 | 0.032968935 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_b/raw/checkpoints/discrete_joint_base.pt` | `10a2c9eaf0c0d0da41f146221abd7d39942a2dfee6e004e2625d394727b5526c` | [x9qrzmwd](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/x9qrzmwd) |
| DiT-B | Raw | Joint DD | ttRTC | 102,682,368 | 0.023193398 | 0.03653868 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_b/raw/checkpoints/discrete_joint_ttrtc.pt` | `f6ef22b79382d464ecdb069b94b0cf8d07b9b3a8e4bc789f55bd0af361229940` | [1hux9mwd](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/1hux9mwd) |
| DiT-B | B-spline | FM | Base | 108,048,903 | 0.00069655431 | 0.027097748 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_b/bspline/checkpoints/fm_base.pt` | `340951daece1369c97a5a8c4e5f605fca41b5dca5138ed58351ce73471a11897` | [0c77dslj](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/0c77dslj) |
| DiT-B | B-spline | FM | ttRTC | 108,048,903 | 0.0006673591 | 0.029929488 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_b/bspline/checkpoints/fm_ttrtc.pt` | `6215dfaaed5288b06a1c77181d87e609fb816a5161612fc1caf157ecc8c7d8ac` | [25fzgt8t](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/25fzgt8t) |
| DiT-B | B-spline | Layerwise DD | Base | 107,328,000 | 0.19218299 | 0.19169513 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_b/bspline/checkpoints/discrete_layerwise_base.pt` | `067aa87efdc7a8bbf3a6322e1c2c1c147c0d0745ceb60f7f099253c4eb731877` | [i40wvlpn](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/i40wvlpn) |
| DiT-B | B-spline | Layerwise DD | ttRTC | 107,328,000 | 0.17721389 | 0.17480701 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_b/bspline/checkpoints/discrete_layerwise_ttrtc.pt` | `a7baae410941ff39f1c51641179a2d8a804de5f8ebe0a6a9869f4b81cf85dd98` | [wjpva0fp](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/wjpva0fp) |
| DiT-B | B-spline | Joint DD | Base | 102,617,856 | 0.0007201549 | 0.018151555 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_b/bspline/checkpoints/discrete_joint_base.pt` | `15d59a6a5b7634964c3e8ac409a4a3325732d02a3897c0315c94b82d43fa1155` | [tsg4colk](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/tsg4colk) |
| DiT-B | B-spline | Joint DD | ttRTC | 102,617,856 | 0.00064085366 | 0.02252504 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_b/bspline/checkpoints/discrete_joint_ttrtc.pt` | `8e1e75c73d34cb50340d3b3ecd53a4620b2e76d18f9c956d4f1e3251b0f5e0d0` | [jjch9idz](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/jjch9idz) |
| DiT-L | Raw | FM | Base | 367,582,471 | 0.00048533682 | 0.029271012 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_l/raw/checkpoints/fm_base.pt` | `b56ac6f764ce6abe17bd9d9129efec1f6e1a21c336faa91b1c18740e528033ba` | [tziiazri](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/tziiazri) |
| DiT-L | Raw | FM | ttRTC | 367,582,471 | 0.00057860714 | 0.027811256 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_l/raw/checkpoints/fm_ttrtc.pt` | `fc84379971878778051da9d2c9689889d7caad33e629f502fa9703ab91779902` | [redv2qv4](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/redv2qv4) |
| DiT-L | Raw | Layerwise DD | Base | 366,170,624 | 0.0091187395 | 0.042262553 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_l/raw/checkpoints/discrete_layerwise_base.pt` | `05a827fb7570ec3ab5efded67a986f77d8c461e1b76b0ebde135247bba908988` | [o0tj6klu](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/o0tj6klu) |
| DiT-L | Raw | Layerwise DD | ttRTC | 366,170,624 | 0.0089394553 | 0.038144829 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_l/raw/checkpoints/discrete_layerwise_ttrtc.pt` | `92107ac042da050da01a31604d31549d3e039952604bac815a391b9d394effaf` | [7ejattim](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/7ejattim) |
| DiT-L | Raw | Joint DD | Base | 357,842,432 | 0.0056533539 | 0.051996653 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_l/raw/checkpoints/discrete_joint_base.pt` | `46ba37da6e38d427aebd36ad5e567595721731bd30983d75a885987c639b152b` | [kvby9apd](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/kvby9apd) |
| DiT-L | Raw | Joint DD | ttRTC | 357,842,432 | 0.0020319659 | 0.050682377 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_l/raw/checkpoints/discrete_joint_ttrtc.pt` | `be4af0aee1f27807ba8b0cd0cfab5d5cc9132081742d4499675774ef7a2fe178` | [0i7ac2b6](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/0i7ac2b6) |
| DiT-L | B-spline | FM | Base | 367,570,183 | 0.00058947108 | 0.02595718 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_l/bspline/checkpoints/fm_base.pt` | `dfbaa30912bfc30adb7b95ec2935213931c9c7b9accfe144e3384897e4f021be` | [hso7oc5a](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/hso7oc5a) |
| DiT-L | B-spline | FM | ttRTC | 367,570,183 | 0.00064662012 | 0.029353741 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_l/bspline/checkpoints/fm_ttrtc.pt` | `2941af73c960752b4b3e3ac40bf7b860a5322ecf6b08ac017415123d04cefdaf` | [s56r3udm](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/s56r3udm) |
| DiT-L | B-spline | Layerwise DD | Base | 366,084,608 | 0.19847374 | 0.18238732 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_l/bspline/checkpoints/discrete_layerwise_base.pt` | `8a0e8e7a0993651103405f25481b792ed6bbc59c3b4a57590bdbf819ae791e87` | [i9zgz3hh](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/i9zgz3hh) |
| DiT-L | B-spline | Layerwise DD | ttRTC | 366,084,608 | 0.18263401 | 0.17526557 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_l/bspline/checkpoints/discrete_layerwise_ttrtc.pt` | `a205ce7c6706ffb66717bdffbd5b51a5855c5f5ff9fee2b865bd73ed28d246b7` | [pgnhmdfl](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/pgnhmdfl) |
| DiT-L | B-spline | Joint DD | Base | 357,756,416 | 0.0052995775 | 0.019476333 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_l/bspline/checkpoints/discrete_joint_base.pt` | `5c011eb80529ada7f02eb07d1b8101bbc320eb19e92f2791d2ea60ba9e8fbb5b` | [us6r4jrf](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/us6r4jrf) |
| DiT-L | B-spline | Joint DD | ttRTC | 357,756,416 | 0.0040660067 | 0.02367902 | `/scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/model_size_sweep_50k_bs32/dit_l/bspline/checkpoints/discrete_joint_ttrtc.pt` | `c66cae18ade46f2d1183dc6879cb189f5a6e1f2919c6e11d2ea98f3d5bfa7860` | [ra5ijsh8](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/ra5ijsh8) |

## Verification

The completion audit verifies exactly 36 expected `.pt` files and no extras; exact update counts and model shapes; batch size 32; parameter counts; local train/validation `action_mse`; independent state-dict reload; manifest uniqueness, hashes, and parent lineage; W&B finished state, update summaries, action-MSE history, and model artifacts.

Machine-readable evidence: `robot_policy/outputs/model_size_sweep_50k_bs32/completion_audit.json`.
