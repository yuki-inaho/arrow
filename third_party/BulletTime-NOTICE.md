The camera normalization and Plucker-ray formulas in scripts/infer_bullettime.py are adapted from BulletTime finetune/my_datasets/cam_utils.py, commit e0331619d74d09ea149874cbde145a3a0a3faf7a.

Upstream: https://github.com/19reborn/BulletTime
License: Apache-2.0 (see BulletTime-LICENSE).

Local changes add input validation, integer-center to edge-center intrinsic conversion at the model boundary, provenance, and standalone inference integration. The upstream model implementation is loaded from a separate pinned checkout. Model weights are not redistributed in this repository.
