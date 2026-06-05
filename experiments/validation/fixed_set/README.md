# Fixed Validation Set

This directory defines the initial reproducible validation subset for JointDiT and JointConn-v2 comparisons.

The current subset is intentionally small so that smoke comparisons can run on a single RTX 4090 without becoming a long experiment. It should be expanded before paper-level quantitative reporting.

## Cases

- `d2i_flower`: depth-conditioned image generation with a fixed depth map and prompt.
- `joint_castle`: joint RGB-depth generation from a fixed text prompt.

## Inference Control

- Seed: `42`
- Resolution: `512 x 512`
- Smoke steps: `4`
- Paper-level steps: to be confirmed before long experiments
- Main smoke metric: successful non-empty RGB/depth outputs
- Auxiliary smoke metrics: edge consistency and depth-image structure metrics after metric scripts are added

## Expansion Rule

Add samples only through `manifest.jsonl`. Each sample must include a stable `case_id`, `task`, prompt, seed, resolution, and any conditioning input path.
