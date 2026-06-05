# JointConn-v2 Follow-up Execution Design

## 1. Purpose

This document is the resume plan for continuing the JointConn-v2 paper project after the initial implementation, smoke training, and smoke inference have been completed.

When returning to this project later, follow this document before launching more training, ablations, quantitative experiments, or paper-output generation.

## 2. Current Project State

### 2.1 Implemented Code

JointConn-v2 has already been implemented and integrated into the current JointConn-v2 workspace.

Key files:

- `jointconn_v2_library/jointconn_v2.py`
- `jointconn_v2_library/geometry.py`
- `jointconn_v2_library/relative_position.py`
- `jointconn_v2_library/gcm_wfm.py`
- `jointconn_v2_library/jointconn_v2_model.py`
- `jointconn_v2_library/jointconn_v2_utils.py`
- `jointconn_v2_library/inference_pipeline.py`
- `train.py`
- `train_coco_train2017.py`
- `inference.py`
- `dataset/jointconn_v2_dataset.py`
- `sd-scripts/library/custom_offloading_utils.py`
- `scripts/smoke_jointconn_v2.py`

Important inference fixes already applied:

- Text encoders are loaded and released before JointConn-v2 is loaded.
- JointConn-v2/adapter weights are loaded on CPU first, then non-swapped parts are moved to GPU.
- `--sample_steps` controls inference denoise steps.
- `--inference_output_dir` prevents method outputs from overwriting each other.
- Inference uses `args.seed` in both `joint_generation` and `depth_to_image`.

### 2.2 Environment

Use this Python environment:

```text
C:\Users\zhao\Documents\New project\.venvs\swiftedit\Scripts\python.exe
```

Known runtime:

- PyTorch: `2.2.1+cu118`
- GPU: RTX 4090 24GB
- Current PyTorch build does not have flash attention enabled, so speed and VRAM are worse than ideal.

Before resuming, run:

```powershell
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
python -m py_compile inference.py train.py train_coco_train2017.py jointconn_v2_library\jointconn_v2_model.py jointconn_v2_library\inference_pipeline.py
```

If Python without PyTorch is picked up by default, use the explicit `swiftedit` Python path.

### 2.3 Completed Experiments

Read `experiments/progress.md` first. The most important completed experiment IDs are:

- `20260604_jointconnv2_one_step_clean_seed42`
- `20260604_jointconnv2_20step_stability_seed42`
- `20260604_jointconnv2_100step_stability_seed42`
- `20260604_jointconnv2_inference_smoke_step100_seed42`
- `20260604_fixed_validation_baseline_v2_smoke_seed42`

Current JointConn-v2 checkpoint for resume:

```text
experiments/dry_runs/jointconnv2_100step_stability/jointconn_v2_addons_step_000100.safetensors
```

Original JointDiT baseline repository:

```text
F:\Phd_Work\Neural Networks and ICML 2026\JointDiT-main
```

The current `D:\code\JointConn-v2` project no longer keeps the original JointDiT run scripts. Use the external repository above for future JointDiT baseline/comparison runs.

### 2.4 Fixed Smoke Validation Set

Initial fixed validation set:

```text
experiments/validation/fixed_set/manifest.jsonl
```

Current samples:

- `d2i_flower`: `depth_to_image`
- `joint_castle`: `joint_generation`

Current smoke comparison figure:

```text
experiments/qualitative/jointconnv2_step100_smoke_20260604/figures/smoke_comparison_grid.png
```

Observed smoke result:

- `depth_to_image`: step-100 v2 RGB is visibly blurrier than original JointDiT.
- `joint_generation`: step-100 v2 has clearer castle structure and depth than original JointDiT under 4-step smoke inference.

Conclusion: the v2 implementation path is runnable, but the 100-step checkpoint is not final paper quality.

## 3. Hard Stops and User Confirmation Points

Pause and ask the user before:

- Treating the current smoke baseline as the official paper baseline.
- Running any experiment expected to exceed 1 hour.
- Running a training job expected to produce more than 10GB outputs.
- Downloading large models or datasets.
- Reporting third-party or paper-reported results instead of reproduced comparison results.
- Removing or substantially changing a JointConn-v2 innovation point.
- Public release, GitHub upload, project renaming, or license-sensitive cleanup.

## 4. Resume Checklist

When resuming work:

1. Read:
   - `docs/method_intro.md`
   - `docs/jointconn_v2_code_implementation_design.md`
   - this document
   - `experiments/progress.md`
2. Verify no stale training or inference process is running.
3. Verify GPU memory is mostly free.
4. Run `py_compile` on modified Python files.
5. Confirm the current fixed validation manifest.
6. Continue from the next incomplete step below.

## 5. Next Execution Path

### Step A: Confirm or Expand the Fixed Validation Set

Current set is only a smoke set. Before official paper experiments, expand it.

Recommended staged expansion:

- Smoke: 2 samples, already done.
- Mini validation: 10 samples, balanced between `depth_to_image` and `joint_generation`.
- Paper validation: 50-100 samples, selected from COCO-style cached data and user-approved representative prompts.

Manifest format:

```json
{"case_id":"sample_id","task":"depth_to_image","prompt":"...","input_depth":"...","reference_image":"...","seed":42,"resolution":[512,512],"sample_steps_smoke":4}
{"case_id":"sample_id","task":"joint_generation","prompt":"...","input_depth":null,"reference_image":null,"seed":42,"resolution":[512,512],"sample_steps_smoke":4}
```

Keep all validation inputs under:

```text
experiments/validation/fixed_set/
```

### Step B: Train a Stronger v2 Checkpoint

The immediate priority is to train v2 beyond 100 steps.

Use the lightweight config that already passed:

- `jointconn_lora_rank=4`
- `jointconn_gate_hidden_dim=32`
- `jointconn_routing_hidden_dim=16`
- `jointconn_output_bottleneck_dim=32`
- `blocks_to_swap=12`
- `train_batch_size=1`
- `is_latent_training=true`
- `skip_train_end_full_model_save=true`

Recommended next training ladder:

1. 500-step checkpoint: quick quality check.
2. 1000-step checkpoint: first meaningful mini comparison.
3. 2000-step checkpoint: only after user confirmation, because it may exceed 1 hour.

Command template for 1000 steps:

```powershell
$python = 'C:\Users\zhao\Documents\New project\.venvs\swiftedit\Scripts\python.exe'
& $python -m accelerate.commands.launch --config_file default_config.yaml --mixed_precision bf16 train_coco_train2017.py `
  --is_latent_training `
  --enable_jointconn_v2 `
  --jointconn_lora_rank 4 `
  --jointconn_gate_hidden_dim 32 `
  --jointconn_routing_hidden_dim 16 `
  --jointconn_output_bottleneck_dim 32 `
  --train_batch_size 1 `
  --max_train_steps 1000 `
  --save_every_n_steps 250 `
  --blocks_to_swap 12 `
  --skip_train_end_full_model_save `
  --output_dir experiments/training/jointconnv2_1000step_seed42
```

Expected output checkpoints:

```text
experiments/training/jointconnv2_1000step_seed42/jointconn_v2_addons_step_000250.safetensors
experiments/training/jointconnv2_1000step_seed42/jointconn_v2_addons_step_000500.safetensors
experiments/training/jointconnv2_1000step_seed42/jointconn_v2_addons_step_000750.safetensors
experiments/training/jointconnv2_1000step_seed42/jointconn_v2_addons_step_001000.safetensors
```

Record the experiment in:

```text
experiments/progress.md
```

If `depth_to_image` remains blurred after 1000 steps, tune in this order:

1. Increase `p_joint_task` or reduce it depending on which branch underperforms.
2. Adjust `jointconn_beta_loss`, `jointconn_alpha_min`, and `jointconn_alpha_max`.
3. Increase `jointconn_output_bottleneck_dim` from `32` to `64`.
4. Test `p_sync` changes.
5. Consider adding cached depth-edge tensors for more stable edge loss.

### Step C: Inference Evaluation After Each Training Milestone

For each saved v2 checkpoint, run both fixed tasks.

Depth-conditioned template:

```powershell
$python = 'C:\Users\zhao\Documents\New project\.venvs\swiftedit\Scripts\python.exe'
& $python inference.py `
  --jointconn_v2_addons_path <V2_CHECKPOINT> `
  --pretrained_model_name_or_path models/flux/flux1-dev.safetensors `
  --clip_l models/flux/clip_l.safetensors `
  --t5xxl models/flux/t5xxl_fp16.safetensors `
  --ae models/flux/ae.safetensors `
  --gen_type depth_to_image `
  --text_prompt "a red flower with yellow centers is blooming" `
  --input_depth experiments/validation/fixed_set/depth_inputs/d2i_flower_depth.npy `
  --seed 42 `
  --sdpa `
  --guidance_scale 3.5 `
  --blocks_to_swap 12 `
  --mixed_precision bf16 `
  --save_precision bf16 `
  --full_bf16 `
  --sample_steps 4 `
  --inference_output_dir experiments/qualitative/<EXPERIMENT_ID>/outputs
```

Joint-generation template:

```powershell
$python = 'C:\Users\zhao\Documents\New project\.venvs\swiftedit\Scripts\python.exe'
& $python inference.py `
  --jointconn_v2_addons_path <V2_CHECKPOINT> `
  --pretrained_model_name_or_path models/flux/flux1-dev.safetensors `
  --clip_l models/flux/clip_l.safetensors `
  --t5xxl models/flux/t5xxl_fp16.safetensors `
  --ae models/flux/ae.safetensors `
  --gen_type joint_generation `
  --text_prompt castle `
  --output_resolution 512 512 `
  --seed 42 `
  --sdpa `
  --guidance_scale 3.5 `
  --blocks_to_swap 12 `
  --mixed_precision bf16 `
  --save_precision bf16 `
  --full_bf16 `
  --sample_steps 4 `
  --inference_output_dir experiments/qualitative/<EXPERIMENT_ID>/outputs
```

For paper-quality qualitative outputs, use more steps after smoke is healthy:

- `sample_steps=20` for quick paper-style review.
- `sample_steps=40` only after user confirms runtime.

### Step D: Add Formal Metric Scripts

The current `metrics.csv` is only a sanity check. Add real metrics before paper-level claims.

Recommended files:

```text
jointconn_v2_library/metrics/
  __init__.py
  edge_metrics.py
  depth_metrics.py
  image_metrics.py
scripts/eval_jointconn_outputs.py
scripts/make_qualitative_grid.py
```

Minimum metrics:

- Non-empty output check.
- RGB edge density.
- Depth edge density.
- RGB-depth gradient consistency.
- Depth-conditioned adherence: compare input depth to generated/re-estimated or generated depth.
- Scale-shift aligned depth RMSE for generated depth when a pseudo-depth target exists.

Paper-level metrics to add later:

- CLIPScore for text-image alignment.
- FID or KID for distribution quality.
- Edge F1 around geometry boundaries.
- SSIM/LPIPS where a reference image exists.

Do not claim final quantitative superiority from smoke metrics.

### Step E: Official Baseline Reproduction

Current baseline is a smoke baseline only.

Before official baseline:

1. Ask the user to confirm whether original JointDiT results from the smoke set are acceptable as a reproduction direction.
2. Expand fixed validation set.
3. Run original JointDiT from `F:\Phd_Work\Neural Networks and ICML 2026\JointDiT-main` at the same sample steps planned for v2.
4. Save outputs under:

```text
experiments/baseline/<EXPERIMENT_ID>/
```

Each baseline experiment directory must contain:

```text
command.txt
metrics.csv
summary.json
logs/
outputs/
report.md
```

Do not use the current JointConn-v2 workspace as the original JointDiT baseline runner.

### Step F: Ablation Preparation

Current code supports full v2 vs original baseline. Internal module ablations require new switches.

Recommended ablation flags:

```text
--jointconn_disable_geom_bias
--jointconn_disable_content_gate
--jointconn_disable_regional_routing
--jointconn_disable_layer_schedule
--jointconn_zero_edge_att
```

Recommended ablation groups:

- `JointDiT-original`
- `JointConn-v2-full`
- `NoGeomBias`
- `NoContentGate`
- `NoRegionalRouting`
- `NoLayerSchedule`
- `JointOnly`
- `DepthConditionedOnly`

Each ablation should be trained or evaluated under matched data, seed, step count, and checkpoint cadence.

Output location:

```text
experiments/ablation/<EXPERIMENT_ID>/
```

Stop if an ablation is clearly broken or severely underperforms after reasonable tuning; ask user whether to continue optimizing, downweight, or remove that innovation point.

### Step G: Qualitative Figures

Raw outputs belong under:

```text
experiments/qualitative/<EXPERIMENT_ID>/outputs/
```

Paper-ready figures belong under:

```text
paper_outputs/figures/
paper_outputs/captions/
```

All figure text must be English:

- Method names
- Titles
- Captions
- Axes
- Legends
- Annotations
- Subfigure labels

Recommended qualitative grid columns:

```text
Input / Condition | JointDiT | JointConn-v2 | Depth / Geometry | Zoomed Edge Region
```

### Step H: Quantitative Comparisons

Only begin after:

- Official baseline is accepted.
- v2 checkpoint is strong enough for paper-level comparison.
- Metrics are implemented and validated.

If user has not provided comparison methods, search recent related works and propose at least five candidates with code availability. Ask user before using paper-reported results or third-party outputs.

Output location:

```text
experiments/quantitative/
```

Required files:

```text
experiments/quantitative/design.md
experiments/quantitative/compared_methods.md
experiments/quantitative/report.md
experiments/quantitative/results/
experiments/quantitative/figures/
```

### Step I: Motivation Figure

Use real outputs to explain the paper problem:

- Original JointDiT can blur or mismatch RGB/depth structure.
- JointConn-v2 aims to improve geometry-aware selective communication.

Potential source cases:

- `d2i_flower`: use if v2 improves after longer training; currently it is a failure case.
- `joint_castle`: currently shows a promising v2 structure improvement.

Save under:

```text
figures/motivation/
paper_outputs/figures/
```

All labels must be English.

### Step J: Paper Outputs

After final experiments:

```text
paper_outputs/
  tables/
  figures/
  captions/
  latex/
  analysis/
```

Minimum outputs:

- Ablation table in CSV and LaTeX.
- Quantitative comparison table in CSV and LaTeX.
- Qualitative comparison figure.
- Motivation figure.
- English captions.
- English result-analysis paragraphs.

### Step K: Final Verification and Backup

Before final handoff:

1. Run syntax checks.
2. Run smoke inference.
3. Verify final checkpoint can load.
4. Verify fixed validation set can reproduce outputs.
5. Create backup under:

```text
backups/
```

Ask user before GitHub upload or public release.

## 6. Recommended Immediate Next Action

The best next action is:

1. Ask user to confirm that the smoke comparison direction is acceptable.
2. Run `JointConn-v2` 500-step or 1000-step training.
3. Re-run the fixed validation set at step 500/1000.
4. Decide whether depth-conditioned blur is resolved enough to proceed to ablation.

If the user simply says "execute", start with 500-step training unless they explicitly approve a longer run.

## 7. Failure Handling

### OOM During Inference Loading

Confirm `inference.py` still loads the model on CPU before moving non-swapped blocks to GPU. Do not revert to `to_empty(cuda)` before `load_state_dict`.

### OOM During Training

Use:

- `--blocks_to_swap 12`
- `--train_batch_size 1`
- lightweight v2 dims `4/32/16/32`
- `--skip_train_end_full_model_save`

If still OOM, reduce `jointconn_output_bottleneck_dim` or run fewer sample steps for inference validation.

### Poor `depth_to_image` Quality

Expected at 100 steps. First train longer. If still poor after 1000 steps:

- inspect task sampling ratio,
- inspect depth-conditioned task masks,
- increase depth-conditioned proportion,
- tune edge-weighted loss,
- verify `E_att=Edge(input_depth)` and `lambda_y=0` for depth-conditioned inference.

### Poor `joint_generation` Quality

Check that `E_att=zero_edge` for joint generation and no clean target depth leakage is used. Then train longer and compare with baseline at equal sample steps.

## 8. Required Logging Standard

Every significant run must append to:

```text
experiments/progress.md
```

Use this shape:

```text
## Experiment ID: YYYYMMDD_short_name_seed42

- Date:
- Goal:
- Code version:
- Environment:
- Dataset / samples:
- Command:
- Config:
- Resource estimate:
- Actual runtime:
- Actual disk usage:
- Main metric:
- Auxiliary metrics:
- Results:
- Output path:
- Status:
- Failure reason:
- Fixes made during execution:
- Next step:
```

Do not leave long-running processes active at the end of a turn.
