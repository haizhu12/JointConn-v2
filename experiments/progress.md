## Experiment ID: 20260604_jointconnv2_phaseabc_static_seed42

- Date: 2026-06-04
- Goal: Rebuild the first JointConn-v2 implementation layer on top of the existing JointDiT codebase.
- Code version: local workspace, not a Git repository
- Dataset / samples: not used
- Command: `python -m py_compile jointconn_v2_library\geometry.py jointconn_v2_library\relative_position.py jointconn_v2_library\jointconn_v2.py jointconn_v2_library\gcm_wfm.py jointconn_v2_library\jointconn_v2_utils.py jointconn_v2_library\jointconn_v2_model.py jointconn_v2_library\inference_pipeline.py train.py inference.py`
- Config: default flags keep `enable_jointconn_v2=False`; v2 path uses `lora_branch_mode=depth_only`
- Resource estimate: CPU-only static validation
- Actual runtime: under 1 minute
- Actual disk usage: negligible
- Main metric: syntax/static integration success
- Auxiliary metrics: not applicable
- Results: Python compilation passed for modified and new files
- Output path: source files under `jointconn_v2_library/`, `train.py`, `inference.py`
- Status: partial
- Failure reason: runtime tensor smoke test could not run because default Python 3.13 environment has no `torch`
- Notes: v2 connector, geometry edge map, relative position bias, and packed-space GCM-WFM loss are implemented behind opt-in flags.
- Next step: run smoke tests in the project Python 3.10/PyTorch environment, then perform one-batch training validation.

## Experiment ID: 20260604_jointconnv2_coco_entry_static_seed42

- Date: 2026-06-04
- Goal: Align `train_coco_train2017.py` with the JointConn-v2 implementation path and add a small module smoke test entry.
- Code version: local workspace, not a Git repository
- Dataset / samples: not used
- Command: `python -m py_compile jointconn_v2_library\geometry.py jointconn_v2_library\relative_position.py jointconn_v2_library\jointconn_v2.py jointconn_v2_library\gcm_wfm.py jointconn_v2_library\jointconn_v2_utils.py jointconn_v2_library\jointconn_v2_model.py jointconn_v2_library\inference_pipeline.py train.py train_coco_train2017.py inference.py scripts\smoke_jointconn_v2.py`
- Config: default flags keep `enable_jointconn_v2=False`; v2 path uses `lora_branch_mode=depth_only`, `p_joint_task=0.5`, `p_sync=0.5`
- Resource estimate: CPU-only static validation
- Actual runtime: under 1 minute
- Actual disk usage: negligible
- Main metric: syntax/static integration success
- Auxiliary metrics: not applicable
- Results: Python compilation passed for modified training, inference, model, loss, geometry, connector, and smoke test files
- Output path: `train_coco_train2017.py`, `scripts/smoke_jointconn_v2.py`, updated source modules
- Status: partial
- Failure reason: runtime tensor smoke test still requires a Python environment with `torch`
- Notes: COCO training entry now supports opt-in JointConn-v2, packed-space GCM-WFM, task masks, checkpoint JSON flags, and Windows-safe `device_norm` usage. Inference now rejects `enable_jointconn_v2=True` for `depth_estimation`, matching the confirmed v2 scope.
- Next step: run `python scripts\smoke_jointconn_v2.py` inside the project PyTorch environment, then run a one-batch COCO training dry run with `--enable_jointconn_v2`.

## Experiment ID: 20260604_jointconnv2_runtime_smoke_and_dryrun_seed42

- Date: 2026-06-04
- Goal: Execute JointConn-v2 tensor smoke tests and attempt a one-step COCO latent training dry run.
- Code version: local workspace, not a Git repository
- Environment: `C:\Users\zhao\Documents\New project\.venvs\swiftedit\Scripts\python.exe`, PyTorch `2.2.1+cu118`, RTX 4090 24GB
- Dataset / samples: `data/train2017` latent cache, 200 common samples
- Commands:
  - `python scripts\smoke_jointconn_v2.py`
  - `accelerate launch ... train_coco_train2017.py --is_latent_training --enable_jointconn_v2 --max_train_steps 1`
- Config:
  - Smoke: synthetic tensors
  - Dry run 1: default v2 rank/hidden settings with `--blocks_to_swap 12`
  - Dry run 2: `jointconn_lora_rank=4`, gate/routing/bottleneck `32/16/32`, `--blocks_to_swap 12`
  - Dry run 3: `jointconn_lora_rank=1`, gate/routing/bottleneck `8/4/8`, no block swap
- Resource estimate: one GPU, no checkpoint save
- Actual runtime: smoke under 5 seconds; dry-run attempts loaded the full FLUX model and were stopped/failed during first step
- Actual disk usage: log files under `experiments/dry_runs/`
- Main metric: runtime path reaches forward/backward/optimizer validation
- Auxiliary metrics: trainable parameter count, GPU memory usage
- Results:
  - Smoke test passed.
  - Dataset latent smoke passed: `image_latent/depth_latent=(16,64,64)`, `t5_latents=(512,4096)`, `len=200`.
  - Default connector was too large: 2,645,861,998 trainable params and OOM.
  - Lightweight connector reduced dry-run trainable params to 88,976,062 and reached forward/backward, but block swap caused AdamW CPU/CUDA parameter-state mismatch at optimizer step.
  - No-swap minimal connector reduced trainable params to 50,783,554 and entered the training step, but without flash attention it saturated RTX 4090 memory (`~24GB/24.6GB`) and did not finish in a practical time window.
- Output path: `experiments/dry_runs/jointconnv2_one_step*.log`
- Status: partial
- Failure reason: full FLUX one-step validation is constrained by the current PyTorch build lacking flash attention and by block-swap optimizer-device incompatibility for trainable adapters.
- Fixes made during execution:
  - Replaced full `JointConnV2Block.out_proj(2H->2H)` with a low-rank bottleneck projection.
  - Added configurable `jointconn_lora_rank`, `jointconn_gate_hidden_dim`, `jointconn_routing_hidden_dim`, and `jointconn_output_bottleneck_dim`.
  - Fixed `JointConnV2Model.enable_block_swap()` argument order for `ModelOffloader`.
  - Fixed CUDA stream creation in `custom_offloading_utils.py` for PyTorch 2.2.
  - Made latent dataset empty-prompt dropout robust when `empty_prompts/` is absent.
- Next step: install/use a PyTorch build with flash attention or xFormers-compatible attention, then rerun one-step with lightweight v2 config; separately fix block-swap optimizer compatibility if block swapping is required for final training.

## Experiment ID: 20260604_jointconnv2_one_step_clean_seed42

- Date: 2026-06-04
- Goal: Complete a clean one-step JointConn-v2 COCO latent training dry run with block swap enabled.
- Code version: local workspace, not a Git repository
- Environment: `C:\Users\zhao\Documents\New project\.venvs\swiftedit\Scripts\python.exe`, PyTorch `2.2.1+cu118`, RTX 4090 24GB
- Dataset / samples: `data/train2017` latent cache, 200 common samples
- Command: `accelerate launch --config_file default_config.yaml --mixed_precision bf16 train_coco_train2017.py --is_latent_training --enable_jointconn_v2 --jointconn_lora_rank 4 --jointconn_gate_hidden_dim 32 --jointconn_routing_hidden_dim 16 --jointconn_output_bottleneck_dim 32 --train_batch_size 1 --max_train_steps 1 --blocks_to_swap 12 --skip_train_end_full_model_save`
- Config: v2 lightweight dry-run config, no final full-model save, no step checkpoint save
- Resource estimate: one GPU, no checkpoint save
- Actual runtime: model load plus one training step; step time about 10.55 seconds
- Actual disk usage: log files only
- Main metric: one training step completes with clean process exit
- Auxiliary metrics: trainable parameter count and loss
- Results:
  - One-step training completed successfully.
  - Trainable parameters: 88,976,062.
  - Final progress: `1/1`, `avr_loss=0.576`.
  - GPU memory was released after process exit.
- Output path: `experiments/dry_runs/jointconnv2_one_step_clean_stdout.log`, `experiments/dry_runs/jointconnv2_one_step_clean_stderr.log`
- Status: success
- Fixes made during execution:
  - Added `JointConnV2Model.wait_for_pending_block_swaps()`.
  - Added optimizer-step preparation to move trainable swapped params, grads, and optimizer states back to GPU before AdamW update.
  - Added `--skip_train_end_full_model_save` for adapter-only dry runs.
- Next step: run a short 20-100 step stability experiment with the same lightweight config and save adapter checkpoints every 20-50 steps.

## Experiment ID: 20260604_jointconnv2_20step_stability_seed42

- Date: 2026-06-04
- Goal: Verify short-run JointConn-v2 training stability and adapter checkpoint saving.
- Code version: local workspace, not a Git repository
- Environment: `C:\Users\zhao\Documents\New project\.venvs\swiftedit\Scripts\python.exe`, PyTorch `2.2.1+cu118`, RTX 4090 24GB
- Dataset / samples: `data/train2017` latent cache, 200 common samples
- Command: `accelerate launch --config_file default_config.yaml --mixed_precision bf16 train_coco_train2017.py --is_latent_training --enable_jointconn_v2 --jointconn_lora_rank 4 --jointconn_gate_hidden_dim 32 --jointconn_routing_hidden_dim 16 --jointconn_output_bottleneck_dim 32 --train_batch_size 1 --max_train_steps 20 --save_every_n_steps 20 --blocks_to_swap 12 --skip_train_end_full_model_save`
- Config: v2 lightweight stability config, adapter checkpoint save at step 20
- Resource estimate: one GPU, one adapter checkpoint
- Actual runtime: about 1 minute 14 seconds for the training loop
- Actual disk usage: about 178 MB adapter checkpoint plus JSON/logs
- Main metric: 20-step training completes without NaN/OOM and saves adapter checkpoint
- Auxiliary metrics: moving-average loss and checkpoint integrity
- Results:
  - 20/20 steps completed successfully.
  - Final progress: `avr_loss=0.76`.
  - Adapter checkpoint saved: `experiments/dry_runs/jointconnv2_20step_stability/jointconn_v2_addons_step_000020.safetensors`.
  - Training flags JSON saved and includes JointConn-v2 config.
- Output path: `experiments/dry_runs/jointconnv2_20step_stability/`
- Status: success
- Next step: expand to 100-step stability training with adapter saves at steps 50 and 100.

## Experiment ID: 20260604_jointconnv2_100step_stability_seed42

- Date: 2026-06-04
- Goal: Verify 100-step JointConn-v2 stability before baseline/ablation experiments.
- Code version: local workspace, not a Git repository
- Environment: `C:\Users\zhao\Documents\New project\.venvs\swiftedit\Scripts\python.exe`, PyTorch `2.2.1+cu118`, RTX 4090 24GB
- Dataset / samples: `data/train2017` latent cache, 200 common samples
- Command: `accelerate launch --config_file default_config.yaml --mixed_precision bf16 train_coco_train2017.py --is_latent_training --enable_jointconn_v2 --jointconn_lora_rank 4 --jointconn_gate_hidden_dim 32 --jointconn_routing_hidden_dim 16 --jointconn_output_bottleneck_dim 32 --train_batch_size 1 --max_train_steps 100 --save_every_n_steps 50 --blocks_to_swap 12 --skip_train_end_full_model_save`
- Config: v2 lightweight stability config, adapter checkpoint saves at steps 50 and 100
- Resource estimate: one GPU, two adapter checkpoints
- Actual runtime: about 5 minutes 39 seconds for the training loop
- Actual disk usage: about 356 MB adapter checkpoints plus JSON/logs
- Main metric: 100-step training completes without NaN/OOM and saves adapter checkpoints
- Auxiliary metrics: moving-average loss, checkpoint integrity, GPU release after process exit
- Results:
  - 100/100 steps completed successfully.
  - Final progress: `avr_loss=0.793`.
  - Adapter checkpoints saved at steps 50 and 100.
  - Step-100 JSON confirms `enable_jointconn_v2=true`, `jointconn_lora_rank=4`, gate/routing/bottleneck `32/16/32`, and `lora_branch_mode=depth_only`.
  - GPU memory was released after process exit.
- Output path: `experiments/dry_runs/jointconnv2_100step_stability/`
- Status: success
- Next step: run a qualitative inference smoke test from `jointconn_v2_addons_step_000100.safetensors` for `depth_to_image` and `joint_generation`, then start baseline reproduction and ablation planning.

## Experiment ID: 20260604_jointconnv2_inference_smoke_step100_seed42

- Date: 2026-06-04
- Goal: Verify qualitative inference paths from the 100-step JointConn-v2 adapter checkpoint.
- Code version: local workspace, not a Git repository
- Environment: `C:\Users\zhao\Documents\New project\.venvs\swiftedit\Scripts\python.exe`, PyTorch `2.2.1+cu118`, RTX 4090 24GB
- Checkpoint: `experiments/dry_runs/jointconnv2_100step_stability/jointconn_v2_addons_step_000100.safetensors`
- Commands:
  - `python inference.py --gen_type depth_to_image --text_prompt flower --input_depth experiments/dry_runs/depth_jointconnv2_step100.npy --seed 42 --blocks_to_swap 12 --mixed_precision bf16 --full_bf16 --sample_steps 4`
  - `python inference.py --gen_type joint_generation --text_prompt castle --output_resolution 512 512 --seed 42 --blocks_to_swap 12 --mixed_precision bf16 --full_bf16 --sample_steps 4`
- Config: JointConn-v2 config loaded from the step-100 JSON sidecar; 4-step qualitative smoke inference
- Resource estimate: one GPU, small PNG outputs
- Actual runtime: text encoding plus model load and 4 denoise steps per mode; denoise took about 16 seconds per run
- Actual disk usage: four PNG outputs plus logs
- Main metric: both enabled v2 inference modes complete and save non-empty image/depth PNGs
- Auxiliary metrics: GPU memory release after process exit, visual sanity check
- Results:
  - `depth_to_image` completed successfully and saved image/depth outputs.
  - `joint_generation` completed successfully and saved image/depth outputs.
  - Visual inspection passed for both image/depth pairs.
  - GPU memory was released after each process exit.
- Output paths:
  - `outputs/depth_to_image/image_depth_jointconnv2_step100.png`
  - `outputs/depth_to_image/depth_depth_jointconnv2_step100.png`
  - `outputs/joint_generation/image_20260604_225206.png`
  - `outputs/joint_generation/depth_20260604_225206.png`
  - `experiments/dry_runs/jointconnv2_depth_to_image_smoke_*.log`
  - `experiments/dry_runs/jointconnv2_joint_generation_smoke_*.log`
- Status: success
- Fixes made during execution:
  - Reordered inference to encode text before loading JointConn-v2, then release CLIP-L/T5-XXL to avoid a 24GB VRAM stall.
  - Added `--sample_steps` for controlled inference smoke tests.
  - Seeded inference noise from `args.seed` inside both generation paths.
- Next step: define a fixed qualitative/quantitative validation set, then run old JointDiT baseline and JointConn-v2 ablation variants on the same samples.

## Experiment ID: 20260604_fixed_validation_baseline_v2_smoke_seed42

- Date: 2026-06-04
- Goal: Create an initial fixed validation set and run original JointDiT vs JointConn-v2 on the same samples.
- Code version: local workspace, not a Git repository
- Environment: `C:\Users\zhao\Documents\New project\.venvs\swiftedit\Scripts\python.exe`, PyTorch `2.2.1+cu118`, RTX 4090 24GB
- Fixed set: `experiments/validation/fixed_set/manifest.jsonl`
- Methods:
  - `JointDiT-original`: `models/jointdit/jointdit_addons.safetensors`
  - `JointConn-v2-step100`: `experiments/dry_runs/jointconnv2_100step_stability/jointconn_v2_addons_step_000100.safetensors`
- Commands: see `experiments/baseline/commands.md`; v2 commands used the same samples with `--inference_output_dir experiments/qualitative/jointconnv2_step100_smoke_20260604/outputs`
- Config: seed `42`, resolution `512x512`, `sample_steps=4`, `guidance_scale=3.5`, block swap enabled
- Resource estimate: one GPU, small PNG outputs and logs
- Actual runtime: about 1-2 minutes per inference process depending on checkpoint and block-swap loading
- Actual disk usage: PNG outputs, logs, `metrics.csv`, `summary.json`, and one comparison grid
- Main metric: both methods complete both fixed samples and save non-empty RGB/depth PNGs
- Auxiliary metrics: simple image/depth edge density and visual sanity check
- Results:
  - Initial fixed set created with `d2i_flower` and `joint_castle`.
  - Original JointDiT baseline completed `depth_to_image` and `joint_generation`.
  - JointConn-v2 step100 completed `depth_to_image` and `joint_generation`.
  - All outputs are non-empty.
  - `d2i_flower`: v2 RGB is visibly blurrier than the original baseline, consistent with low RGB edge density.
  - `joint_castle`: v2 produces clearer castle structure and a more detailed depth map than the baseline under the same 4-step smoke setting.
- Output paths:
  - `experiments/baseline/jointdit_original_smoke_20260604/`
  - `experiments/qualitative/jointconnv2_step100_smoke_20260604/`
  - `experiments/qualitative/jointconnv2_step100_smoke_20260604/figures/smoke_comparison_grid.png`
- Status: success for smoke comparison; not yet official paper baseline
- Fixes made during execution:
  - Added `--inference_output_dir` to avoid output overwrites.
  - Changed inference loading to load JointConn-v2 weights on CPU first, then move only non-swapped model parts to GPU for stable 24GB inference.
- Next step: ask user to confirm the smoke baseline direction, then run longer-step baseline/v2 qualitative comparison or train v2 longer before ablation.

## Experiment ID: 20260604_code_split_jointdit_vs_jointconnv2

- Date: 2026-06-04
- Goal: Separate and classify original JointDiT reference files from modified JointConn-v2 implementation files.
- Code version: local workspace, not a Git repository
- Environment: Windows PowerShell, local filesystem operation only
- Dataset / samples: not applicable
- Command: copied selected source/documentation files into `code_split/20260604_jointdit_vs_jointconnv2/`
- Config: non-destructive copy-based archive; root working project remains runnable
- Resource estimate: small source/document copies; no large model/data copy
- Actual runtime: under one minute
- Actual disk usage: small code/document archive; large checkpoints and datasets intentionally not copied
- Main metric: original and modified files are separated into clear archive groups
- Auxiliary metrics: classification manifests and separation docs exist
- Results:
  - Original reference files copied to `code_split/20260604_jointdit_vs_jointconnv2/original_jointdit/`.
  - JointConn-v2 modified/added files copied to `code_split/20260604_jointdit_vs_jointconnv2/jointconn_v2_modified/`.
  - Classification manifests created under `code_split/20260604_jointdit_vs_jointconnv2/manifests/`.
  - Project-level separation map created at `docs/project_code_separation_map.md`.
  - Large checkpoints, datasets, and model weights were not copied.
- Output path: `code_split/20260604_jointdit_vs_jointconnv2/`
- Status: success
- Failure reason: not applicable
- Fixes made during execution: none
- Next step: keep running code from the root workspace; use the split archive for review, cleanup, or future standalone packaging.

## Experiment ID: 20260604_remove_local_jointdit_run_scripts

- Date: 2026-06-04
- Goal: Remove original JointDiT run scripts from the current JointConn-v2 workspace and register the external JointDiT baseline path.
- Code version: local workspace, not a Git repository
- Environment: Windows PowerShell, local filesystem/documentation operation only
- Dataset / samples: not applicable
- Command: deleted `scripts/joint_generation.sh`, `scripts/depth_to_image.sh`, and `scripts/depth_estimation.sh`
- Config: external JointDiT baseline repository is `F:\Phd_Work\Neural Networks and ICML 2026\JointDiT-main`
- Resource estimate: negligible
- Actual runtime: under one minute
- Actual disk usage: negligible
- Main metric: current project no longer contains original JointDiT run scripts
- Auxiliary metrics: baseline docs point future JointDiT comparisons to the external repository
- Results:
  - Removed original JointDiT run scripts from `D:\code\JointConn-v2\scripts\`.
  - Kept `scripts/smoke_jointconn_v2.py` for JointConn-v2 validation.
  - Confirmed external JointDiT repository exists.
  - Added `docs/external_jointdit_baseline_reference.md`.
  - Updated baseline and follow-up docs to use the external JointDiT path.
- Output path:
  - `docs/external_jointdit_baseline_reference.md`
  - `experiments/baseline/design.md`
  - `experiments/baseline/commands.md`
  - `docs/jointconn_v2_followup_execution_design.md`
- Status: success
- Failure reason: not applicable
- Fixes made during execution: none
- Next step: when running baseline comparisons, execute original JointDiT from `F:\Phd_Work\Neural Networks and ICML 2026\JointDiT-main` and copy outputs into the current project's `experiments/baseline/` directory.

## Experiment ID: 20260604_jointconn_v2_project_rename_cleanup

- Date: 2026-06-04
- Goal: Rename current project-facing code, scripts, checkpoints, and docs from legacy JointDiT names to JointConn-v2 names while preserving the external JointDiT baseline identity.
- Changes:
  - Renamed active package to `jointconn_v2_library/`.
  - Renamed active modules to `jointconn_v2_model.py`, `jointconn_v2_utils.py`, and `dataset/jointconn_v2_dataset.py`.
  - Renamed inference argument to `--jointconn_v2_addons_path`.
  - Renamed future and existing v2 checkpoint files to `jointconn_v2_addons_step_*`.
  - Moved local original baseline weights to `models/original_baseline/`.
- Verification:
  - `python -m py_compile ...` passed with the default Python.
  - `C:\Users\zhao\Documents\New project\.venvs\swiftedit\Scripts\python.exe -m py_compile ...` passed.
  - `C:\Users\zhao\Documents\New project\.venvs\swiftedit\Scripts\python.exe scripts\smoke_jointconn_v2.py` passed.
  - `inference.py --help` exposes `--jointconn_v2_addons_path` and no longer exposes the old addon argument.
