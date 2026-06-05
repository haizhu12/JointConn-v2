# Project Code Separation Map

## Purpose

The current workspace contains both original JointDiT assets and modified JointConn-v2 paper code. They have been separated into a non-destructive classification archive:

```text
code_split/20260604_jointdit_vs_jointconnv2/
```

The working project root remains runnable. No runtime files were moved.

Update: the original JointDiT run scripts have been removed from the current workspace. Future JointDiT baseline runs should use:

```text
F:\Phd_Work\Neural Networks and ICML 2026\JointDiT-main
```

## Archive Structure

```text
code_split/20260604_jointdit_vs_jointconnv2/
  README.md
  original_jointdit/
  jointconn_v2_modified/
  manifests/
    original_jointdit_files.md
    jointconn_v2_modified_files.md
```

## Original JointDiT Group

Original or unchanged reference files are copied under:

```text
code_split/20260604_jointdit_vs_jointconnv2/original_jointdit/
```

This includes:

- original README/config/requirements,
- archived copies of the original shell inference scripts,
- original dataset preprocessing helpers,
- original LoRA adapter helper,
- original JointDiT addon JSON metadata.

Large original checkpoints and datasets are not copied.

The archived shell scripts are for reference only. They should not be restored into the current JointConn-v2 project unless the user explicitly requests it.

## JointConn-v2 Modified Group

Modified and new files are copied under:

```text
code_split/20260604_jointdit_vs_jointconnv2/jointconn_v2_modified/
```

This includes:

- JointConn-v2 modules,
- modified model and utility code,
- modified training and inference scripts,
- v2 dataset/offloading fixes,
- v2 smoke test,
- method/design/follow-up docs,
- experiment planning files.

## Why This Is Non-Destructive

The root code is still the active development copy. The split folder is a classified snapshot for handoff, review, and future cleanup.

Do not run training directly from the split archive unless it is later promoted into a standalone package.

## Next Cleanup Option

If a full clean project is needed later, create a new runnable project folder from `jointconn_v2_modified/` plus required original dependencies, then restore imports and add a dedicated README.
