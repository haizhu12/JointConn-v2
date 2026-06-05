# External JointDiT Baseline Reference

## Purpose

The current project is now treated as the JointConn-v2 modified workspace. Original JointDiT baseline/comparison runs should be launched from the separately downloaded full JointDiT repository.

## External JointDiT Path

```text
F:\Phd_Work\Neural Networks and ICML 2026\JointDiT-main
```

## Current Workspace Policy

Original JointDiT run scripts have been removed from:

```text
D:\code\JointConn-v2\scripts\
```

The current scripts directory should keep JointConn-v2-specific utilities only, such as:

```text
scripts/smoke_jointconn_v2.py
```

## Baseline Execution Rule

For future JointDiT baseline comparison:

1. Run original JointDiT from the external repository.
2. Use the same fixed validation inputs from the JointConn-v2 workspace when needed.
3. Copy outputs, logs, metrics, and command records into:

```text
D:\code\JointConn-v2\experiments\baseline\<EXPERIMENT_ID>\
```

Do not reintroduce original JointDiT run scripts into the JointConn-v2 workspace unless explicitly requested.
