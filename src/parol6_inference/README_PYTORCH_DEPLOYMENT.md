# PyTorch Policy Deployment Guide

This guide explains how to use the `policy.pt` (TorchScript) file from Isaac Lab training in the ROS2 Gazebo simulation.

## Overview

The ROS2 inference system now supports using the `policy.pt` file exported by Isaac Lab's `play.py` script, which provides better compatibility with the training environment.

## Key Changes

### 1. Policy File Format

- **Previously**: Used ONNX format (`.onnx`)
- **Now**: Uses TorchScript format (`.pt`) exported by Isaac Lab's `export_policy_as_jit()`

### 2. Observation Normalization

**TorchScript models** (`.pt` files):
- Have **embedded normalization** - no separate `_norm.npz` file needed
- Normalization is applied automatically inside the model

**ONNX models** (`.onnx` files):
- Require external `_norm.npz` file for observation normalization
- Normalization is applied before inference

### 3. Action Output Handling

**Critical Fix Applied**: TorchScript exports from Isaac Lab don't include the final `tanh` activation layer, so we apply it manually in the inference code:

```python
action_tensor = torch.tanh(action_tensor)  # Ensures actions are in [-1, 1]
```

This ensures policy outputs are bounded to `[-1, 1]` range before being denormalized to joint positions.

## How to Use

### Step 1: Export Policy from Isaac Lab

Run the training play script to export the policy:

```bash
cd /home/gunesh_pop_nvidia/PAROL6_Lab
python parol6/scripts/rsl_rl/play.py --task Isaac-Parol6-Reach-Position-v0
```

This creates:
- `logs/rsl_rl/parol6_reach_position/<experiment_name>/exported/policy.pt`
- `logs/rsl_rl/parol6_reach_position/<experiment_name>/exported/policy.onnx`

### Step 2: Copy Policy to ROS2 Workspace

Copy the `policy.pt` file to the ROS2 workspace:

```bash
cp /home/gunesh_pop_nvidia/PAROL6_Lab/parol6/logs/rsl_rl/parol6_reach_position/<experiment_name>/exported/policy.pt \
   /home/gunesh_pop_nvidia/parol6_gz_ws/src/parol6_inference/policy/policy.pt
```

Replace `<experiment_name>` with your training run (e.g., `test7_final`).

### Step 3: Launch ROS2 Inference

The default launch configuration now uses `policy.pt`:

```bash
cd /home/gunesh_pop_nvidia/parol6_gz_ws
ros2 launch parol6_inference parol6_inference.launch.py
```

To use a different policy file or type:

```bash
ros2 launch parol6_inference parol6_inference.launch.py \
  policy_path:=/path/to/your/policy.pt \
  policy_type:=torchscript
```

## Policy Type Options

| Policy Type | File Extension | Normalization | Use Case |
|------------|----------------|---------------|----------|
| `torchscript` | `.pt`, `.ts` | Embedded | **Recommended** - Direct export from Isaac Lab |
| `onnx` | `.onnx` | External (`.npz`) | Alternative format, faster inference |
| `pytorch` | `.pth` | External (`.npz`) | Raw PyTorch state dicts (advanced) |

## Troubleshooting

### Issue: Robot makes erratic movements

**Cause**: Policy outputs not in `[-1, 1]` range

**Solution**: Ensure you're using `policy_type:=torchscript` for `.pt` files. The code automatically applies `tanh` activation.

### Issue: Policy not converging to target

**Possible causes**:
1. **Observation mismatch** - Check that ROS2 observations match Isaac Lab format (19D for position task)
2. **Joint limits mismatch** - Verify joint limits in `policy_inference.py:43-44` match Isaac Lab config
3. **Action smoothing** - The ROS2 system uses EMA smoothing (alpha=0.3) matching Isaac Lab

### Issue: TorchScript model not loading

**Error**: `torch not installed`

**Solution**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Technical Details

### Observation Format (19D)

```
[0:6]   - Joint positions (radians)
[6:12]  - Joint velocities (rad/s)
[12:15] - End-effector position relative to base (meters)
[15:18] - Target position relative to base (meters)
[18]    - Distance to target (meters)
```

### Action Processing Pipeline

1. **Policy Output**: Raw values (unbounded)
2. **Tanh Activation**: Bound to `[-1, 1]` range
3. **Denormalization**: Map to joint limits `[lower, upper]`
4. **EMA Smoothing**: Apply exponential moving average (optional, enabled by default)
5. **Command**: Send to joint controller

### Joint Limits (PAROL6)

```python
lower = [-1.083, -1.221, -1.8825, -1.841, -1.571, -3.142]  # radians
upper = [ 2.148,  0.907,  1.2566,  1.841,  1.571,  3.142]  # radians
```

## Comparison: ONNX vs TorchScript

| Feature | ONNX | TorchScript |
|---------|------|-------------|
| Inference Speed | Faster | Slightly slower |
| Normalization | External file | Embedded |
| Compatibility | Better cross-platform | PyTorch only |
| Debugging | Harder | Easier |
| **Recommended** | Production | **Development & Direct from Isaac Lab** |

## Files Modified

1. `policy_inference.py` - Updated to handle TorchScript models with tanh activation
2. `parol6_inference.launch.py` - Changed defaults to use `policy.pt` and `torchscript` type

## References

- Isaac Lab policy export: `isaaclab_rl/rsl_rl/export_policy_as_jit()`
- ROS2 inference node: `parol6_inference/policy_inference.py`
- Environment config: `parol6/tasks/direct/parol6/parol6_env_cfg.py`
