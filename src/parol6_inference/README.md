# PAROL6 Inference Package

ROS2 package for running Isaac Lab trained policies on the PAROL6 robot in Gazebo.

## Overview

This package provides three main nodes:

1. **observation_publisher.py**: Collects robot observations and publishes them
   - Subscribes to `/joint_states`
   - Uses TF2 to get end-effector position
   - Publishes 19D observation vector to `/parol6/observations`
   - Supports dynamic goal updates via `/target_position` topic

2. **policy_inference.py**: Runs trained policy and publishes actions
   - Subscribes to `/parol6/observations`
   - Runs policy inference (ONNX/PyTorch/TorchScript)
   - Publishes joint commands to `/joint_trajectory_controller/joint_trajectory`

3. **interactive_goal_setter.py**: Provides visual goal setting in RViz
   - Creates draggable red sphere marker for target position
   - Publishes goal updates to `/target_position`
   - Enables real-time interactive control

## Installation

### 1. Install Python Dependencies

```bash
# For ONNX support (recommended)
pip install onnxruntime-gpu  # or onnxruntime for CPU only

# For PyTorch support (optional)
pip install torch torchvision

# For NumPy (required)
pip install numpy
```

### 2. Build the Package

```bash
cd ~/parol6_gz_ws
colcon build --packages-select parol6_inference
source install/setup.bash
```

## Usage

### Running Observation Publisher Only (Testing)

Test that observations are being published correctly:

```bash
ros2 launch parol6_inference parol6_inference.launch.py run_inference:=false
```

This will:
- Publish observations to `/parol6/observations`
- Publish EE position to `/parol6/ee_position`
- NOT run policy inference

**Check observations:**
```bash
# View observation messages
ros2 topic echo /parol6/observations

# View EE position
ros2 topic echo /parol6/ee_position
```

### Running Full Inference System

Run with a trained policy:

```bash
ros2 launch parol6_inference parol6_inference.launch.py \
    policy_path:=/path/to/your/policy.onnx \
    policy_type:=onnx \
    target_x:=0.20 \
    target_y:=0.03 \
    target_z:=0.24
```

**Policy formats supported:**
- ONNX: `policy_type:=onnx` (`.onnx` files)
- PyTorch: `policy_type:=pytorch` (`.pt`, `.pth` files)
- TorchScript: `policy_type:=torchscript` (`.ts` files)

### Setting Dynamic Goals

#### Method 1: Using Interactive RViz Markers (Recommended)

The easiest way to set goals is using the interactive marker in RViz:

```bash
# Launch RViz with the PAROL6 configuration
rviz2 -d ~/parol6_gz_ws/src/parol6_description/rviz/display.rviz
```

You'll see a **RED SPHERE** marker at the target position. Simply:
1. Click and drag the sphere to move it in 3D space
2. Use the colored arrows (X=red, Y=green, Z=blue) for precise axis-aligned movements
3. The robot will automatically track to the new goal position!

The interactive marker node launches by default. To disable it:
```bash
ros2 launch parol6_inference parol6_inference.launch.py \
    use_interactive_marker:=false \
    policy_path:=/path/to/policy.onnx
```

#### Method 2: Using ROS2 Topic

Publish new target positions programmatically:

```bash
# Set new goal via topic
ros2 topic pub --once /target_position geometry_msgs/msg/Point "{x: 0.25, y: 0.0, z: 0.30}"
```

The robot will automatically move to the new target!

## Launch File Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `policy_path` | '' | Path to trained policy file |
| `policy_type` | 'onnx' | Policy format (onnx/pytorch/torchscript) |
| `target_x` | 0.20 | Target X position (meters) |
| `target_y` | 0.03 | Target Y position (meters) |
| `target_z` | 0.24 | Target Z position (meters) |
| `base_frame` | 'base_link' | Robot base TF frame |
| `ee_frame` | 'L6' | End-effector TF frame |
| `publish_rate` | 100.0 | Observation publishing rate (Hz) |
| `control_rate` | 100.0 | Policy inference rate (Hz) |
| `use_ema_smoothing` | true | Enable action smoothing |
| `ema_alpha` | 0.3 | Smoothing factor (0.0-1.0) |
| `run_inference` | true | Enable policy inference |
| `use_interactive_marker` | true | Enable interactive RViz marker |
| `marker_scale` | 0.05 | Interactive marker sphere size (meters) |

## Topics

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Robot joint positions and velocities |
| `/target_position` | `geometry_msgs/Point` | Dynamic target position updates |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/parol6/observations` | `std_msgs/Float32MultiArray` | 19D observation vector |
| `/parol6/ee_position` | `geometry_msgs/Point` | End-effector position (debug) |
| `/joint_trajectory_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | Joint commands |

## Observation Vector Format (19D)

```
[0:6]   - Joint positions (J1-J6) in radians
[6:12]  - Joint velocities (J1-J6) in rad/s
[12:15] - End-effector position relative to base (x, y, z) in meters
[15:18] - Target position relative to base (x, y, z) in meters
[18]    - Euclidean distance to target in meters
```

## Exporting Isaac Lab Policy to ONNX

To export your trained Isaac Lab policy:

```python
import torch
import onnx

# Load your trained policy
policy = ... # Your trained policy

# Create dummy input (19D observation)
dummy_input = torch.randn(1, 19)

# Export to ONNX
torch.onnx.export(
    policy,
    dummy_input,
    "policy.onnx",
    input_names=['obs'],
    output_names=['action'],
    dynamic_axes={'obs': {0: 'batch'}, 'action': {0: 'batch'}}
)

print("Policy exported to policy.onnx")
```

## Troubleshooting

### TF Lookup Fails

**Problem:** "Could not transform" warnings

**Solutions:**
1. Check TF tree: `ros2 run tf2_tools view_frames`
2. Verify frame names match your URDF:
   ```bash
   ros2 topic echo /tf_static
   ```
3. Update launch file parameters: `base_frame`, `ee_frame`

### No Joint States

**Problem:** Observations not publishing

**Solutions:**
1. Check if Gazebo is publishing joint states:
   ```bash
   ros2 topic echo /joint_states
   ```
2. Verify joint names match: `['J1', 'J2', 'J3', 'J4', 'J5', 'J6']`

### Policy Not Loading

**Problem:** "Failed to load policy"

**Solutions:**
1. Check file path: `ls -la /path/to/policy.onnx`
2. Install dependencies:
   - ONNX: `pip install onnxruntime-gpu`
   - PyTorch: `pip install torch`
3. Verify policy format matches file extension

### Robot Not Moving

**Problem:** Observations published but robot stationary

**Solutions:**
1. Check controller is running:
   ```bash
   ros2 control list_controllers
   ```
2. Verify topic remapping:
   ```bash
   ros2 topic list | grep trajectory
   ```
3. Check if inference node is running:
   ```bash
   ros2 node list
   ```

## Example Workflow

### Complete workflow from Isaac Lab training to ROS2 deployment:

```bash
# 1. In Isaac Lab: Train policy
cd IsaacLab
python scripts/rsl_rl/train.py --task Parol6-Reach-Position-Direct-v0

# 2. Export policy to ONNX
python export_policy.py --checkpoint logs/.../model_1700.pt --output policy.onnx

# 3. In ROS2: Launch Gazebo with PAROL6
ros2 launch parol6_bringup parol6_gazebo.launch.py

# 4. Launch inference system (in new terminal)
ros2 launch parol6_inference parol6_inference.launch.py \
    policy_path:=/path/to/policy.onnx \
    target_x:=0.20 target_y:=0.03 target_z:=0.24

# 5. Launch RViz for visualization (in new terminal)
rviz2 -d ~/parol6_gz_ws/src/parol6_description/rviz/display.rviz

# 6. Interact with the robot!
# - Drag the RED SPHERE marker in RViz to set new goals
# - Or publish goals via topic:
ros2 topic pub --once /target_position geometry_msgs/msg/Point "{x: 0.15, y: -0.05, z: 0.20}"
```

## Interactive Marker Node

The `interactive_goal_setter.py` node provides visual goal setting through RViz:

**Features:**
- Draggable red sphere marker at target position
- 3D movement controls (X/Y/Z axes)
- Real-time goal updates published to `/target_position`
- Integrates seamlessly with observation publisher

**Usage:**
```bash
# Launch with default settings
ros2 launch parol6_inference parol6_inference.launch.py \
    policy_path:=/path/to/policy.onnx

# Customize marker appearance
ros2 launch parol6_inference parol6_inference.launch.py \
    policy_path:=/path/to/policy.onnx \
    marker_scale:=0.08 \
    target_x:=0.25 target_y:=0.0 target_z:=0.30

# Run standalone (without policy inference)
ros2 run parol6_inference interactive_goal_setter.py \
    --ros-args \
    -p target_x:=0.20 \
    -p target_y:=0.03 \
    -p target_z:=0.24
```

**In RViz:**
1. The red sphere appears at the initial target position
2. Click and drag the sphere to move it freely in 3D
3. Use the colored arrows for precise axis-aligned movements:
   - **Red arrow** = X-axis
   - **Green arrow** = Y-axis
   - **Blue arrow** = Z-axis
4. The robot updates its goal in real-time as you move the marker!

## Performance Tips

1. **GPU Acceleration**: Use `onnxruntime-gpu` for faster inference
2. **Control Rate**: Match Isaac Lab training rate (100-120 Hz recommended)
3. **Smoothing**: Enable EMA smoothing to reduce jitter
4. **Batch Size**: Inference runs with batch_size=1 for real-time

## License

MIT

## Author

Gunesh Gupta (guneshguptag@gmail.com)

You can test moving the robot joints by publishing a JointTrajectory message to the /arm_controller/joint_trajectory topic. Here's a command to move all joints to a specific configuration:
```
  ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
    joint_names: ['J1', 'J2', 'J3', 'J4', 'J5', 'J6'],
    points: [
      {
        positions: [0.0, -0.5, 0.8, 0.0, 0.5, 0.0],
        velocities: [],
        accelerations: [],
        time_from_start: {sec: 2, nanosec: 0}
      }
    ]
  }"
```
  This command will move the robot to a reasonable reaching pose over 2 seconds.

  For a smoother test with multiple waypoints:
```
  ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
    joint_names: ['J1', 'J2', 'J3', 'J4', 'J5', 'J6'],
    points: [
      {
        positions: [0.5, -0.3, 0.5, 0.0, 0.3, 0.0],
        time_from_start: {sec: 3, nanosec: 0}
      },
      {
        positions: [-0.5, -0.5, 0.8, 0.0, 0.5, 0.0],
        time_from_start: {sec: 6, nanosec: 0}
      }
    ]
  }"
```
  To return to home position (all zeros):
```
  ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
    joint_names: ['J1', 'J2', 'J3', 'J4', 'J5', 'J6'],
    points: [
      {
        positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        time_from_start: {sec: 3, nanosec: 0}
      }
    ]
  }"
```