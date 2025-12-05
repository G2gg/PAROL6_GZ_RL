# Sim-to-Sim Transfer Debugging Guide

## Problem Summary

The policy works well in Isaac Lab but doesn't converge to the target in Gazebo. The robot moves but doesn't reach and hold the desired position.

## Root Causes (Sim-to-Sim Gap)

When transferring a policy from Isaac Lab (Isaac Sim) to Gazebo, several factors can prevent proper convergence:

### 1. **Physics Engine Differences** ⚠️ **MAJOR IMPACT**

| Aspect | Isaac Sim (PhysX) | Gazebo (ODE/Bullet) | Impact |
|--------|-------------------|---------------------|--------|
| Solver | PhysX GPU solver | CPU-based ODE/Bullet | Different joint dynamics |
| Timestep accuracy | Very stable (GPU) | Can drift | Position drift |
| Contact dynamics | High-fidelity | Approximated | Different contact behavior |
| Friction model | Advanced | Basic | Different surface interactions |

**Why this matters**: The policy learned control strategies based on PhysX dynamics. In Gazebo, the same commands produce different movements.

### 2. **Joint Velocity Noise** ⚠️ **HIGH IMPACT**

**Isaac Lab**: Clean, low-noise velocity measurements from PhysX
**Gazebo**: Often has noisy velocity measurements, especially when using finite differencing

**Example**:
- Isaac Lab velocity noise: ~0.01 rad/s
- Gazebo velocity noise: ~1-5 rad/s (can be 100x higher!)

**Why this matters**: The policy uses velocities [6:12] in observations. High noise confuses the policy.

### 3. **Actuator Model Differences** ⚠️ **MAJOR IMPACT**

**Isaac Lab** (from parol6_robot.py):
```python
ImplicitActuatorCfg(
    stiffness={"J1": 65.0, "J2": 512.5, "J3": 340.0, ...},
    damping={"J1": 3.25, "J2": 25.625, ...}  # 5% of stiffness
)
```
- Uses PD control: `τ = K*(q_target - q) - D*q_dot`
- Smooth, gradual movement to targets
- Natural compliance

**Gazebo**:
```
joint_trajectory_controller with trajectory_time
```
- Tracks trajectory with spline interpolation
- Different control law
- Less compliant, more aggressive

### 4. **Observation Distribution Mismatch** ⚠️ **MODERATE IMPACT**

The TorchScript model's **embedded normalization** was computed from Isaac Lab training data:
- Joint position ranges
- Joint velocity ranges
- End-effector position distribution
- Target position distribution

In Gazebo, these distributions might be different:
- Different joint limit enforcement
- Different velocity profiles
- Different coordinate frame conventions
- Different noise characteristics

**Why this matters**: If Gazebo observations have different statistics, the normalized observations will be out-of-distribution for the policy.

### 5. **Coordinate Frame Conventions** ⚠️ **MODERATE IMPACT**

Potential differences:
- Base frame location/orientation
- End-effector frame definition
- TF tree structure
- Frame update rates

### 6. **Control Loop Differences** ⚠️ **MODERATE IMPACT**

| Parameter | Isaac Lab | Gazebo | Status |
|-----------|-----------|---------|---------|
| Policy rate | 120 Hz | 100 Hz | ✓ Acceptable |
| Physics timestep | 1/120 s | Variable | ⚠ Different |
| Actuator response | Smooth PD | Trajectory interp | ⚠ Different |
| Smoothing | EMA (alpha=0.3) | EMA (alpha=0.3) | ✓ Matched |

## Diagnostic Steps

### Step 1: Rebuild and Run Diagnostic Tool

```bash
cd /home/gunesh_pop_nvidia/parol6_gz_ws
colcon build --packages-select parol6_inference
source install/setup.bash

# Run the diagnostic in a new terminal while simulation is running
ros2 run parol6_inference diagnose_observations.py
```

This will show you:
- Observation statistics (mean, std, min, max)
- Comparison with expected Isaac Lab ranges
- Potential issues (velocity noise, joint limit violations, etc.)

### Step 2: Check for Common Issues

Run the diagnostic and look for these warning signs:

#### High Velocity Noise
```
⚠ J2 velocity has high noise (std=4.523 rad/s). This may confuse the policy!
```

**Fix**: Filter velocities or use position-only observations

#### Joint Limit Violations
```
⚠ J1 exceeds expected limits! Observed: [-1.200, 2.300], Expected: [-1.083, 2.148]
```

**Fix**: Check URDF joint limits match Isaac Lab config

#### Unrealistic End-Effector Position
```
⚠ End-effector position seems unrealistic! Distance from base: 0.85m
```

**Fix**: Check TF frames and robot model

#### Non-Converging Distance
```
⚠ Robot not converging to target! Average distance: 0.245m
```

**Fix**: Address root causes below

### Step 3: Compare with Isaac Lab

Create a simple test in Isaac Lab:
1. Run play.py in Isaac Lab
2. Record observations from successful episode
3. Compare observation statistics with Gazebo diagnostics

Look for distribution mismatches in:
- Joint velocities (most common issue)
- End-effector positions
- Joint position ranges

## Potential Fixes (Ranked by Impact)

### Fix 1: Reduce Trajectory Time (Increase Smoothness) ✅ DONE

We already increased `trajectory_time` to 0.5s to match Isaac Lab's smooth PD control.

### Fix 2: Filter Joint Velocities (HIGH PRIORITY)

If diagnostic shows high velocity noise, apply filtering:

```python
# In observation_publisher.py, add low-pass filter
from collections import deque

class ObservationPublisher:
    def __init__(self):
        ...
        self.vel_filter = deque(maxlen=5)  # 5-sample moving average

    def joint_state_callback(self, msg):
        ...
        # Apply moving average filter
        self.vel_filter.append(self.joint_vel.copy())
        self.joint_vel = np.mean(self.vel_filter, axis=0)
```

### Fix 3: Adjust Actuator Gains in Gazebo

Match Isaac Lab's PD gains in Gazebo URDF/controller config.

From Isaac Lab:
```yaml
# Add to your gazebo controller config
gains:
  J1: {p: 65.0, d: 3.25}
  J2: {p: 512.5, d: 25.625}
  J3: {p: 340.0, d: 17.0}
  J4: {p: 25.0, d: 1.25}
  J5: {p: 25.0, d: 1.25}
  J6: {p: 22.0, d: 1.1}
```

### Fix 4: Retrain with Domain Randomization

Long-term solution: Retrain policy in Isaac Lab with:
- Randomized actuator gains
- Randomized observation noise
- Randomized physics parameters

This makes the policy robust to sim-to-sim differences.

### Fix 5: Fine-tune in Gazebo

If other fixes don't work:
1. Collect Gazebo data using a simple controller
2. Fine-tune the policy on Gazebo observations
3. Reduces domain gap

## Expected Behavior After Fixes

After applying fixes, you should see:
- ✓ Distance consistently decreasing
- ✓ Robot reaching target (distance < 0.02m)
- ✓ Smooth, controlled movements
- ✓ Holding position at target
- ✓ Policy outputs stable near target

## Quick Test: Position-Only Policy

To test if velocities are the issue, try temporarily using only position observations:

```python
# In policy_inference.py, modify normalize_observation():
def normalize_observation(self, observation):
    # TEMPORARY TEST: Zero out velocities
    obs_modified = observation.copy()
    obs_modified[6:12] = 0.0  # Zero velocities
    return obs_modified if self.policy_type == 'torchscript' else ...
```

If this improves convergence, velocity noise is the culprit.

## Contact for Help

If issues persist after diagnostics:
1. Share diagnostic output
2. Share example observation logs
3. Share video of robot behavior
4. Compare with Isaac Lab video side-by-side
