#!/usr/bin/env python3
"""
Create default observation normalization statistics for PAROL6 reaching task.

Based on typical ranges observed during training.
"""

import numpy as np

# Observation vector (19D):
# [0:6]   - Joint positions (radians)
# [6:12]  - Joint velocities (rad/s)
# [12:15] - End-effector position (meters)
# [15:18] - Target position (meters)
# [18]    - Distance to target (meters)

# Reasonable defaults based on PAROL6 workspace and dynamics
obs_mean = np.array([
    # Joint positions (mean around 0, but varies per joint)
    0.0, -0.2, 0.2, 0.0, 0.0, 0.0,
    # Joint velocities (mean around 0)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # EE position (mean around robot workspace center)
    0.20, 0.0, 0.30,
    # Target position (mean around workspace center)
    0.20, 0.0, 0.30,
    # Distance (mean around 0.15m)
    0.15
], dtype=np.float32)

obs_std = np.array([
    # Joint positions std (based on joint limits range / 4)
    0.8, 0.5, 0.8, 0.9, 0.8, 1.5,
    # Joint velocities std (typical velocities during reaching)
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    # EE position std (workspace spans ~0.4m in each dimension)
    0.10, 0.10, 0.10,
    # Target position std (similar to EE)
    0.10, 0.10, 0.10,
    # Distance std
    0.10
], dtype=np.float32)

# Save
output_path = 'policy_norm.npz'
np.savez(output_path, mean=obs_mean, std=obs_std)

print("Created default observation normalization:")
print(f"  Mean shape: {obs_mean.shape}")
print(f"  Std shape: {obs_std.shape}")
print(f"\nMean (first 10): {obs_mean[:10]}")
print(f"Std (first 10): {obs_std[:10]}")
print(f"\n✓ Saved to: {output_path}")
print("\nNOTE: These are default values. For best performance,")
print("      collect actual statistics from your robot or Isaac Lab.")
