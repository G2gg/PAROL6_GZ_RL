#!/usr/bin/env python3
"""
Test the policy directly with sample observations to verify it's working correctly.

This helps determine if the policy is the problem or if it's the observations/actions.
"""

import torch
import numpy as np
import sys

def test_policy(policy_path):
    """Test policy with sample observations."""

    print("="*80)
    print("POLICY DIRECT TEST")
    print("="*80)

    # Load policy
    print(f"\nLoading policy from: {policy_path}")
    try:
        policy = torch.jit.load(policy_path)
        policy.eval()
        print("✓ Policy loaded successfully (TorchScript)")
    except Exception as e:
        print(f"✗ Failed to load policy: {e}")
        return

    # Test 1: Zero observation (robot at home position)
    print("\n" + "="*80)
    print("TEST 1: Zero Observation (Home Position)")
    print("="*80)

    obs_zero = np.zeros(19, dtype=np.float32)
    obs_zero[15:18] = [0.2, 0.03, 0.24]  # Target position
    obs_zero[18] = 0.3  # Distance

    print(f"Observation: {obs_zero}")

    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs_zero.reshape(1, -1))
        action = policy(obs_tensor).numpy()[0]
        action_tanh = np.tanh(action)

    print(f"Policy output (raw): {action}")
    print(f"Policy output (tanh): {action_tanh}")
    print(f"Output range: [{action_tanh.min():.3f}, {action_tanh.max():.3f}]")

    if action_tanh.min() >= -1.0 and action_tanh.max() <= 1.0:
        print("✓ Output in valid range [-1, 1]")
    else:
        print("⚠ Output outside [-1, 1] range!")

    # Test 2: Same observation 10 times (determinism check)
    print("\n" + "="*80)
    print("TEST 2: Determinism Check (Same Input → Same Output?)")
    print("="*80)

    outputs = []
    for i in range(10):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_zero.reshape(1, -1))
            action = policy(obs_tensor).numpy()[0]
            action_tanh = np.tanh(action)
            outputs.append(action_tanh)

    outputs = np.array(outputs)
    std_dev = np.std(outputs, axis=0)

    print(f"Standard deviation across 10 runs: {std_dev}")
    print(f"Max std dev: {std_dev.max():.6f}")

    if std_dev.max() < 1e-6:
        print("✓ Policy is deterministic")
    else:
        print("⚠ Policy is stochastic or has randomness!")

    # Test 3: Random observations (policy sensitivity)
    print("\n" + "="*80)
    print("TEST 3: Policy Sensitivity (Random Observations)")
    print("="*80)

    np.random.seed(42)
    actions_list = []

    for i in range(5):
        # Random observation
        obs_random = np.random.randn(19).astype(np.float32) * 0.5
        obs_random[15:18] = [0.2, 0.03, 0.24]  # Keep target constant
        obs_random[18] = np.random.uniform(0.05, 0.3)  # Random distance

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_random.reshape(1, -1))
            action = policy(obs_tensor).numpy()[0]
            action_tanh = np.tanh(action)
            actions_list.append(action_tanh)

        print(f"  Obs {i+1} → Action: {action_tanh}")

    actions_array = np.array(actions_list)
    action_std = np.std(actions_array, axis=0)

    print(f"\nAction std dev across random obs: {action_std}")
    print(f"Mean action change: {action_std.mean():.3f}")

    if action_std.mean() > 0.01:
        print("✓ Policy is sensitive to observations")
    else:
        print("⚠ Policy outputs similar actions regardless of input!")

    # Test 4: Isaac Lab-like observation
    print("\n" + "="*80)
    print("TEST 4: Isaac Lab-like Observation")
    print("="*80)

    # Typical observation from successful Isaac Lab episode
    obs_isaaclab = np.array([
        # Joint positions (near home)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # Joint velocities (small)
        0.01, -0.01, 0.0, 0.0, 0.01, 0.0,
        # EE position (forward and up)
        0.15, 0.0, 0.25,
        # Target position
        0.2, 0.03, 0.24,
        # Distance
        0.06
    ], dtype=np.float32)

    print(f"Observation (Isaac Lab-like):")
    print(f"  Joints: {obs_isaaclab[0:6]}")
    print(f"  Joint vels: {obs_isaaclab[6:12]}")
    print(f"  EE pos: {obs_isaaclab[12:15]}")
    print(f"  Target: {obs_isaaclab[15:18]}")
    print(f"  Distance: {obs_isaaclab[18]}")

    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs_isaaclab.reshape(1, -1))
        action = policy(obs_tensor).numpy()[0]
        action_tanh = np.tanh(action)

    print(f"\nPolicy output: {action_tanh}")

    # Denormalize to joint positions
    joint_limits_lower = np.array([-1.083, -1.221, -1.8825, -1.841, -1.571, -3.142])
    joint_limits_upper = np.array([2.148, 0.907, 1.2566, 1.841, 1.571, 3.142])
    joint_commands = joint_limits_lower + (action_tanh + 1.0) / 2.0 * (joint_limits_upper - joint_limits_lower)

    print(f"Joint commands: {joint_commands}")
    print(f"Joint change from home: {np.linalg.norm(joint_commands - obs_isaaclab[0:6]):.3f} rad")

    # Test 5: Current Gazebo observation
    print("\n" + "="*80)
    print("TEST 5: Paste Your Current Gazebo Observation Here")
    print("="*80)
    print("To test with actual Gazebo observation:")
    print("  ros2 topic echo /parol6/observations --once")
    print("Then paste the 19 values here and re-run this script")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 test_policy_directly.py <policy.pt>")
        print("Example: python3 test_policy_directly.py /path/to/policy.pt")
        sys.exit(1)

    policy_path = sys.argv[1]
    test_policy(policy_path)
