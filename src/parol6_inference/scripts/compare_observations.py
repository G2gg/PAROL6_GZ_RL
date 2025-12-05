#!/usr/bin/env python3
"""
Compare Gazebo observations with Isaac Lab expected ranges.
This helps identify observation normalization mismatches.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import sys


class ObservationComparison(Node):
    """Compare observations with Isaac Lab expectations."""

    def __init__(self):
        super().__init__('observation_comparison')

        # Expected ranges from Isaac Lab training (approximate)
        # These are the UNNORMALIZED ranges the policy expects to see
        self.expected_ranges = {
            # Joint positions
            'joint_pos': {
                'min': np.array([-1.083, -1.221, -1.8825, -1.841, -1.571, -3.142]),
                'max': np.array([2.148, 0.907, 1.2566, 1.841, 1.571, 3.142]),
                'typical_std': 0.5  # Typical variation during reaching
            },
            # Joint velocities (rad/s)
            'joint_vel': {
                'min': np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0]),
                'max': np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
                'typical_std': 0.3
            },
            # End-effector position (meters)
            'ee_pos': {
                'min': np.array([0.05, -0.3, 0.05]),  # Min reach
                'max': np.array([0.4, 0.3, 0.5]),     # Max reach
                'typical_std': 0.05
            },
            # Target position (meters)
            'target_pos': {
                'min': np.array([0.06, -0.06, 0.08]),
                'max': np.array([0.25, 0.05, 0.25]),
                'typical_std': 0.0  # Constant during episode
            },
            # Distance
            'distance': {
                'min': 0.0,
                'max': 0.4,
                'typical_std': 0.05
            }
        }

        self.observations = []
        self.max_samples = 100

        self.obs_sub = self.create_subscription(
            Float32MultiArray,
            '/parol6/observations',
            self.observation_callback,
            10
        )

        self.get_logger().info('Collecting observations... (0/100)')

    def observation_callback(self, msg):
        """Store observations."""
        obs = np.array(msg.data, dtype=np.float32)
        if obs.shape[0] == 19:
            self.observations.append(obs)

            if len(self.observations) % 10 == 0:
                self.get_logger().info(f'Collecting observations... ({len(self.observations)}/100)')

            if len(self.observations) >= self.max_samples:
                self.analyze_and_exit()

    def analyze_and_exit(self):
        """Analyze observations and print report."""
        obs_array = np.array(self.observations)

        print("\n" + "="*80)
        print("OBSERVATION DISTRIBUTION ANALYSIS")
        print("="*80)

        # Joint positions [0:6]
        print("\nJOINT POSITIONS [0:6]:")
        joint_pos = obs_array[:, 0:6]
        for i in range(6):
            actual_min, actual_max = joint_pos[:, i].min(), joint_pos[:, i].max()
            actual_mean, actual_std = joint_pos[:, i].mean(), joint_pos[:, i].std()
            expected_min, expected_max = self.expected_ranges['joint_pos']['min'][i], self.expected_ranges['joint_pos']['max'][i]

            # Check if within expected range
            in_range = (actual_min >= expected_min - 0.1) and (actual_max <= expected_max + 0.1)
            status = '✓' if in_range else '✗ OUT OF RANGE'

            print(f"  J{i+1}: mean={actual_mean:7.3f} std={actual_std:.3f} "
                  f"range=[{actual_min:7.3f}, {actual_max:7.3f}] "
                  f"expected=[{expected_min:7.3f}, {expected_max:7.3f}] {status}")

        # Joint velocities [6:12]
        print("\nJOINT VELOCITIES [6:12] (rad/s):")
        joint_vel = obs_array[:, 6:12]
        for i in range(6):
            actual_min, actual_max = joint_vel[:, i].min(), joint_vel[:, i].max()
            actual_mean, actual_std = joint_vel[:, i].mean(), joint_vel[:, i].std()

            # Check for excessive noise
            noisy = actual_std > 1.0
            status = '⚠ NOISY' if noisy else '✓'

            print(f"  J{i+1}_vel: mean={actual_mean:7.3f} std={actual_std:.3f} "
                  f"range=[{actual_min:7.3f}, {actual_max:7.3f}] {status}")

        # End-effector position [12:15]
        print("\nEND-EFFECTOR POSITION [12:15] (meters):")
        ee_pos = obs_array[:, 12:15]
        labels = ['X', 'Y', 'Z']
        for i, label in enumerate(labels):
            actual_min, actual_max = ee_pos[:, i].min(), ee_pos[:, i].max()
            actual_mean, actual_std = ee_pos[:, i].mean(), ee_pos[:, i].std()
            expected_min, expected_max = self.expected_ranges['ee_pos']['min'][i], self.expected_ranges['ee_pos']['max'][i]

            in_range = (actual_min >= expected_min - 0.05) and (actual_max <= expected_max + 0.05)
            status = '✓' if in_range else '✗ OUT OF RANGE'

            print(f"  EE_{label}: mean={actual_mean:7.3f} std={actual_std:.3f} "
                  f"range=[{actual_min:7.3f}, {actual_max:7.3f}] "
                  f"expected=[{expected_min:7.3f}, {expected_max:7.3f}] {status}")

        # Target position [15:18]
        print("\nTARGET POSITION [15:18] (meters):")
        target_pos = obs_array[:, 15:18]
        for i, label in enumerate(labels):
            actual_min, actual_max = target_pos[:, i].min(), target_pos[:, i].max()
            actual_mean, actual_std = target_pos[:, i].mean(), target_pos[:, i].std()

            varying = actual_std > 0.01
            status = '(varying)' if varying else '(constant)'

            print(f"  Target_{label}: mean={actual_mean:7.3f} std={actual_std:.3f} "
                  f"range=[{actual_min:7.3f}, {actual_max:7.3f}] {status}")

        # Distance [18]
        print("\nDISTANCE [18] (meters):")
        distance = obs_array[:, 18]
        print(f"  Distance: mean={distance.mean():7.3f} std={distance.std():.3f} "
              f"range=[{distance.min():7.3f}, {distance.max():7.3f}]")

        # Overall statistics
        print("\n" + "="*80)
        print("NORMALIZATION IMPACT ANALYSIS")
        print("="*80)

        # Calculate what normalized observations would look like
        obs_mean = obs_array.mean(axis=0)
        obs_std = obs_array.std(axis=0)

        print("\nCurrent Gazebo statistics (for embedded normalization):")
        print(f"  Mean (first 6): {obs_mean[0:6]}")
        print(f"  Std (first 6):  {obs_std[0:6]}")

        print("\nExpected Isaac Lab statistics (approximate):")
        print(f"  Mean (first 6): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (home position)")
        print(f"  Std (first 6):  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] (typical variation)")

        # Check for distribution mismatch
        print("\n" + "="*80)
        print("POTENTIAL ISSUES")
        print("="*80)

        issues = []

        # Check if joint positions are far from zero mean
        if np.abs(obs_mean[0:6]).max() > 0.5:
            issues.append(f"⚠ Joint positions have non-zero mean: {obs_mean[0:6]}")
            issues.append("  → Policy expects joints centered around home position (zeros)")

        # Check if joint velocities are noisy
        if obs_std[6:12].max() > 1.0:
            issues.append(f"⚠ Joint velocities are noisy: std={obs_std[6:12]}")
            issues.append("  → High noise confuses the policy")

        # Check if EE position is out of range
        ee_mean = obs_mean[12:15]
        if ee_mean[0] < 0.05 or ee_mean[0] > 0.4:
            issues.append(f"⚠ EE X position out of range: {ee_mean[0]:.3f}m")

        # Check if distance is not decreasing
        if distance.std() < 0.01:
            issues.append("⚠ Distance not changing - robot not moving toward target!")

        if issues:
            for issue in issues:
                print(issue)
        else:
            print("✓ No obvious distribution mismatches detected")

        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)

        if np.abs(obs_mean[0:6]).max() > 0.5:
            print("The observations have very different statistics than Isaac Lab training data.")
            print("The embedded normalization in policy.pt may be incorrect for Gazebo.")
            print("\nSOLUTION: Export ONNX with separate normalization file, or retrain with Gazebo data.")
        else:
            print("Observation distributions look reasonable.")
            print("Issue may be elsewhere (controller gains, physics, etc.)")

        rclpy.shutdown()


def main():
    rclpy.init()
    node = ObservationComparison()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
