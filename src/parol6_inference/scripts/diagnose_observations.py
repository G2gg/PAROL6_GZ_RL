#!/usr/bin/env python3
"""
Diagnostic tool to analyze observations and compare with Isaac Lab expectations.

This script subscribes to observations and analyzes their statistics to help
debug sim-to-sim transfer issues.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from collections import deque


class ObservationDiagnostic(Node):
    """Analyzes observation statistics for debugging."""

    def __init__(self):
        super().__init__('observation_diagnostic')

        # Storage for observation history
        self.obs_history = deque(maxlen=1000)

        # Subscribe to observations
        self.obs_sub = self.create_subscription(
            Float32MultiArray,
            '/parol6/observations',
            self.observation_callback,
            10
        )

        # Timer for periodic analysis
        self.timer = self.create_timer(5.0, self.analyze_observations)

        self.get_logger().info('=== Observation Diagnostic Started ===')
        self.get_logger().info('Collecting observations for analysis...')

    def observation_callback(self, msg):
        """Store observation for analysis."""
        obs = np.array(msg.data, dtype=np.float32)
        if obs.shape[0] == 19:
            self.obs_history.append(obs)

    def analyze_observations(self):
        """Analyze observation statistics."""
        if len(self.obs_history) < 10:
            self.get_logger().info(f'Collecting data... ({len(self.obs_history)}/1000)')
            return

        # Convert to numpy array
        obs_array = np.array(self.obs_history)  # Shape: (N, 19)

        # Calculate statistics
        obs_mean = np.mean(obs_array, axis=0)
        obs_std = np.std(obs_array, axis=0)
        obs_min = np.min(obs_array, axis=0)
        obs_max = np.max(obs_array, axis=0)

        # Expected ranges from Isaac Lab
        expected_joint_limits_lower = np.array([-1.083, -1.221, -1.8825, -1.841, -1.571, -3.142])
        expected_joint_limits_upper = np.array([2.148, 0.907, 1.2566, 1.841, 1.571, 3.142])

        self.get_logger().info('=' * 80)
        self.get_logger().info(f'OBSERVATION ANALYSIS ({len(self.obs_history)} samples)')
        self.get_logger().info('=' * 80)

        # Joint positions [0:6]
        self.get_logger().info('\nJOINT POSITIONS [0:6]:')
        for i in range(6):
            in_range = expected_joint_limits_lower[i] <= obs_mean[i] <= expected_joint_limits_upper[i]
            status = '✓' if in_range else '✗ OUT OF RANGE'
            self.get_logger().info(
                f'  J{i+1}: mean={obs_mean[i]:7.3f} std={obs_std[i]:.3f} '
                f'range=[{obs_min[i]:7.3f}, {obs_max[i]:7.3f}] '
                f'expected=[{expected_joint_limits_lower[i]:7.3f}, {expected_joint_limits_upper[i]:7.3f}] {status}'
            )

        # Joint velocities [6:12]
        self.get_logger().info('\nJOINT VELOCITIES [6:12] (rad/s):')
        for i in range(6):
            idx = 6 + i
            velocity_issue = obs_std[idx] > 5.0  # High velocity noise
            status = '⚠ HIGH NOISE' if velocity_issue else '✓'
            self.get_logger().info(
                f'  J{i+1}_vel: mean={obs_mean[idx]:7.3f} std={obs_std[idx]:.3f} '
                f'range=[{obs_min[idx]:7.3f}, {obs_max[idx]:7.3f}] {status}'
            )

        # End-effector position [12:15]
        self.get_logger().info('\nEND-EFFECTOR POSITION [12:15] (meters, relative to base):')
        ee_labels = ['X', 'Y', 'Z']
        for i, label in enumerate(ee_labels):
            idx = 12 + i
            self.get_logger().info(
                f'  EE_{label}: mean={obs_mean[idx]:7.3f} std={obs_std[idx]:.3f} '
                f'range=[{obs_min[idx]:7.3f}, {obs_max[idx]:7.3f}]'
            )

        # Target position [15:18]
        self.get_logger().info('\nTARGET POSITION [15:18] (meters, relative to base):')
        for i, label in enumerate(ee_labels):
            idx = 15 + i
            is_constant = obs_std[idx] < 0.001
            status = '(constant)' if is_constant else '(varying)'
            self.get_logger().info(
                f'  Target_{label}: mean={obs_mean[idx]:7.3f} std={obs_std[idx]:.3f} '
                f'range=[{obs_min[idx]:7.3f}, {obs_max[idx]:7.3f}] {status}'
            )

        # Distance [18]
        self.get_logger().info('\nDISTANCE TO TARGET [18] (meters):')
        self.get_logger().info(
            f'  Distance: mean={obs_mean[18]:7.3f} std={obs_std[18]:.3f} '
            f'range=[{obs_min[18]:7.3f}, {obs_max[18]:7.3f}]'
        )

        # Check for potential issues
        self.get_logger().info('\n' + '=' * 80)
        self.get_logger().info('POTENTIAL ISSUES:')
        self.get_logger().info('=' * 80)

        issues_found = False

        # Check joint limits
        for i in range(6):
            if obs_min[i] < expected_joint_limits_lower[i] - 0.1 or obs_max[i] > expected_joint_limits_upper[i] + 0.1:
                self.get_logger().warn(
                    f'⚠ J{i+1} exceeds expected limits! '
                    f'Observed: [{obs_min[i]:.3f}, {obs_max[i]:.3f}], '
                    f'Expected: [{expected_joint_limits_lower[i]:.3f}, {expected_joint_limits_upper[i]:.3f}]'
                )
                issues_found = True

        # Check velocity noise
        for i in range(6):
            idx = 6 + i
            if obs_std[idx] > 5.0:
                self.get_logger().warn(
                    f'⚠ J{i+1} velocity has high noise (std={obs_std[idx]:.3f} rad/s). '
                    f'This may confuse the policy!'
                )
                issues_found = True

        # Check if target is constant
        if obs_std[15] > 0.01 or obs_std[16] > 0.01 or obs_std[17] > 0.01:
            self.get_logger().warn(
                '⚠ Target position is varying! Expected constant target for position task.'
            )
            issues_found = True

        # Check distance convergence
        if obs_mean[18] > 0.05:  # More than 5cm average distance
            self.get_logger().warn(
                f'⚠ Robot not converging to target! Average distance: {obs_mean[18]:.3f}m'
            )
            issues_found = True

        # Check if EE position is reasonable
        ee_distance_from_base = np.linalg.norm(obs_mean[12:15])
        if ee_distance_from_base < 0.05 or ee_distance_from_base > 0.6:
            self.get_logger().warn(
                f'⚠ End-effector position seems unrealistic! '
                f'Distance from base: {ee_distance_from_base:.3f}m '
                f'(expected 0.05-0.6m for PAROL6)'
            )
            issues_found = True

        if not issues_found:
            self.get_logger().info('✓ No obvious issues detected in observations')

        self.get_logger().info('=' * 80)


def main(args=None):
    rclpy.init(args=args)
    node = ObservationDiagnostic()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
