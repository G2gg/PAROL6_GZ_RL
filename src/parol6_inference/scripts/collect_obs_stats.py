#!/usr/bin/env python3
"""
Collect observation statistics from ROS2 topic for normalization.

Run this while the robot is moving to collect observation samples,
then compute mean/std for policy normalization.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import argparse
from pathlib import Path


class ObservationCollector(Node):
    def __init__(self, num_samples=1000):
        super().__init__('observation_collector')

        self.num_samples = num_samples
        self.observations = []

        self.sub = self.create_subscription(
            Float32MultiArray,
            '/parol6/observations',
            self.obs_callback,
            10
        )

        self.get_logger().info(f'Collecting {num_samples} observation samples...')
        self.get_logger().info('Move the robot around to get diverse samples!')

    def obs_callback(self, msg):
        obs = np.array(msg.data, dtype=np.float32)
        self.observations.append(obs)

        if len(self.observations) % 100 == 0:
            self.get_logger().info(f'Collected {len(self.observations)}/{self.num_samples} samples')

        if len(self.observations) >= self.num_samples:
            self.compute_and_save_stats()
            rclpy.shutdown()

    def compute_and_save_stats(self):
        obs_array = np.array(self.observations)

        mean = np.mean(obs_array, axis=0)
        std = np.std(obs_array, axis=0)

        # Ensure std is not too small (prevent division by zero)
        std = np.maximum(std, 0.01)

        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('Computed Observation Statistics:')
        self.get_logger().info('='*60)
        self.get_logger().info(f'Mean (first 5): {mean[:5]}')
        self.get_logger().info(f'Std (first 5): {std[:5]}')
        self.get_logger().info(f'Mean shape: {mean.shape}')
        self.get_logger().info(f'Std shape: {std.shape}')

        # Save to file
        output_path = '/home/gunesh_pop_nvidia/parol6_gz_ws/src/parol6_inference/policy/policy_norm.npz'
        np.savez(output_path, mean=mean, std=std)

        self.get_logger().info(f'\n✓ Saved normalization stats to: {output_path}')
        self.get_logger().info('\nYou can now run inference with observation normalization!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of observation samples to collect')
    args = parser.parse_args()

    rclpy.init()
    node = ObservationCollector(num_samples=args.samples)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
