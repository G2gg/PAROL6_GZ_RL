#!/usr/bin/env python3
"""
Policy Inference Node for PAROL6 Reaching Task

Subscribes to observations and publishes actions from trained policy.
Supports multiple policy formats: ONNX, PyTorch, TorchScript
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import os
from pathlib import Path


class PolicyInference(Node):
    """Runs trained Isaac Lab policy for PAROL6 reaching."""

    def __init__(self):
        super().__init__('policy_inference')

        # Declare parameters
        self.declare_parameter('policy_path', '')
        self.declare_parameter('policy_type', 'onnx')  # 'onnx', 'pytorch', 'torchscript'
        self.declare_parameter('control_rate', 100.0)  # Hz
        self.declare_parameter('use_ema_smoothing', True)
        self.declare_parameter('ema_alpha', 0.3)
        self.declare_parameter('trajectory_time', 0.1)  # Time to reach target position (seconds)

        # Get parameters
        policy_path = self.get_parameter('policy_path').value
        self.policy_type = self.get_parameter('policy_type').value
        control_rate = self.get_parameter('control_rate').value
        self.use_ema_smoothing = self.get_parameter('use_ema_smoothing').value
        self.ema_alpha = self.get_parameter('ema_alpha').value
        self.trajectory_time = self.get_parameter('trajectory_time').value

        # Joint configuration
        self.joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        self.num_joints = 6

        # Joint limits (from Isaac Lab config)
        self.joint_limits_lower = np.array([-1.083, -1.221, -1.8825, -1.841, -1.571, -3.142])
        self.joint_limits_upper = np.array([2.148, 0.907, 1.2566, 1.841, 1.571, 3.142])

        # Observation subscriber
        self.obs_sub = self.create_subscription(
            Float32MultiArray,
            '/parol6/observations',
            self.observation_callback,
            10
        )

        # Action publisher (JointTrajectory for position control)
        self.action_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        # Action smoothing state
        self.prev_action = None

        # Observation normalization (if available)
        self.obs_mean = None
        self.obs_std = None

        # Load policy
        self.policy = self.load_policy(policy_path)

        # Try to load observation normalization after policy is loaded
        self.load_obs_normalization(policy_path)

        if self.policy is None:
            self.get_logger().error('Failed to load policy. Exiting.')
            raise RuntimeError('Policy loading failed')

        # Statistics
        self.inference_count = 0
        self.last_observation = None

        self.get_logger().info('=== Policy Inference Node Started ===')
        self.get_logger().info(f'Policy path: {policy_path}')
        self.get_logger().info(f'Policy type: {self.policy_type}')
        self.get_logger().info(f'Control rate: {control_rate} Hz')
        self.get_logger().info(f'EMA smoothing: {self.use_ema_smoothing} (alpha={self.ema_alpha})')
        self.get_logger().info(f'Trajectory time: {self.trajectory_time}s ({self.trajectory_time*1000:.0f}ms)')

    def load_policy(self, policy_path):
        """
        Load trained policy from file.

        Supports:
        - ONNX (.onnx)
        - PyTorch (.pth for state dicts)
        - TorchScript (.pt, .ts for JIT-compiled models with embedded normalization)

        Note: .pt files from Isaac Lab's export_policy_as_jit() are TorchScript models
        and should use policy_type='torchscript'.

        Args:
            policy_path: Path to policy file

        Returns:
            Loaded policy object or None if failed
        """
        if not policy_path or not os.path.exists(policy_path):
            self.get_logger().error(f'Policy file not found: {policy_path}')
            self.get_logger().info('Please provide a valid policy_path parameter')
            return None

        policy_path = Path(policy_path)
        extension = policy_path.suffix.lower()

        try:
            if self.policy_type == 'onnx' or extension == '.onnx':
                return self._load_onnx_policy(policy_path)

            elif self.policy_type == 'pytorch' and extension == '.pth':
                return self._load_pytorch_policy(policy_path)

            elif self.policy_type == 'torchscript' or extension in ['.pt', '.ts']:
                # .pt files from export_policy_as_jit are TorchScript models
                return self._load_torchscript_policy(policy_path)

            else:
                self.get_logger().error(f'Unsupported policy type: {self.policy_type}')
                self.get_logger().info('For .pt files from Isaac Lab, use policy_type="torchscript"')
                return None

        except Exception as e:
            self.get_logger().error(f'Error loading policy: {e}')
            return None

    def _load_onnx_policy(self, policy_path):
        """Load ONNX model."""
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(policy_path),
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

            self.get_logger().info(f'✓ ONNX policy loaded: {policy_path}')
            self.get_logger().info(f'Input name: {session.get_inputs()[0].name}')
            self.get_logger().info(f'Input shape: {session.get_inputs()[0].shape}')

            return session

        except ImportError:
            self.get_logger().error('onnxruntime not installed. Run: pip install onnxruntime-gpu')
            return None

    def _load_pytorch_policy(self, policy_path):
        """Load PyTorch model."""
        try:
            import torch

            model = torch.load(str(policy_path), map_location='cpu')
            model.eval()

            self.get_logger().info(f'✓ PyTorch policy loaded: {policy_path}')
            return model

        except ImportError:
            self.get_logger().error('torch not installed. Run: pip install torch')
            return None

    def _load_torchscript_policy(self, policy_path):
        """
        Load TorchScript model.

        TorchScript models from Isaac Lab's export_policy_as_jit() include
        embedded observation normalization, so no separate _norm.npz file is needed.
        """
        try:
            import torch

            model = torch.jit.load(str(policy_path))
            model.eval()

            self.get_logger().info(f'✓ TorchScript policy loaded: {policy_path}')
            self.get_logger().info('  Note: Normalization is embedded in TorchScript model')
            return model

        except ImportError:
            self.get_logger().error('torch not installed. Run: pip install torch')
            return None

    def load_obs_normalization(self, policy_path):
        """
        Load observation normalization statistics if available.

        Note: TorchScript models from export_policy_as_jit() have normalization
        embedded and don't need external _norm.npz files.
        """
        # TorchScript models have embedded normalization
        if self.policy_type == 'torchscript':
            self.get_logger().info('Using TorchScript model with embedded normalization')
            return

        # Try to find normalization file for ONNX and PyTorch models
        norm_path = str(policy_path).replace('.onnx', '_norm.npz').replace('.pt', '_norm.npz').replace('.pth', '_norm.npz').replace('.ts', '_norm.npz')

        if os.path.exists(norm_path):
            try:
                norm_data = np.load(norm_path)
                self.obs_mean = norm_data['mean']
                self.obs_std = norm_data['std']
                self.get_logger().info(f'✓ Loaded observation normalization from: {norm_path}')
                self.get_logger().info(f'  Mean (first 3): [{self.obs_mean[0]:.3f}, {self.obs_mean[1]:.3f}, {self.obs_mean[2]:.3f}]')
                self.get_logger().info(f'  Std (first 3): [{self.obs_std[0]:.3f}, {self.obs_std[1]:.3f}, {self.obs_std[2]:.3f}]')
            except Exception as e:
                self.get_logger().warn(f'Failed to load normalization: {e}')
        else:
            self.get_logger().warn(f'No observation normalization found at: {norm_path}')
            self.get_logger().warn('  Policy may not work correctly if trained with normalization!')
            self.get_logger().warn('  Create default normalization with: cd policy && python3 create_default_norm.py')

    def normalize_observation(self, observation):
        """
        Normalize observation using mean/std if available.

        For TorchScript models, normalization is embedded in the model,
        so we return the raw observation.

        Args:
            observation: Raw observation numpy array

        Returns:
            normalized_observation: Normalized observation (or raw if TorchScript)
        """
        # TorchScript models have embedded normalization
        if self.policy_type == 'torchscript':
            return observation

        # Apply external normalization for ONNX and PyTorch models
        if self.obs_mean is not None and self.obs_std is not None:
            # Normalize: (obs - mean) / std
            return (observation - self.obs_mean) / (self.obs_std + 1e-8)
        else:
            # No normalization available
            return observation

    def run_inference(self, observation):
        """
        Run policy inference.

        For TorchScript models: observation normalization is handled inside the model.
        For ONNX/PyTorch models: observation normalization is applied here.

        Args:
            observation: 19D numpy array (raw)

        Returns:
            action: 6D numpy array (normalized [-1, 1])
        """
        # Prepare observation (normalization handled differently based on model type)
        obs_normalized = self.normalize_observation(observation)
        obs_input = obs_normalized.reshape(1, -1).astype(np.float32)

        if self.policy_type == 'onnx':
            # ONNX inference
            input_name = self.policy.get_inputs()[0].name
            action = self.policy.run(None, {input_name: obs_input})[0]
            return action[0]  # Extract from batch dimension

        elif self.policy_type in ['pytorch', 'torchscript']:
            # PyTorch/TorchScript inference
            import torch

            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs_input)
                action_tensor = self.policy(obs_tensor)

                # IMPORTANT: Apply tanh activation if not already in model
                # TorchScript exports from Isaac Lab may not include final tanh layer
                # This ensures actions are bounded to [-1, 1] range
                if self.policy_type == 'torchscript':
                    action_tensor = torch.tanh(action_tensor)

                action = action_tensor.cpu().numpy()[0]

            return action

        else:
            self.get_logger().error(f'Unknown policy type: {self.policy_type}')
            return np.zeros(self.num_joints)

    def denormalize_action(self, action):
        """
        Convert normalized action [-1, 1] to joint positions [lower, upper].

        Args:
            action: 6D array in range [-1, 1]

        Returns:
            joint_positions: 6D array in joint limits
        """
        # Map from [-1, 1] to [lower, upper]
        joint_pos = self.joint_limits_lower + (action + 1.0) / 2.0 * (
            self.joint_limits_upper - self.joint_limits_lower
        )

        # Clamp to limits for safety
        joint_pos = np.clip(joint_pos, self.joint_limits_lower, self.joint_limits_upper)

        return joint_pos

    def apply_ema_smoothing(self, action):
        """
        Apply Exponential Moving Average smoothing to actions.

        Args:
            action: Current action (6D)

        Returns:
            smoothed_action: Smoothed action (6D)
        """
        if self.prev_action is None:
            self.prev_action = action.copy()
            return action

        # Smooth: new = alpha * current + (1 - alpha) * previous
        smoothed = self.ema_alpha * action + (1.0 - self.ema_alpha) * self.prev_action
        self.prev_action = smoothed.copy()

        return smoothed

    def observation_callback(self, msg):
        """
        Process observation and publish action.

        Args:
            msg: Float32MultiArray containing 19D observation
        """
        # Convert to numpy array
        observation = np.array(msg.data, dtype=np.float32)

        if observation.shape[0] != 19:
            self.get_logger().error(f'Invalid observation size: {observation.shape[0]} (expected 19)')
            return

        self.last_observation = observation

        # Run policy inference
        action_normalized = self.run_inference(observation)  # Range: [-1, 1]

        # Denormalize to joint positions
        joint_positions = self.denormalize_action(action_normalized)

        # Apply EMA smoothing if enabled
        if self.use_ema_smoothing:
            joint_positions = self.apply_ema_smoothing(joint_positions)

        # Publish action as JointTrajectory
        self.publish_action(joint_positions)

        # Update statistics
        self.inference_count += 1

        # Log occasionally with detailed debugging
        if self.inference_count % 100 == 0:
            distance = observation[18]
            ee_pos = observation[12:15]
            target_pos = observation[15:18]
            joint_pos = observation[0:6]
            joint_vel = observation[6:12]

            self.get_logger().info(
                f'Inference #{self.inference_count} | '
                f'Distance: {distance:.4f}m | '
                f'EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] | '
                f'Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]'
            )
            self.get_logger().info(
                f'  Current joints: [{joint_pos[0]:.3f}, {joint_pos[1]:.3f}, {joint_pos[2]:.3f}, '
                f'{joint_pos[3]:.3f}, {joint_pos[4]:.3f}, {joint_pos[5]:.3f}]'
            )
            self.get_logger().info(
                f'  Joint velocities: [{joint_vel[0]:.3f}, {joint_vel[1]:.3f}, {joint_vel[2]:.3f}, '
                f'{joint_vel[3]:.3f}, {joint_vel[4]:.3f}, {joint_vel[5]:.3f}]'
            )
            self.get_logger().info(
                f'  Policy output (normalized): [{action_normalized[0]:.3f}, {action_normalized[1]:.3f}, '
                f'{action_normalized[2]:.3f}, {action_normalized[3]:.3f}, {action_normalized[4]:.3f}, {action_normalized[5]:.3f}]'
            )
            self.get_logger().info(
                f'  Commanded joints: [{joint_positions[0]:.3f}, {joint_positions[1]:.3f}, {joint_positions[2]:.3f}, '
                f'{joint_positions[3]:.3f}, {joint_positions[4]:.3f}, {joint_positions[5]:.3f}]'
            )

            # Calculate error metrics
            pos_error = np.linalg.norm(ee_pos - target_pos)
            joint_change = np.linalg.norm(joint_positions - joint_pos)
            self.get_logger().info(
                f'  Position error: {pos_error:.4f}m | Joint change: {joint_change:.4f}rad'
            )

    def publish_action(self, joint_positions):
        """
        Publish joint positions as JointTrajectory.

        Args:
            joint_positions: 6D array of target joint positions (radians)
        """
        msg = JointTrajectory()

        # Leave header stamp empty - controller will use current time
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()
        point.velocities = [0.0] * self.num_joints  # Add zero velocities for smooth motion

        # Convert trajectory time to sec + nanosec
        traj_time_sec = int(self.trajectory_time)
        traj_time_nsec = int((self.trajectory_time - traj_time_sec) * 1e9)
        point.time_from_start.sec = traj_time_sec
        point.time_from_start.nanosec = traj_time_nsec

        msg.points = [point]

        self.action_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = PolicyInference()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
