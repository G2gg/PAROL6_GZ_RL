#!/usr/bin/env python3
"""
Policy Inference Node for PAROL6 Reaching Task

Subscribes to 25D observations and publishes actions from trained policy.
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
        self.declare_parameter('trajectory_time', 1.0)  # Time to reach target position (seconds)

       

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

        # Joint limits (from As per the HARDWARE config)
        self.joint_limits_lower = np.array([-1.083, -1.221, -1.8825, -1.841, -1.571, -3.142])
        self.joint_limits_upper = np.array([2.148, 0.907, 1.2566, 1.841, 1.571, 3.142])

        # Action scale (from Isaac Lab training config)
        self.action_scale = 0.25  # From parol6_env_cfg.py: ActionsCfg.arm_action.scale

        # Current joint positions (updated from observations)
        self.current_joint_pos = np.zeros(6)

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

        # Action feedback publisher (for observation loop)
        self.action_feedback_pub = self.create_publisher(
            Float32MultiArray,
            '/parol6/last_action',
            10
        )

        # Action smoothing state
        self.prev_action = None

        # Observation normalization (if available)
        self.obs_mean = None
        self.obs_std = None

        # Load policy
        self.policy = self.load_policy(policy_path)
        self.policy_counter = 0
        self.last_action = None  # Store last computed action

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
            observation: 25D numpy array (raw)

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

                # IMPORTANT: Apply tanh activation to bound actions to [-1, 1]
                # The exported policy may not include the final tanh layer
                action_tensor = torch.tanh(action_tensor)

                action = action_tensor.cpu().numpy()[0]

            return action

        else:
            self.get_logger().error(f'Unknown policy type: {self.policy_type}')
            return np.zeros(self.num_joints)

    # def apply_action_delta(self, action_normalized):
    #     """
    #     Convert normalized action [-1, 1] to joint position command.

    #     Matches Isaac Lab training: action = current + (normalized * scale)

    #     This is the CORRECT approach matching the training configuration:
    #     - Policy outputs normalized actions in [-1, 1]
    #     - Actions are scaled by 0.25 (from parol6_env_cfg.py)
    #     - Actions are DELTAS added to current position, not absolute positions

    #     Args:
    #         action_normalized: 6D array in range [-1, 1]

    #     Returns:
    #         joint_positions: 6D array of commanded joint positions
    #     """
    #     # Compute delta from normalized action (matching Isaac Lab's JointPositionActionCfg)
    #     delta = action_normalized * self.action_scale

    #     # Add delta to current position
    #     joint_pos = self.current_joint_pos + delta

    #     # Clamp to limits for safety
    #     joint_pos = np.clip(joint_pos, self.joint_limits_lower, self.joint_limits_upper)

    #     return joint_pos

    def apply_action_delta(self, action_normalized):
      # Use DEFAULT positions (all zeros for PAROL6)
      default_joint_pos = np.zeros(6)

      delta = action_normalized * self.action_scale
      joint_pos = default_joint_pos + delta  # Use default, not current!

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
            msg: Float32MultiArray containing 25D observation
        """
        # Convert to numpy array
        observation = np.array(msg.data, dtype=np.float32)

        if observation.shape[0] != 25:
            self.get_logger().error(f'Invalid observation size: {observation.shape[0]} (expected 25)')
            return

        self.last_observation = observation

        # Extract current joint positions (needed for delta actions)
        self.current_joint_pos = observation[0:6]

        # Run policy inference at full rate (no decimation)
        action_normalized = self.run_inference(observation)  # Range: [-1, 1]

        # Publish action feedback (normalized actions for observation loop)
        feedback_msg = Float32MultiArray()
        feedback_msg.data = action_normalized.astype(np.float32).tolist()
        self.action_feedback_pub.publish(feedback_msg)

        # Apply action delta to current position (matching Isaac Lab training)
        joint_positions = self.apply_action_delta(action_normalized)

        # Apply EMA smoothing if enabled
        if self.use_ema_smoothing:
            joint_positions = self.apply_ema_smoothing(joint_positions)

        # Store the action
        self.last_action = joint_positions

        # Publish action at full rate
        self.publish_action(joint_positions)

        # Update statistics
        self.inference_count += 1

        # Log occasionally with detailed debugging
        if self.inference_count % 100 == 0:
            target_pos = observation[12:15]
            target_quat = observation[15:19]
            prev_actions = observation[19:25]
            joint_pos = observation[0:6]
            joint_vel = observation[6:12]

            self.get_logger().info(
                f'Inference #{self.inference_count} (25D) | '
                f'Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] | '
                f'Quat: [{target_quat[0]:.2f}, {target_quat[1]:.2f}, {target_quat[2]:.2f}, {target_quat[3]:.2f}]'
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
                f'  Previous actions: [{prev_actions[0]:.3f}, {prev_actions[1]:.3f}, {prev_actions[2]:.3f}, '
                f'{prev_actions[3]:.3f}, {prev_actions[4]:.3f}, {prev_actions[5]:.3f}]'
            )

            # Only log action if we have one
            if self.last_action is not None:
                # Compute action delta for logging
                action_delta = self.last_action - self.current_joint_pos

                self.get_logger().info(
                    f'  Policy output (normalized): [{prev_actions[0]:.3f}, {prev_actions[1]:.3f}, '
                    f'{prev_actions[2]:.3f}, {prev_actions[3]:.3f}, {prev_actions[4]:.3f}, {prev_actions[5]:.3f}]'
                )
                self.get_logger().info(
                    f'  Action delta (scaled): [{action_delta[0]:.3f}, {action_delta[1]:.3f}, {action_delta[2]:.3f}, '
                    f'{action_delta[3]:.3f}, {action_delta[4]:.3f}, {action_delta[5]:.3f}]'
                )
                self.get_logger().info(
                    f'  Commanded joints: [{self.last_action[0]:.3f}, {self.last_action[1]:.3f}, {self.last_action[2]:.3f}, '
                    f'{self.last_action[3]:.3f}, {self.last_action[4]:.3f}, {self.last_action[5]:.3f}]'
                )

    def publish_action(self, joint_positions):
        """
        Publish joint positions as JointTrajectory.

        Creates a single-point trajectory (matching robotis_lab approach):
        - Point 0: Target position at t=trajectory_time

        Args:
            joint_positions: 6D array of target joint positions (radians)
        """
        msg = JointTrajectory()

        # Set header timestamp to current time
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.joint_names

        # Single point: Move to target position
        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()
        point.velocities = []  # Empty velocities
        point.accelerations = []  # Empty accelerations
        point.effort = []  # Empty effort

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
