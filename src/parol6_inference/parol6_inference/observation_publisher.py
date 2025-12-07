#!/usr/bin/env python3
"""
Observation Publisher Node for PAROL6 Reaching Task

Publishes 25D observations matching Isaac Lab format:
- Joint positions (6D)
- Joint velocities (6D)
- Target position relative to base (3D)
- Target orientation as quaternion [w,x,y,z] (4D)
- Previous actions (6D, normalized)
"""

import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros import TransformException
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
import numpy as np


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion [w, x, y, z].

    Args:
        roll: Rotation around X-axis (radians)
        pitch: Rotation around Y-axis (radians)
        yaw: Rotation around Z-axis (radians)

    Returns:
        np.array([w, x, y, z]) normalized quaternion
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Normalize quaternion
    quat = np.array([w, x, y, z])
    return quat / np.linalg.norm(quat)


def compute_parol6_fk(joint_positions):
    """
    Compute forward kinematics for PAROL6 robot.

    Returns end-effector position in base frame.

    Args:
        joint_positions: [J1, J2, J3, J4, J5, J6] in radians

    Returns:
        np.array([x, y, z]) end-effector position in meters
    """
    # DH parameters for PAROL6 (from URDF)
    # These are approximate - adjust if needed based on actual URDF
    q1, q2, q3, q4, q5, q6 = joint_positions

    # Link lengths (meters) - from PAROL6 URDF
    d1 = 0.11399  # Base height
    a2 = 0.17628  # Link 2 length
    a3 = 0.17628  # Link 3 length
    d4 = 0.11399  # Link 4 offset
    d6 = 0.0655   # End-effector offset

    # Simplified FK using transformation matrices
    # T = Rz(q1) * Tz(d1) * Ry(q2) * Tx(a2) * Ry(q3) * Tx(a3) * ...

    # Joint 1: Rotation around Z
    c1, s1 = np.cos(q1), np.sin(q1)

    # Joint 2: Rotation around Y (at height d1)
    c2, s2 = np.cos(q2), np.sin(q2)

    # Joint 3: Rotation around Y (at distance a2 from J2)
    c3, s3 = np.cos(q3), np.sin(q3)
    c23 = np.cos(q2 + q3)
    s23 = np.sin(q2 + q3)

    # Simplified position calculation (3DOF for position)
    # Ignoring J4, J5, J6 for position (they only affect orientation)

    # Position in XZ plane (before J1 rotation)
    reach = a2 * c2 + a3 * c23 + d4 * c23 + d6
    height = d1 + a2 * s2 + a3 * s23 + d4 * s23

    # Rotate by J1 to get final position
    x = reach * c1
    y = reach * s1
    z = height

    return np.array([x, y, z])


class ObservationPublisher(Node):
    """Publishes observations for PAROL6 reaching policy."""

    def __init__(self):
        super().__init__('observation_publisher')

        # Declare parameters
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('ee_frame', 'L6')
        self.declare_parameter('publish_rate', 100.0)  # Hz
        self.declare_parameter('target_x', 0.20)
        self.declare_parameter('target_y', 0.03)
        self.declare_parameter('target_z', 0.24)
        self.declare_parameter('target_yaw', 0.0)  # Target yaw angle (radians)
        self.declare_parameter('use_fk', False)  # Use TF by default (coded FK has incorrect DH params)

        # Get parameters
        self.base_frame = self.get_parameter('base_frame').value
        self.ee_frame = self.get_parameter('ee_frame').value
        publish_rate = self.get_parameter('publish_rate').value
        self.use_fk = self.get_parameter('use_fk').value

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Joint state subscriber
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Target position subscriber
        self.target_sub = self.create_subscription(
            Point,
            '/target_position',
            self.target_callback,
            10
        )

        # Action feedback subscriber (for previous actions)
        self.action_feedback_sub = self.create_subscription(
            Float32MultiArray,
            '/parol6/last_action',
            self.action_feedback_callback,
            10
        )

        # Observation publisher (25D vector)
        self.obs_pub = self.create_publisher(
            Float32MultiArray,
            '/parol6/observations',
            10
        )

        # End-effector position publisher (for debugging)
        self.ee_pos_pub = self.create_publisher(
            Point,
            '/parol6/ee_position',
            10
        )

        # Storage for joint states
        self.joint_pos = np.zeros(6)
        self.joint_vel = np.zeros(6)
        self.joint_state_received = False

        # Target position (relative to base)
        target_x = self.get_parameter('target_x').value
        target_y = self.get_parameter('target_y').value
        target_z = self.get_parameter('target_z').value
        self.target_pos_rel = np.array([target_x, target_y, target_z])

        # Target orientation (fixed for reaching task)
        target_yaw = self.get_parameter('target_yaw').value
        roll = 0.0  # Fixed
        pitch = np.pi / 2  # Point down
        self.target_quat = euler_to_quaternion(roll, pitch, target_yaw)

        # Previous actions (normalized, initialized to zeros)
        self.prev_actions = np.zeros(6)

        # Joint name mapping (PAROL6 specific)
        self.joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']

        # Publisher timer
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_observations)

        self.get_logger().info('=== Observation Publisher Started (25D) ===')
        self.get_logger().info(f'Base frame: {self.base_frame}')
        self.get_logger().info(f'EE frame: {self.ee_frame}')
        self.get_logger().info(f'Publish rate: {publish_rate} Hz')
        self.get_logger().info(f'Use FK: {self.use_fk} (instead of TF lookup)')
        self.get_logger().info(f'Target position: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}]')
        self.get_logger().info(f'Target orientation (RPY): [0.0, {pitch:.3f}, {target_yaw:.3f}]')
        self.get_logger().info(f'Target quaternion: [{self.target_quat[0]:.3f}, {self.target_quat[1]:.3f}, {self.target_quat[2]:.3f}, {self.target_quat[3]:.3f}]')

    def joint_state_callback(self, msg):
        """Update joint positions and velocities."""
        # Create mapping from joint names to indices
        name_to_idx = {name: idx for idx, name in enumerate(msg.name)}

        # Extract joint positions and velocities in correct order
        try:
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in name_to_idx:
                    idx = name_to_idx[joint_name]
                    self.joint_pos[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        self.joint_vel[i] = msg.velocity[idx]

            self.joint_state_received = True

        except (IndexError, KeyError) as e:
            self.get_logger().error(f'Error extracting joint states: {e}')

    def target_callback(self, msg):
        """Update target position when new goal is received."""
        self.target_pos_rel = np.array([msg.x, msg.y, msg.z])
        self.get_logger().info(f'Target updated: [{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}]')

    def action_feedback_callback(self, msg):
        """Update previous actions from policy inference feedback."""
        if len(msg.data) == 6:
            self.prev_actions = np.array(msg.data, dtype=np.float32)
        else:
            self.get_logger().warning(f'Invalid action feedback size: {len(msg.data)} (expected 6)')

    def get_ee_position_relative_to_base(self):
        """
        Get end-effector position in base frame using TF2.

        Returns:
            np.ndarray: [x, y, z] position in meters, or None if TF lookup fails
        """
        try:
            # Lookup transform from base to end-effector
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,  # Target frame (robot base)
                self.ee_frame,    # Source frame (end-effector)
                rclpy.time.Time(),  # Latest available
                timeout=rclpy.duration.Duration(seconds=0.05)
            )

            # Extract position (already in base frame)
            ee_pos_rel = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])

            return ee_pos_rel

        except TransformException as ex:
            # Only log warning occasionally to avoid spam
            if not hasattr(self, '_tf_warning_count'):
                self._tf_warning_count = 0

            self._tf_warning_count += 1
            if self._tf_warning_count % 100 == 0:
                self.get_logger().warning(
                    f'TF lookup failed ({self._tf_warning_count} times): {ex}'
                )

            return None

    def publish_observations(self):
        """
        Publish complete 25D observation vector.

        Observation structure:
        - [0:5]   Joint positions (6D, radians)
        - [6:11]  Joint velocities (6D, rad/s)
        - [12:14] Target position relative to base (3D, meters)
        - [15:18] Target orientation quaternion [w,x,y,z] (4D)
        - [19:24] Previous actions (6D, normalized)
        """
        # Wait for first joint state message
        if not self.joint_state_received:
            return

        # Get end-effector position (for debugging only, not in observation)
        if self.use_fk:
            # Compute using forward kinematics (faster and more reliable)
            ee_pos_rel = compute_parol6_fk(self.joint_pos)
        else:
            # Get via TF2 (may be slower or stale)
            ee_pos_rel = self.get_ee_position_relative_to_base()
            if ee_pos_rel is None:
                ee_pos_rel = np.zeros(3)  # Use zeros if TF not available

        # Construct 25D observation vector
        observation = np.concatenate([
            self.joint_pos,        # 0-5:   6D joint positions
            self.joint_vel,        # 6-11:  6D joint velocities
            self.target_pos_rel,   # 12-14: 3D target position
            self.target_quat,      # 15-18: 4D target quaternion [w,x,y,z]
            self.prev_actions      # 19-24: 6D previous actions (normalized)
        ])

        # Publish observation
        msg = Float32MultiArray()
        msg.data = observation.astype(np.float32).tolist()
        self.obs_pub.publish(msg)

        # Publish EE position for debugging
        ee_msg = Point()
        ee_msg.x = float(ee_pos_rel[0])
        ee_msg.y = float(ee_pos_rel[1])
        ee_msg.z = float(ee_pos_rel[2])
        self.ee_pos_pub.publish(ee_msg)

        # Log status occasionally
        if not hasattr(self, '_pub_count'):
            self._pub_count = 0

        self._pub_count += 1
        if self._pub_count % 1000 == 0:  # Every 10 seconds at 100Hz
            distance = np.linalg.norm(ee_pos_rel - self.target_pos_rel)
            self.get_logger().info(
                f'Published {self._pub_count} observations (25D) | '
                f'EE: [{ee_pos_rel[0]:.3f}, {ee_pos_rel[1]:.3f}, {ee_pos_rel[2]:.3f}] | '
                f'Distance: {distance:.4f}m'
            )


def main(args=None):
    rclpy.init(args=args)

    node = ObservationPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down observation publisher')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
