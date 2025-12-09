#!/usr/bin/env python3
"""
Launch file for PAROL6 Isaac Lab policy inference system.

Launches:
1. Observation publisher (TF2-based end-effector tracking)
2. Policy inference node (ONNX/PyTorch model)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition
import os


def generate_launch_description():
    """Generate launch description for PAROL6 inference system."""

    # Declare launch arguments
    policy_path_arg = DeclareLaunchArgument(
        'policy_path',
        default_value='/home/gunesh_pop_nvidia/parol6_gz_ws/src/parol6_inference/policy/policy.pt',
        description='Path to trained policy file (.pt for TorchScript, .onnx for ONNX, .pth for PyTorch)'
    )

    policy_type_arg = DeclareLaunchArgument(
        'policy_type',
        default_value='torchscript',
        description='Policy type: torchscript (for .pt from Isaac Lab), onnx, or pytorch'
    )

    target_x_arg = DeclareLaunchArgument(
        'target_x',
        default_value='0.20',
        description='Target X position relative to robot base (meters)'
    )

    target_y_arg = DeclareLaunchArgument(
        'target_y',
        default_value='0.03',
        description='Target Y position relative to robot base (meters)'
    )

    target_z_arg = DeclareLaunchArgument(
        'target_z',
        default_value='0.24',
        description='Target Z position relative to robot base (meters)'
    )

    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='base_link',
        description='Robot base frame name'
    )

    ee_frame_arg = DeclareLaunchArgument(
        'ee_frame',
        default_value='L6',
        description='End-effector frame name'
    )

    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='120.0',
        description='Observation publishing rate (Hz)'
    )

    use_fk_arg = DeclareLaunchArgument(
        'use_fk',
        default_value='false',
        description='Use coded FK for EE position (false = use TF, which is more accurate)'
    )

    control_rate_arg = DeclareLaunchArgument(
        'control_rate',
        default_value='120.0',
        description='Policy inference rate (Hz)'
    )

    use_ema_smoothing_arg = DeclareLaunchArgument(
        'use_ema_smoothing',
        default_value='false',
        description='Enable exponential moving average action smoothing'
    )

    ema_alpha_arg = DeclareLaunchArgument(
        'ema_alpha',
        default_value='0.3',
        description='EMA smoothing factor (0.0-1.0, lower = more smoothing)'
    )

    trajectory_time_arg = DeclareLaunchArgument(
        'trajectory_time',
        default_value='0.5',  # 200ms trajectory execution time
        description='Trajectory execution time in seconds'
    )

    run_inference_arg = DeclareLaunchArgument(
        'run_inference',
        default_value='true',
        description='Launch policy inference node (requires policy_path)'
    )

    use_interactive_marker_arg = DeclareLaunchArgument(
        'use_interactive_marker',
        default_value='true',
        description='Launch interactive RViz marker for visual goal setting'
    )

    marker_scale_arg = DeclareLaunchArgument(
        'marker_scale',
        default_value='0.05',
        description='Scale of the interactive marker sphere (meters)'
    )

    # Observation publisher node
    observation_publisher_node = Node(
        package='parol6_inference',
        executable='observation_publisher.py',
        name='observation_publisher',
        output='screen',
        parameters=[{
            'base_frame': LaunchConfiguration('base_frame'),
            'ee_frame': LaunchConfiguration('ee_frame'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'use_fk': LaunchConfiguration('use_fk'),
            'target_x': LaunchConfiguration('target_x'),
            'target_y': LaunchConfiguration('target_y'),
            'target_z': LaunchConfiguration('target_z'),
            'use_sim_time': True,  # CRITICAL: Must use simulation time
        }],
        remappings=[
            ('/joint_states', '/joint_states'),
            ('/target_position', '/target_position'),
        ]
    )

    # Policy inference node
    policy_inference_node = Node(
        package='parol6_inference',
        executable='policy_inference.py',
        name='policy_inference',
        output='screen',
        parameters=[{
            'policy_path': LaunchConfiguration('policy_path'),
            'policy_type': LaunchConfiguration('policy_type'),
            'control_rate': LaunchConfiguration('control_rate'),
            'use_ema_smoothing': LaunchConfiguration('use_ema_smoothing'),
            'ema_alpha': LaunchConfiguration('ema_alpha'),
            'trajectory_time': LaunchConfiguration('trajectory_time'),
            'use_sim_time': True,  # CRITICAL: Must use simulation time
        }],
        condition=IfCondition(LaunchConfiguration('run_inference'))
    )

    # Interactive marker node for visual goal setting
    interactive_marker_node = Node(
        package='parol6_inference',
        executable='interactive_goal_setter.py',
        name='interactive_goal_setter',
        output='screen',
        parameters=[{
            'target_x': LaunchConfiguration('target_x'),
            'target_y': LaunchConfiguration('target_y'),
            'target_z': LaunchConfiguration('target_z'),
            'base_frame': LaunchConfiguration('base_frame'),
            'marker_scale': LaunchConfiguration('marker_scale'),
            'use_sim_time': True,  # CRITICAL: Must use simulation time
        }],
        condition=IfCondition(LaunchConfiguration('use_interactive_marker'))
    )

    return LaunchDescription([
        # Declare arguments
        policy_path_arg,
        policy_type_arg,
        target_x_arg,
        target_y_arg,
        target_z_arg,
        base_frame_arg,
        ee_frame_arg,
        publish_rate_arg,
        use_fk_arg,
        control_rate_arg,
        use_ema_smoothing_arg,
        ema_alpha_arg,
        trajectory_time_arg,
        run_inference_arg,
        use_interactive_marker_arg,
        marker_scale_arg,

        # Launch nodes
        observation_publisher_node,
        policy_inference_node,
        interactive_marker_node,
    ])
