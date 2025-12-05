from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from moveit_configs_utils import MoveItConfigsBuilder
import os
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration

def generate_launch_description():
    is_sim = LaunchConfiguration("is_sim")
    
    is_sim_arg = DeclareLaunchArgument(
        "is_sim",
        default_value="True"
    )
    
    parol6_description_dir = get_package_share_directory('parol6_description')

    model_arg = DeclareLaunchArgument(name='model', default_value=os.path.join(
                                        parol6_description_dir, 'urdf', 'parol6.urdf.xacro'
                                        ),
                                      description='Absolute path to robot urdf file')

    robot_description = ParameterValue(Command(['xacro ', LaunchConfiguration('model')]),
                                       value_type=str)
    

    # Add argument for trajectory execution
    allow_trajectory_execution_arg = DeclareLaunchArgument(
        "allow_trajectory_execution",
        default_value="false"  # Set to false to disable trajectory execution
    )
    
    allow_trajectory_execution = LaunchConfiguration("allow_trajectory_execution")

    moveit_config = (
        MoveItConfigsBuilder("parol6", package_name="parol6_moveit")
        .robot_description(file_path=os.path.join(get_package_share_directory("parol6_description"), "urdf", "parol6.urdf.xacro"))
        .robot_description_semantic(file_path="config/parol6.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(), 
            {"use_sim_time": is_sim}, 
            {"publish_robot_description_semantic": True},
            {"allow_trajectory_execution": allow_trajectory_execution}  # Add this parameter
        ],
        arguments=["--ros-args", "--log-level", "info"]
    )

    rviz_config = os.path.join(get_package_share_directory("parol6_moveit"), "config", "moveit.rviz")

    # rviz_node = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     name="rviz2",
    #     output="log",
    #     arguments=["-d", rviz_config],
    #     parameters=[
    #         moveit_config.robot_description,
    #         moveit_config.robot_description_semantic,
    #         moveit_config.robot_description_kinematics,
    #         moveit_config.joint_limits,
    #     ],
    # )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(parol6_description_dir, 'rviz', 'display.rviz')],
    )
    
    # Add Flask GUI node
    flask_gui_node = Node(
        package='parol6_gui',  # Replace with your package name
        executable='moveit_ik_solver.py',
        name='moveit_ik_solver',
        output='screen'
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    return LaunchDescription([
        is_sim_arg,
        model_arg,
        allow_trajectory_execution_arg,  # Add this argument
        move_group_node,
        rviz_node,
        flask_gui_node,  # Add Flask GUI node
        robot_state_publisher  # Add state publisher
    ])