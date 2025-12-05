import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from moveit_configs_utils import MoveItConfigsBuilder
import os
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration


def generate_launch_description():

    micro_ros_agent_node = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        arguments=['serial', '--dev', '/dev/ttyACM0'],
        output='screen',
    )

    controller = IncludeLaunchDescription(
            os.path.join(
                get_package_share_directory("parol6_controller"),
                "launch",
                "controller.launch.py"
            ),
            launch_arguments={"is_sim": "False"}.items()
        )
    
    moveit_config = (
        MoveItConfigsBuilder("parol6", package_name="parol6_moveit")
        .robot_description(file_path=os.path.join(get_package_share_directory("parol6_description"), "urdf", "parol6.urdf.xacro"))
        .robot_description_semantic(file_path="config/parol6.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        # .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(), 
            {"publish_robot_description_semantic": True},
            {"allow_trajectory_execution": True},
            {"capabilities": ""},
            {"disable_capabilities": ""},
            {"jiggle_fraction": 0.05},
            {"max_safe_path_cost": 1},
            {"max_velocity_scaling_factor": 0.1},
            {"max_acceleration_scaling_factor": 0.1}
        ],
        arguments=["--ros-args", "--log-level", "info"]
    )
    

    
    return LaunchDescription([
        controller,
        move_group_node,
        micro_ros_agent_node
    ])