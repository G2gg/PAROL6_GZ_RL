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
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    is_sim = LaunchConfiguration("is_sim")
    svg_path = LaunchConfiguration("svg_path")
    
    is_sim_arg = DeclareLaunchArgument(
        "is_sim",
        default_value="True"
    )

    svg_arg = DeclareLaunchArgument(
        "svg_path",
        default_value=os.path.join(
            get_package_share_directory('parol6_drawing'),
            'share', 'parol6_drawing', 'svg_files', 'test_drawing.svg'
        ),
        description='Path to the input SVG file to draw'
    )


    gazebo = IncludeLaunchDescription(
            os.path.join(
                get_package_share_directory("parol6_description"),
                "launch",
                "gz_sim.launch.py"
            )
        )
    
    controller = IncludeLaunchDescription(
            os.path.join(
                get_package_share_directory("parol6_controller"),
                "launch",
                "controller.launch.py"
            ),
            launch_arguments={"is_sim": "True"}.items()
        )
    
    moveit_config = (
        MoveItConfigsBuilder("parol6", package_name="parol6_moveit")
        .robot_description(file_path=os.path.join(get_package_share_directory("parol6_description"), "urdf", "parol6.urdf.xacro"))
        .robot_description_semantic(file_path="config/parol6.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_pipelines(pipelines=["ompl"])
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

    robot_draw_node = Node(
        package='parol6_drawing',  
        executable='robot_draw_using_gui',
        name='robot_draw_using_gui',
        output='screen'
    )
    

    return LaunchDescription([
        is_sim_arg,
        # svg_arg,
        gazebo,
        controller,
        move_group_node,
        # robot_draw_node
        
    ])