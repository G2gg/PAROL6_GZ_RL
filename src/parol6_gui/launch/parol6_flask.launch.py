import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration

def generate_launch_description():
    parol6_description_dir = get_package_share_directory('parol6_description')

    model_arg = DeclareLaunchArgument(name='model', default_value=os.path.join(
                                        parol6_description_dir, 'urdf', 'parol6.urdf.xacro'
                                        ),
                                      description='Absolute path to robot urdf file')

    robot_description = ParameterValue(Command(['xacro ', LaunchConfiguration('model')]),
                                       value_type=str)
    
    # Launch RViz2 for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(parol6_description_dir, 'rviz', 'display.rviz')],
    )
    
    # Launch Flask GUI
    flask_gui_node = Node(
        package='parol6_gui',  # Replace with your package name
        executable='parol6_flask_gui.py',
        name='parol6_flask_gui',
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
        model_arg,
        robot_state_publisher,
        rviz_node,
        flask_gui_node
    ])