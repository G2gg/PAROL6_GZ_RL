#!/usr/bin/env python3
"""
Interactive Goal Setter Node for PAROL6 Reaching Task

Provides an interactive RViz marker that can be dragged to set target positions.
Publishes goal updates to /target_position topic.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker, InteractiveMarkerFeedback
from interactive_markers import InteractiveMarkerServer


class InteractiveGoalSetter(Node):
    """Creates interactive RViz marker for setting PAROL6 reach goals."""

    def __init__(self):
        super().__init__('interactive_goal_setter')

        # Declare parameters
        self.declare_parameter('target_x', 0.20)
        self.declare_parameter('target_y', 0.03)
        self.declare_parameter('target_z', 0.24)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('marker_scale', 0.05)

        # Get parameters
        target_x = self.get_parameter('target_x').value
        target_y = self.get_parameter('target_y').value
        target_z = self.get_parameter('target_z').value
        self.base_frame = self.get_parameter('base_frame').value
        self.marker_scale = self.get_parameter('marker_scale').value

        # Current target position
        self.target_pos = Point()
        self.target_pos.x = float(target_x)
        self.target_pos.y = float(target_y)
        self.target_pos.z = float(target_z)

        # Publisher for target position
        self.target_pub = self.create_publisher(
            Point,
            '/target_position',
            10
        )

        # Interactive marker server
        self.server = InteractiveMarkerServer(self, 'goal_marker')

        # Create the interactive marker
        self.create_interactive_marker()

        # Apply the changes
        self.server.applyChanges()

        self.get_logger().info('=== Interactive Goal Setter Started ===')
        self.get_logger().info(f'Initial target: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})')
        self.get_logger().info(f'Base frame: {self.base_frame}')
        self.get_logger().info('Drag the RED SPHERE in RViz to set new goals!')

    def create_interactive_marker(self):
        """Create the interactive marker for goal setting."""

        # Create interactive marker
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.base_frame
        int_marker.name = 'target_goal'
        int_marker.description = 'PAROL6 Target Goal\n(Drag to move)'
        int_marker.pose.position.x = self.target_pos.x
        int_marker.pose.position.y = self.target_pos.y
        int_marker.pose.position.z = self.target_pos.z
        int_marker.scale = self.marker_scale * 4.0  # Make interaction area larger

        # Create a red sphere marker for visualization
        sphere_marker = Marker()
        sphere_marker.type = Marker.SPHERE
        sphere_marker.scale.x = self.marker_scale
        sphere_marker.scale.y = self.marker_scale
        sphere_marker.scale.z = self.marker_scale
        sphere_marker.color.r = 1.0
        sphere_marker.color.g = 0.0
        sphere_marker.color.b = 0.0
        sphere_marker.color.a = 0.8

        # Create a control that contains the sphere
        sphere_control = InteractiveMarkerControl()
        sphere_control.always_visible = True
        sphere_control.markers.append(sphere_marker)
        int_marker.controls.append(sphere_control)

        # Create movement controls (X, Y, Z axes)

        # X-axis control (red arrow)
        control_x = InteractiveMarkerControl()
        control_x.name = 'move_x'
        control_x.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control_x.orientation.w = 1.0
        control_x.orientation.x = 1.0
        control_x.orientation.y = 0.0
        control_x.orientation.z = 0.0
        int_marker.controls.append(control_x)

        # Y-axis control (green arrow)
        control_y = InteractiveMarkerControl()
        control_y.name = 'move_y'
        control_y.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control_y.orientation.w = 1.0
        control_y.orientation.x = 0.0
        control_y.orientation.y = 0.0
        control_y.orientation.z = 1.0
        int_marker.controls.append(control_y)

        # Z-axis control (blue arrow)
        control_z = InteractiveMarkerControl()
        control_z.name = 'move_z'
        control_z.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control_z.orientation.w = 1.0
        control_z.orientation.x = 0.0
        control_z.orientation.y = 1.0
        control_z.orientation.z = 0.0
        int_marker.controls.append(control_z)

        # 3D movement control (move freely)
        control_3d = InteractiveMarkerControl()
        control_3d.name = 'move_3d'
        control_3d.interaction_mode = InteractiveMarkerControl.MOVE_3D
        control_3d.orientation.w = 1.0
        control_3d.orientation.x = 0.0
        control_3d.orientation.y = 1.0
        control_3d.orientation.z = 0.0
        int_marker.controls.append(control_3d)

        # Insert the marker and set the feedback callback
        self.server.insert(int_marker, feedback_callback=self.marker_feedback)

    def marker_feedback(self, feedback):
        """
        Handle interactive marker feedback.

        Args:
            feedback: InteractiveMarkerFeedback message
        """
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            # Update target position from marker pose
            self.target_pos.x = feedback.pose.position.x
            self.target_pos.y = feedback.pose.position.y
            self.target_pos.z = feedback.pose.position.z

            # Publish new target position
            self.target_pub.publish(self.target_pos)

            # Log the update
            self.get_logger().info(
                f'New target: ({self.target_pos.x:.3f}, '
                f'{self.target_pos.y:.3f}, '
                f'{self.target_pos.z:.3f})',
                throttle_duration_sec=0.5  # Log at most every 0.5 seconds
            )

        # Apply changes to update the marker server
        self.server.applyChanges()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = InteractiveGoalSetter()
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
