#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class PathTracker(Node):
    def __init__(self):
        super().__init__('path_tracker')
        
        self.path_pub = self.create_publisher(Path, '/end_effector_path', 10)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.path = Path()
        self.path.header.frame_id = "base_link"  
        
        self.timer = self.create_timer(0.1, self.update_path)  # 10 Hz
        
        self.get_logger().info('Path tracker node started')
        
    def update_path(self):
        try:
            # Get transform from base to end-effector
            # Replace 'end_effector_link' with your actual end-effector frame name
            # Common names: 'tool0', 'ee_link', 'gripper_link', 'end_effector_link'
            transform = self.tf_buffer.lookup_transform(
                'base_link', 
                'L6',  # Change this to your end-effector frame
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )
            
            # Create pose stamped
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "base_link"
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            
            # Add to path
            self.path.poses.append(pose)
            self.path.header.stamp = self.get_clock().now().to_msg()
            
            if len(self.path.poses) > 1000:
                self.path.poses.pop(0)
            
            # Publish path
            self.path_pub.publish(self.path)
            
        except TransformException as ex:
            self.get_logger().debug(f'Could not transform: {ex}')
            return

def main(args=None):
    rclpy.init(args=args)
    
    path_tracker = PathTracker()
    
    try:
        rclpy.spin(path_tracker)
    except KeyboardInterrupt:
        pass
    finally:
        path_tracker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()