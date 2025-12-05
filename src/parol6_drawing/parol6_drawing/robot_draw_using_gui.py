#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from flask import Flask, request, render_template, jsonify
import numpy as np
import threading
import math
import time
from svgpathtools import svg2paths
import os
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.msg import PositionIKRequest, RobotState
from moveit_msgs.srv import GetPositionIK
import threading
import logging
import tf_transformations
import xml.etree.ElementTree as ET
from ament_index_python.resources import get_resource

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('robot_svg_drawer')
_, package_share_path = get_resource('packages', 'parol6_gui')
templates_path = os.path.join(package_share_path, 'share', 'parol6_gui', 'templates')


app = Flask(__name__)
app.template_folder = templates_path

class RobotSVGDrawer(Node):
    def __init__(self):
        super().__init__('robot_svg_drawer')
        
        # Robot parameters
        self.a1 = 0.11050  # 110.50 mm
        self.a2 = 0.02342  # 23.42 mm
        self.a3 = 0.18000  # 180.00 mm
        self.a4 = 0.04350  # 43.50 mm
        self.a5 = 0.17635  # 176.35 mm
        self.a6 = 0.06280  # 62.8 mm
        self.a7 = 0.04525  # 45.25 mm

        self.joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        # self.joint_limits = {
        #     'J1': [-1.083 + 0.0698132, 2.148 - 0.0698132],
        #     'J2': [-0.960 + 0.0698132, 0.855 - 0.0698132],
        #     'J3': [-1.259 + 0.0698132, 0.995 - 0.0698132],
        #     'J4': [-1.841 + 0.0698132, 1.841 - 0.0698132],
        #     'J5': [-1.571 + 0.0698132, 1.571 - 0.0698132],
        #     'J6': [-3.142 + 0.0698132, 3.142 - 0.0698132]
        # }

        self.joint_limits = {
            'J1': [-1.083, 2.148],
            'J2': [-0.960, 0.855],
            'J3': [-1.259, 0.995],
            'J4': [-1.841, 1.841],
            'J5': [-1.571, 1.571],
            'J6': [-3.142, 3.142]
        }
        
        # Drawing parameters - A4 paper (210x297mm)
        self.paper_width = 0.210  # meters
        self.paper_height = 0.297  # meters
        
        # Default paper position and orientation (can be updated later)
        self.paper_pose = {
            'position': [0.3, 0.0, 0.1],  # x, y, z position in meters
            'orientation': [0.0, 0.0, 0.0, 1.0]  # x, y, z, w quaternion
        }
        
        # Drawing parameters
        self.pen_up_offset = 0.01  # 1cm above the paper when pen is up
        self.speed_drawing = 0.05  # m/s when drawing
        self.speed_travel = 0.1    # m/s when moving without drawing
        self.approach_distance = 0.05  # distance to approach from before drawing
        
        self.current_joint_states = None
        
        # Publishers and subscribers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10)
            
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
            
        # MoveIt IK service client
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for IK service...')
            
        # Initialize current state
        self.is_drawing = False
        self.get_logger().info('Robot SVG Drawer initialized and ready!')
            
    def joint_state_callback(self, msg):
        """Store the current joint states"""
        # Filter for only the robot arm joints
        indices = [i for i, name in enumerate(msg.name) if name in self.joint_names]
        if indices:
            self.current_joint_states = {
                name: msg.position[i] for i, name in enumerate(msg.name) if name in self.joint_names
            }
            
    def compute_ik(self, pose):
        """Compute inverse kinematics for a given pose"""
        request = GetPositionIK.Request()  # Use .Request() to create the proper request type
        request.ik_request = PositionIKRequest()
        request.ik_request.group_name = "arm"  # Your arm's move_group name
        request.ik_request.pose_stamped.header.frame_id = "base_link"  # Reference frame
        request.ik_request.pose_stamped.pose = pose
        
        # Set the seed state from current joint positions if available
        if self.current_joint_states:
            robot_state = RobotState()
            robot_state.joint_state.name = self.joint_names
            robot_state.joint_state.position = [self.current_joint_states[name] for name in self.joint_names]
            request.ik_request.robot_state = robot_state
        
        try:
            
            response = self.ik_client.call(request)
            
            if response.error_code.val != 1:  # 1 means SUCCESS
                self.get_logger().error(f'IK failed with error code: {response.error_code.val}')
                return None
                
            # Extract joint positions from solution
            joint_positions = {}
            for name, position in zip(response.solution.joint_state.name, response.solution.joint_state.position):
                if name in self.joint_names:
                    joint_positions[name] = position
                    
            # Check if all required joints are present
            for name in self.joint_names:
                if name not in joint_positions:
                    self.get_logger().error(f'Joint {name} missing from IK solution')
                    return None
                    
            return [joint_positions[name] for name in self.joint_names]
        except Exception as e:
            self.get_logger().error(f'Exception during IK computation: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None
        
    def paper_to_robot_coordinates(self, x, z):
        """
        Convert paper coordinates (0-1 for both width and height) to robot coordinates
        """
        # Create transformation matrix from paper coordinates to robot base
        translation_matrix = tf_transformations.translation_matrix(self.paper_pose['position'])
        rotation_matrix = tf_transformations.quaternion_matrix(self.paper_pose['orientation'])
        transform_matrix = np.dot(translation_matrix, rotation_matrix)
        
        # Calculate the point in paper frame
        paper_x = x * self.paper_width 
        paper_z = z * self.paper_height
        paper_y = 0.0  # On the paper surface
        
        # Log the coordinates for debugging
        self.get_logger().debug(f"Paper coords: ({paper_x}, {paper_y}, {paper_z})")
        
        # Check if svg_scale attributes exist and use them
        paper_x *= getattr(self, 'svg_scale_x', 1.0)
        paper_z *= getattr(self, 'svg_scale_z', 1.0)
        
        # Transform to robot frame
        point_paper = np.array([paper_x, paper_y, paper_z, 1.0])
        point_robot = np.dot(transform_matrix, point_paper)
        
        self.get_logger().debug(f"Robot coords: {point_robot[:3]}")
        
        return point_robot[:3]  # Return just the x, y, z coordinates
        
    def plan_trajectory_for_path(self, path_points):
        """
        Plan a trajectory through the given path points
        path_points: list of (x, y) normalized coordinates on the paper (0-1)
        Returns a JointTrajectory message
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        
        # Starting point - pen up position
        start_point = path_points[0]
        robot_coords = self.paper_to_robot_coordinates(start_point[0], start_point[1])
        
        # Add pen_up_offset to z
        approach_coords = [robot_coords[0], robot_coords[1] + self.pen_up_offset, robot_coords[2]]
        
        # Compute IK for approach position
        joint_positions = self.compute_ik(self.coords_to_pose(approach_coords))
        if not joint_positions:
            self.get_logger().error("Failed to compute IK for approach position")
            return None
            
        # First point - approach above starting point
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = 1
        trajectory.points.append(point)
        
        # Lower to starting point
        joint_positions = self.compute_ik(self.coords_to_pose(robot_coords))
        if not joint_positions:
            self.get_logger().error("Failed to compute IK for drawing start position")
            return None
            
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = 2
        trajectory.points.append(point)
        
        # Follow the path
        time_from_start = 2.0
        last_point = start_point
        
        for i, (x, z) in enumerate(path_points[1:]):
            robot_coords = self.paper_to_robot_coordinates(x, z)
            joint_positions = self.compute_ik(self.coords_to_pose(robot_coords))
            
            if not joint_positions:
                self.get_logger().error(f"Failed to compute IK for path point {i+1}")
                return None
                
            # Calculate time based on distance and speed
            distance = math.sqrt((x - last_point[0])**2 + (z - last_point[1])**2) * max(self.paper_width, self.paper_height)
            time_from_start += distance / self.speed_drawing
            
            point = JointTrajectoryPoint()
            point.positions = joint_positions
            point.time_from_start.sec = int(time_from_start)
            point.time_from_start.nanosec = int((time_from_start - int(time_from_start)) * 1e9)
            trajectory.points.append(point)
            
            last_point = (x, z)
            
        # Final point - lift pen up
        end_coords = self.paper_to_robot_coordinates(last_point[0], last_point[1])
        end_coords[1] += self.pen_up_offset
        
        joint_positions = self.compute_ik(self.coords_to_pose(end_coords))
        if not joint_positions:
            self.get_logger().error("Failed to compute IK for pen up position")
            return None
            
        time_from_start += 1.0
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = int(time_from_start)
        point.time_from_start.nanosec = int((time_from_start - int(time_from_start)) * 1e9)
        trajectory.points.append(point)
        
        return trajectory
        
    def coords_to_pose(self, coords):
        """Convert [x, y, z] coordinates to a ROS Pose message"""
        pose = Pose()
        pose.position.x = coords[0]
        pose.position.y = coords[1]
        pose.position.z = coords[2]
        
        # Set orientation for end effector (pen tip pointing down)
        pose.orientation.x = 0.0
        pose.orientation.y = 1.0
        pose.orientation.z = 0.0
        pose.orientation.w = 0.0
        
        return pose
        
    def process_svg(self, svg_filename, simplify_tolerance=0.001):
        """
        Process an SVG file and extract drawable paths
        Returns a list of paths, where each path is a list of (x, y) points
        normalized to 0-1 range for both width and height
        """
        self.get_logger().info(f"Processing SVG file: {svg_filename}")
        
        try:
            # Parse the SVG to get paths and attributes
            paths, attributes = svg2paths(svg_filename)
            
            # Extract the viewBox from the SVG to determine scaling
            tree = ET.parse(svg_filename)
            root = tree.getroot()
            viewbox = root.get('viewBox')
            
            if viewbox:
                min_x, min_y, width, height = map(float, viewbox.split())
                min_z = min_y
            else:
                # If no viewBox, try to get width and height attributes
                width = float(root.get('width', '100').rstrip('px'))
                height = float(root.get('height', '100').rstrip('px'))
                min_x, min_z = 0, 0
                
            self.get_logger().info(f"SVG dimensions: {width}x{height}")
            
            all_drawing_paths = []
            
            for path in paths:
                # Sample points along the path
                num_samples = max(100, int(path.length() / simplify_tolerance))
                path_points = []
                
                for i in range(num_samples + 1):
                    t = i / num_samples
                    point = path.point(t)
                    
                    # Convert to real coordinates and normalize to 0-1
                    x = (point.real - min_x) / width
                    z = (point.imag - min_z) / height
                    
                    path_points.append((x, z))
                    
                all_drawing_paths.append(path_points)
                
            return all_drawing_paths
                
        except Exception as e:
            self.get_logger().error(f"Error processing SVG: {str(e)}")
            return []
            
    def execute_drawing(self, svg_filename):
        """
        Main function to execute the drawing of an SVG file
        """
        if self.is_drawing:
            self.get_logger().warn("Already drawing. Please wait until current drawing is complete.")
            return False
            
        self.is_drawing = True
        try:
            # Process the SVG file
            paths = self.process_svg(svg_filename)
            
            if not paths:
                self.get_logger().error("No drawable paths found in SVG file.")
                self.is_drawing = False
                return False
                
            self.get_logger().info(f"Found {len(paths)} paths to draw")
            
            # Draw each path
            for i, path in enumerate(paths):
                self.get_logger().info(f"Drawing path {i+1}/{len(paths)}")
                
                # Plan trajectory for this path
                try:
                    trajectory = self.plan_trajectory_for_path(path)
                    
                    if trajectory:
                        # Execute the trajectory
                        self.get_logger().info(f"Executing trajectory with {len(trajectory.points)} points")
                        self.trajectory_pub.publish(trajectory)
                        
                        # Wait for trajectory to complete (estimate based on last point's time_from_start)
                        last_point = trajectory.points[-1]
                        duration = last_point.time_from_start.sec + (last_point.time_from_start.nanosec / 1e9)
                        self.get_logger().info(f"Waiting {duration} seconds for trajectory to complete")
                        time.sleep(duration + 1.0)  # Add a small buffer
                    else:
                        self.get_logger().error(f"Failed to plan trajectory for path {i+1}")
                except Exception as e:
                    self.get_logger().error(f"Error planning trajectory for path {i+1}: {str(e)}")
                    import traceback
                    self.get_logger().error(traceback.format_exc())
                
            self.get_logger().info("Drawing completed successfully!")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error during drawing execution: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
        finally:
            self.is_drawing = False
            
    def calibrate_paper_position(self, corners):
        """
        Calibrate the paper position and orientation based on three corner points
        corners: list of 3 [x, y, z] points in robot coordinates
        """
        if len(corners) != 3:
            self.get_logger().error("Calibration requires exactly 3 corner points")
            return False
            
        # Define which corners these are (bottom-left, bottom-right, top-left)
        bottom_left = np.array(corners[0])
        bottom_right = np.array(corners[1])
        top_left = np.array(corners[2])
        
        # Calculate paper dimensions
        width_vector = bottom_right - bottom_left
        height_vector = top_left - bottom_left
        
        self.paper_width = np.linalg.norm(width_vector)
        self.paper_height = np.linalg.norm(height_vector)
        
        # Calculate paper orientation
        x_axis = width_vector / np.linalg.norm(width_vector)
        y_axis = np.cross(height_vector, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)

        self.svg_scale_x = 0.7  # Scale factor for X dimension
        self.svg_scale_z = 0.7  # Scale factor for Z dimension
        
        # Create rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # Convert to quaternion
        q = tf_transformations.quaternion_from_matrix(
            np.vstack([
                np.hstack([rotation_matrix, np.zeros((3, 1))]),
                [0, 0, 0, 1]
            ])
        )
        
        # Update paper pose
        self.paper_pose['position'] = bottom_left.tolist()
        self.paper_pose['orientation'] = list(q)
        
        self.get_logger().info(f"Paper calibrated: width={self.paper_width}m, height={self.paper_height}m")
        self.get_logger().info(f"Paper position: {self.paper_pose['position']}")
        self.get_logger().info(f"Paper orientation: {self.paper_pose['orientation']}")
        
        return True

# Global instance for the Flask app to access
robot_drawer = None

@app.route('/')
def index():
    return render_template('svg_index.html')

@app.route('/upload', methods=['POST'])
def upload_svg():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
        
    if file and file.filename.endswith('.svg'):
        # Save the file to a temporary location
        filepath = os.path.join('/tmp', file.filename)
        file.save(filepath)
        
        # Process the SVG in a separate thread
        threading.Thread(target=lambda: robot_drawer.execute_drawing(filepath)).start()
        
        return jsonify({'success': True, 'message': 'Drawing started'})
    else:
        return jsonify({'success': False, 'message': 'File must be an SVG'})

@app.route('/calibrate', methods=['POST'])
def calibrate():
    data = request.json
    if not data or 'corners' not in data:
        return jsonify({'success': False, 'message': 'No corner points provided'})
        
    corners = data['corners']
    if len(corners) != 3:
        return jsonify({'success': False, 'message': 'Exactly 3 corner points required'})
        
    success = robot_drawer.calibrate_paper_position(corners)
    
    return jsonify({'success': success, 'message': 'Calibration successful' if success else 'Calibration failed'})

@app.route('/paper_info', methods=['GET'])
def paper_info():
    return jsonify({
        'width': robot_drawer.paper_width,
        'height': robot_drawer.paper_height,
        'position': robot_drawer.paper_pose['position'],
        'orientation': robot_drawer.paper_pose['orientation']
    })

def run_flask():
    app.run(host='0.0.0.0', port=5000)

def main():
    global robot_drawer
    
    # Initialize ROS node
    rclpy.init()
    robot_drawer = RobotSVGDrawer()
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    try:
        # Run the ROS node
        rclpy.spin(robot_drawer)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        robot_drawer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()