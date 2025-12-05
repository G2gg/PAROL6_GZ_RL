#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetPlanningScene
from moveit_msgs.msg import PositionIKRequest, RobotState, PlanningSceneComponents
from tf2_ros import TransformListener, Buffer
import numpy as np
import threading
import time
import serial
import json
from flask import Flask, render_template, request, jsonify
from ament_index_python.resources import get_resource
import os
import math

_, package_share_path = get_resource('packages', 'parol6_gui')
templates_path = os.path.join(package_share_path, 'share', 'parol6_gui', 'templates')

# Define joint limits - adjust these to match your robot
JOINT_LIMITS = {
    'J1': {'min': -62.047, 'max': 123.047},  # Angular limits in degrees
    'J2': {'min': -70, 'max': 52.0},
    'J3': {'min': -107.86, 'max': 72.0},
    'J4': {'min': -105.47, 'max': 105.47},
    'J5': {'min': -90, 'max': 90},
    'J6': {'min': -180, 'max': 180},
}

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control_node')
        
        # Publisher for joint states (to visualize in RViz)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # Current joint positions in degrees
        self.joint_positions = {
            'J1': 0.0, 'J2': 0.0, 'J3': 0.0, 
            'J4': 0.0, 'J5': 0.0, 'J6': 0.0
        }
        
        # For getting workspace limits
        self.planning_scene_client = self.create_client(
            GetPlanningScene, '/get_planning_scene')
        while not self.planning_scene_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Planning scene service not available, waiting...')

        # Create a client for the FK service
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('FK service not available, waiting...')

        # Create a client for the IK service
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('IK service not available, waiting...')
        
        # Setup serial connection to Teensy
        self.serial_port = None
        try:
            self.serial_port = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
            self.get_logger().info('Connected to Teensy')
            time.sleep(2)  # Allow time for serial connection to stabilize
        except serial.SerialException:
            self.get_logger().error('Failed to connect to Teensy. Check connection.')
        
        # For TF transforms to get end effector position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Store current end effector position
        self.end_effector_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
        # Timer to update end effector position
        self.tf_timer = self.create_timer(0.2, self.update_end_effector_position)
        # Flag to indicate if robot is homed
        self.is_homed = True
        
        # Start publishing joint states
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        
        self.get_logger().info('Robot Control Node initialized')

    def publish_joint_states(self):
        # Create and publish JointState message (converted to radians for RViz)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_positions.keys())
        # Convert degrees to radians for RViz
        msg.position = [self.deg_to_rad(pos) for pos in self.joint_positions.values()]
        self.joint_pub.publish(msg)

    def deg_to_rad(self, deg):
        return deg * 0.01745329  # pi/180

    def rad_to_deg(self, rad):
        return rad * 57.2957795  # 180/pi

    def home_robot(self):
        if self.serial_port:
            try:
                # Send home command to Teensy
                command = "home\n"
                self.serial_port.write(command.encode())
                self.get_logger().info('Sent home command to Teensy')
                
                # Wait for the "Homing complete" message
                self.get_logger().info('Waiting for homing to complete...')
                
                # Set a timeout for homing (180 seconds)
                timeout = time.time() + 180
                homing_complete = False
                
                while time.time() < timeout and not homing_complete:
                    if self.serial_port.in_waiting:
                        line = self.serial_port.readline().decode('utf-8').strip()
                        self.get_logger().info(f'Teensy: {line}')
                        if "Homing complete" in line:
                            homing_complete = True
                            self.is_homed = True
                            # Reset joint positions to standby positions (all zeros as per your code)
                            for joint in self.joint_positions:
                                self.joint_positions[joint] = 0.0
                            self.publish_joint_states()
                
                if homing_complete:
                    self.get_logger().info('Homing completed successfully')
                    return True
                else:
                    self.get_logger().warn('Homing timed out')
                    return False
                    
            except Exception as e:
                self.get_logger().error(f'Failed to send home command: {str(e)}')
                return False
        else:
            self.get_logger().warn('Not connected to Teensy, home command not sent')
            return False

    def move_robot(self):
        if self.serial_port:
            try:
                # Format according to your Teensy code expectations (comma-separated values)
                joint_cmd = f"{self.joint_positions['J1']:.1f},{self.joint_positions['J2']:.1f},"
                joint_cmd += f"{self.joint_positions['J3']:.1f},{self.joint_positions['J4']:.1f},"
                joint_cmd += f"{self.joint_positions['J5']:.1f},{self.joint_positions['J6']:.1f}\n"
                
                self.serial_port.write(joint_cmd.encode())
                self.get_logger().info(f'Sent to Teensy: {joint_cmd.strip()}')
                
                # Wait for "All positions reached" confirmation
                timeout = time.time() + 60  # 60 second timeout
                while time.time() < timeout:
                    if self.serial_port.in_waiting:
                        line = self.serial_port.readline().decode('utf-8').strip()
                        self.get_logger().info(f'Teensy: {line}')
                        if "All positions reached" in line:
                            self.get_logger().info('Robot reached target position')
                            return True
                
                self.get_logger().warn('No position confirmation received')
                return True  # Still return success as the command was sent
                
            except Exception as e:
                self.get_logger().error(f'Failed to send movement command: {str(e)}')
                return False
        else:
            self.get_logger().warn('Not connected to Teensy, movement command not sent')
            return False

    def update_joints(self, joint_values):
        # Update joint positions with new values (in degrees)
        for joint, value in joint_values.items():
            if joint in self.joint_positions:
                # Ensure value is within limits
                if joint in JOINT_LIMITS:
                    value = max(JOINT_LIMITS[joint]['min'], min(value, JOINT_LIMITS[joint]['max']))
                self.joint_positions[joint] = value
        
        # Publish updated joint states
        self.publish_joint_states()
    
    def calculate_fk(self):
        """Calculate forward kinematics to get end effector position from joint angles."""
        try:
            request = GetPositionFK.Request()
            request.fk_link_names = ['L6']  
            
            # Create current robot state
            joint_state = JointState()
            joint_state.name = list(self.joint_positions.keys())
            joint_state.position = [self.deg_to_rad(pos) for pos in self.joint_positions.values()]

            robot_state = RobotState()
            robot_state.joint_state = joint_state
            request.robot_state = robot_state

            future = self.fk_client.call_async(request)

            timeout_sec = 5.0
            start_time = time.time()
            while not future.done() and (time.time() - start_time) < timeout_sec:
                rclpy.spin_once(self, timeout_sec=0.1)

            if not future.done():
                self.get_logger().error('FK calculation timed out')
                return self.end_effector_position  # Return last known

            response = future.result()

            if response.error_code.val == 1:  # SUCCESS
                pose = response.pose_stamped[0].pose
                # Convert to centimeters
                self.end_effector_position['x'] = pose.position.x * 100
                self.end_effector_position['y'] = pose.position.y * 100
                self.end_effector_position['z'] = pose.position.z * 100
                self.get_logger().info(f'FK position: {self.end_effector_position}')
            else:
                self.get_logger().error(f'FK failed with error code: {response.error_code.val}')

        except Exception as e:
            self.get_logger().error(f'Error in FK calculation: {str(e)}')

        return self.end_effector_position
    def calculate_ik(self, x, y, z, orientation_w=1.0):
        """
        Calculate inverse kinematics for a given Cartesian position
        Returns joint angles in degrees
        """
        try:
            request = GetPositionIK.Request()
            request.ik_request = PositionIKRequest()
            request.ik_request.group_name = "arm"  # Make sure this matches your MoveIt SRDF group

            request.ik_request.pose_stamped = PoseStamped()
            request.ik_request.pose_stamped.header.frame_id = "base_link"
            request.ik_request.pose_stamped.pose.position.x = float(x)
            request.ik_request.pose_stamped.pose.position.y = float(y)
            request.ik_request.pose_stamped.pose.position.z = float(z)
            request.ik_request.pose_stamped.pose.orientation.x = 0.0
            request.ik_request.pose_stamped.pose.orientation.y = 0.0
            request.ik_request.pose_stamped.pose.orientation.z = 0.0
            request.ik_request.pose_stamped.pose.orientation.w = float(orientation_w)

            joint_state = JointState()
            joint_state.name = list(self.joint_positions.keys())
            joint_state.position = [self.deg_to_rad(pos) for pos in self.joint_positions.values()]

            robot_state = RobotState()
            robot_state.joint_state = joint_state
            request.ik_request.robot_state = robot_state

            future = self.ik_client.call_async(request)

            
            timeout_sec = 5.0  # 5 seconds timeout
            start_time = time.time()
            while not future.done() and (time.time() - start_time) < timeout_sec:
                rclpy.spin_once(self, timeout_sec=0.1)

            if not future.done():
                self.get_logger().error('IK calculation timed out')
                return False, {}

            response = future.result()

            if response.error_code.val == 1:  # SUCCESS
                result_joints = {}
                for i, joint_name in enumerate(response.solution.joint_state.name):
                    if joint_name in self.joint_positions:
                        angle_deg = self.rad_to_deg(response.solution.joint_state.position[i])
                        result_joints[joint_name] = angle_deg
                self.get_logger().info(f'IK solution found: {result_joints}')
                return True, result_joints
            else:
                self.get_logger().error(f'IK calculation failed with error code: {response.error_code.val}')
                return False, {}

        except Exception as e:
            self.get_logger().error(f'Error in IK calculation: {str(e)}')
            return False, {}
        
    def update_end_effector_position(self):
        """Update the current end effector position using TF"""
        try:
            # Get the transform from base_link to end_effector
            
            transform = self.tf_buffer.lookup_transform(
                'base_link', 'L6', rclpy.time.Time())
            
            # Update the stored position
            self.end_effector_position['x'] = transform.transform.translation.x
            self.end_effector_position['y'] = transform.transform.translation.y
            self.end_effector_position['z'] = transform.transform.translation.z
            
        except Exception as e:
            self.get_logger().debug(f'Could not get end effector transform: {str(e)}')

    def get_workspace_limits(self):
        """Get the workspace limits from MoveIt planning scene"""
        try:
            request = GetPlanningScene.Request()
            request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
            
            future = self.planning_scene_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            
            response = future.result()
            
            # These values should be adjusted based on your robot's actual workspace
            # This is a simplified example - in reality, you would extract these from 
            # the planning scene constraints or calculate from robot kinematics
            workspace_limits = {
                'x': {'min': 0.0 * 100, 'max': 0.6 * 100},  # Adjust these values
                'y': {'min': -0.4 * 100, 'max': 0.4 * 100}, # based on your robot's 
                'z': {'min': 0.0 * 100, 'max': 0.6 * 100}   # actual workspace
            }
            
            return workspace_limits
            
        except Exception as e:
            self.get_logger().error(f'Error getting workspace limits: {str(e)}')
            # Default limits if service fails
            return {
                'x': {'min': 0.0 * 100, 'max': 0.6 * 100},
                'y': {'min': -0.4 * 100, 'max': 0.4 * 100},
                'z': {'min': 0.0 * 100, 'max': 0.6 * 100}
            }
    
    def get_end_effector_position(self):
        """Return the current end effector position"""
        return self.end_effector_position
           
# Initialize ROS2 node in a separate thread
def ros_thread_func(node):
    rclpy.spin(node)

# Create Flask app
app = Flask(__name__)
app.template_folder = templates_path
# Global variable to store ROS2 node reference
ros_node = None

@app.route('/')
def index():
    return render_template('index_try.html', joint_limits=JOINT_LIMITS)

@app.route('/get_joint_values', methods=['GET'])
def get_joint_values():
    global ros_node
    if ros_node:
        return jsonify(ros_node.joint_positions)
    return jsonify({})

@app.route('/set_joint_values', methods=['POST'])
def set_joint_values():
    global ros_node
    if ros_node:
        joint_values = request.json
        ros_node.update_joints(joint_values)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'ROS node not initialized'})

@app.route('/get_end_effector_position', methods=['GET'])

def get_end_effector_position():
    global ros_node
    if ros_node:
        return jsonify(ros_node.calculate_fk())
    return jsonify({'x': 0.0, 'y': 0.0, 'z': 0.0})

@app.route('/get_workspace_limits', methods=['GET'])
def get_workspace_limits():
    global ros_node
    if ros_node:
        limits = ros_node.get_workspace_limits()
        return jsonify(limits)
    return jsonify({
        'x': {'min': 0.0, 'max': 0.6},
        'y': {'min': -0.4, 'max': 0.4},
        'z': {'min': 0.0, 'max': 0.6}
    })

@app.route('/move_robot', methods=['POST'])
def move_robot():
    global ros_node
    if ros_node:
        if not ros_node.is_homed:
            return jsonify({'status': 'error', 'message': 'Robot not homed yet. Please home the robot first.'})
            
        result = ros_node.move_robot()
        if result:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to send command to Teensy'})
    return jsonify({'status': 'error', 'message': 'ROS node not initialized'})

@app.route('/home_robot', methods=['POST'])
def home_robot():
    global ros_node
    if ros_node:
        result = ros_node.home_robot()
        if result:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Homing failed or timed out'})
    return jsonify({'status': 'error', 'message': 'ROS node not initialized'})

@app.route('/emergency_stop', methods=['POST'])
def emergency_stop():
    global ros_node
    if ros_node and ros_node.serial_port:
        try:
            # Send emergency stop command to Teensy
            command = "estop\n"
            ros_node.serial_port.write(command.encode())
            ros_node.get_logger().info('Sent emergency stop command to Teensy')
            return jsonify({'status': 'success'})
        except Exception as e:
            ros_node.get_logger().error(f'Failed to send emergency stop command: {str(e)}')
            return jsonify({'status': 'error', 'message': f'Failed to send emergency stop command: {str(e)}'})
    return jsonify({'status': 'error', 'message': 'ROS node not initialized or not connected to Teensy'})

@app.route('/resume_operation', methods=['POST'])
def resume_operation():
    global ros_node
    if ros_node and ros_node.serial_port:
        try:
            command = "resume\n"
            ros_node.get_logger().info(f'Sending command to Teensy: {command.strip()}')
            ros_node.serial_port.write(command.encode())

            # Read back any serial response
            time.sleep(0.5)
            while ros_node.serial_port.in_waiting:
                line = ros_node.serial_port.readline().decode('utf-8').strip()
                ros_node.get_logger().info(f'Teensy: {line}')

            return jsonify({'status': 'success'})
        except Exception as e:
            ros_node.get_logger().error(f'Failed to send resume command: {str(e)}')
            return jsonify({'status': 'error', 'message': str(e)})
    return jsonify({'status': 'error', 'message': 'ROS node not initialized or not connected to Teensy'})


@app.route('/move_to_position', methods=['POST'])
def move_to_position():
    global ros_node
    if ros_node:
        if not ros_node.is_homed:
            return jsonify({'status': 'error', 'message': 'Robot not homed yet. Please home the robot first.'})
        
        data = request.json
        x = data.get('x', 0.0) / 100
        y = data.get('y', 0.0) / 100
        z = data.get('z', 0.0) / 100
        orientation_w = data.get('orientation_w', 1.0)
        
        # Calculate IK
        success, joint_values = ros_node.calculate_ik(x, y, z, orientation_w)
        
        if success:
            # Update joints immediately
            ros_node.update_joints(joint_values)

            # Move robot to new joints
            result = ros_node.move_robot()

            if result:
                # After move, update FK
                ros_node.calculate_fk()

                return jsonify({
                    'status': 'success',
                    'joint_values': joint_values
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to move robot after IK'
                })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to calculate inverse kinematics for the given position'
            })
    
    return jsonify({'status': 'error', 'message': 'ROS node not initialized'})

def main():
    global ros_node
    
    # Initialize ROS2
    rclpy.init()
    ros_node = RobotControlNode()
    
    # Start ROS2 spin in a separate thread
    ros_thread = threading.Thread(target=ros_thread_func, args=(ros_node,))
    ros_thread.daemon = True
    ros_thread.start()

    app.run(host='0.0.0.0', port=5000)
    
    # Cleanup
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()