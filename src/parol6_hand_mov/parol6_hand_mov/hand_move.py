#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
import time
from threading import Lock
from builtin_interfaces.msg import Duration

class HandTrackingArmController(Node):
    def __init__(self):
        super().__init__('hand_tracking_arm_controller')
        
        # Robot configuration
        self.joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        self.joint_limits = {
            'J1': [-1.083 + 0.0698132, 2.148 - 0.0698132],  # 3 degree offset in limits for safety ~ 0.0698132 radians
            'J2': [-1.221 + 0.0698132, 0.907 - 0.0698132],
            'J3': [-1.8825 + 0.0698132, 1.2566 - 0.0698132],
            'J4': [-1.841 + 0.0698132, 1.841 - 0.0698132],
            'J5': [-1.571 + 0.0698132, 1.571 - 0.0698132],
            'J6': [-3.142 + 0.0698132, 3.142 - 0.0698132]
        }
        
        # Publisher for joint trajectory command
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10)
        
        # Camera subscriber
        self.cv_bridge = CvBridge()
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10)
        
        # MediaPipe hand tracking setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Control parameters
        self.control_rate = 20  # Hz
        self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        self.lock = Lock()
        
        # State variables
        self.last_hand_landmarks = None
        self.last_valid_trajectory = self.create_default_trajectory()
        self.hand_visible = False
        
        # Smoothing filter parameters
        self.filter_alpha = 0.1
        self.filtered_joints = [0.0] * 6
        
        # Previous joint positions for velocity limiting
        self.prev_joint_positions = [0.0] * 6
        self.current_joint_positions = [0.0] * 6
        
        # Maximum velocity and acceleration limits
        self.max_velocity = 0.5  # Maximum joint velocity in radians/second
        self.max_accel = 0.2    # Maximum joint acceleration in radians/second^2
        
        # Time factor to slow down movements
        self.time_factor = 5.0  # Higher value = slower movements
        
        # Starting camera if not using ROS topics
        self.use_direct_camera = True
        self.camera_device_id = 0
        
        # Control mode parameters
        self.publishing_enabled = False  # Start with publishing disabled
        self.active_joint = None  # No joint selected initially
        
        # NEW: Hand position mapping parameters
        self.hand_position_min = 0.05  # Minimum y-coordinate value (near top of screen)
        self.hand_position_max = 0.95  # Maximum y-coordinate value (near bottom of screen)
        
        # NEW: Store the calibration positions for each joint
        self.calibration_positions = {
            'top': {joint: None for joint in self.joint_names},
            'bottom': {joint: None for joint in self.joint_names}
        }
        
        # NEW: Calibration mode
        self.calibration_mode = False
        self.calibration_stage = None  # 'top' or 'bottom'
        
        if self.use_direct_camera:
            self.setup_direct_camera()
            
        self.get_logger().info('Hand tracking arm controller initialized with direct position mapping')
    
    def setup_direct_camera(self):
        """Set up direct camera capture if not using ROS image topics"""
        self.cap = cv2.VideoCapture(self.camera_device_id)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera with ID {self.camera_device_id}")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Create thread for camera processing
        self.camera_timer = self.create_timer(1.0/30.0, self.process_direct_camera)
    
    def process_direct_camera(self):
        """Process frames directly from camera device"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame from camera")
            return
            
        # Flip image for more intuitive control
        frame = cv2.flip(frame, 1)
        
        # Process the frame
        self.process_image(frame)
    
    def camera_callback(self, msg):
        """Callback for ROS camera image topic"""
        if self.use_direct_camera:  # Skip if using direct camera
            return
            
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            # Flip image for more intuitive control
            cv_image = cv2.flip(cv_image, 1)
            self.process_image(cv_image)
        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {str(e)}')
    
    def process_image(self, image):
        """Process the camera image to detect hand landmarks"""
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image with MediaPipe Hands
        results = self.hands.process(rgb_image)
        
        # Draw hand landmarks on the image for visualization
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Use the first detected hand
            with self.lock:
                self.last_hand_landmarks = results.multi_hand_landmarks[0]
                self.hand_visible = True
                
                # Process hand position if in hand control mode
                if self.publishing_enabled and self.active_joint is not None and self.hand_visible:
                    # If in calibration mode, handle calibration
                    if self.calibration_mode and self.calibration_stage:
                        self.handle_calibration(self.last_hand_landmarks)
                    else:
                        # Otherwise, map hand position directly to joint angle
                        self.map_hand_to_joint_angle(self.last_hand_landmarks)
        else:
            with self.lock:
                self.hand_visible = False
        
        # Draw status information on image
        self.draw_status_info(image)
        
        # Draw hand position control range
        self.draw_control_range(image)
        
        # Display the image
        cv2.imshow("Arm Controller", image)
        key = cv2.waitKey(1) & 0xFF
        
        # Process keyboard input
        self.handle_keyboard_input(key)
    
    def draw_control_range(self, image):
        """Draw the hand position control range on the image"""
        h, w = image.shape[:2]
        
        # Draw top and bottom boundaries for control
        top_y = int(self.hand_position_min * h)
        bottom_y = int(self.hand_position_max * h)
        
        # Draw horizontal lines indicating control range
        cv2.line(image, (0, top_y), (w, top_y), (0, 255, 0), 2)
        cv2.line(image, (0, bottom_y), (w, bottom_y), (0, 255, 0), 2)
        
        # Draw text labels
        if self.active_joint:
            # Get the min/max angle for the active joint
            joint_min = self.joint_limits[self.active_joint][0]
            joint_max = self.joint_limits[self.active_joint][1]
            
            # Draw min/max angle labels
            cv2.putText(image, f"Max: {joint_max:.2f}", (w - 600, top_y - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, f"Min: {joint_min:.2f}", (w - 600, bottom_y + 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def draw_status_info(self, image):
        """Draw status information on the image"""
        h, w = image.shape[:2]
        
        # Draw status text
        status_color = (0, 255, 0) if self.publishing_enabled else (0, 0, 255)
        cv2.putText(image, f"Publishing: {'ON' if self.publishing_enabled else 'OFF'}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Show active joint
        joint_text = f"Active Joint: {self.active_joint}" if self.active_joint else "No joint selected"
        cv2.putText(image, joint_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show hand detection status
        hand_status = "Hand detected" if self.hand_visible else "No hand detected"
        cv2.putText(image, hand_status, (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.hand_visible else (0, 0, 255), 2)
        
        # Show calibration status if in calibration mode
        if self.calibration_mode:
            calib_text = f"Calibration: {self.calibration_stage or 'Ready'}"
            cv2.putText(image, calib_text, (20, 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Draw current joint positions
        for i, joint in enumerate(self.joint_names):
            value = self.current_joint_positions[i]
            pct = (value - self.joint_limits[joint][0]) / (self.joint_limits[joint][1] - self.joint_limits[joint][0])
            pct = max(0, min(1, pct))  # Clamp between 0 and 1
            
            # Change color if this is the active joint
            color = (0, 255, 255) if joint == self.active_joint else (200, 200, 200)
            
            # Draw joint name and value
            cv2.putText(image, f"{joint}: {value:.2f}", (w - 150, 30 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw control instructions
        instructions = [
            "SPACE: Start/Stop Publishing",
            "B: Control Base (J1)",
            "S: Control Shoulder (J2)",
            "E: Control Elbow (J3)",
            "R: Control Wrist Roll (J4)",
            "P: Control Wrist Pitch (J5)",
            "Y: Control Wrist Yaw (J6)",
            "C: Calibration Mode",
            "T: Set Top Position (in calibration)",
            "D: Set Bottom Position (in calibration)",
            "ESC: Exit"
        ]
        
        # Draw instructions at the bottom
        y_pos = h - 10 - (len(instructions) * 25)
        for instruction in instructions:
            cv2.putText(image, instruction, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 25
    
    def handle_keyboard_input(self, key):
        """Handle keyboard input for joint selection and control"""
        if key == 27:  # ESC key
            cv2.destroyAllWindows()
            if hasattr(self, 'cap'):
                self.cap.release()
            rclpy.shutdown()
        elif key == 32:  # SPACE key
            self.publishing_enabled = not self.publishing_enabled
            status = "enabled" if self.publishing_enabled else "disabled"
            self.get_logger().info(f"Publishing {status}")
        elif key == ord('b') or key == ord('B'):
            self.active_joint = 'J1'
            self.get_logger().info(f"Controlling Base (J1)")
        elif key == ord('s') or key == ord('S'):
            self.active_joint = 'J2'
            self.get_logger().info(f"Controlling Shoulder (J2)")
        elif key == ord('e') or key == ord('E'):
            self.active_joint = 'J3'
            self.get_logger().info(f"Controlling Elbow (J3)")
        elif key == ord('r') or key == ord('R'):
            self.active_joint = 'J4'
            self.get_logger().info(f"Controlling Wrist Roll (J4)")
        elif key == ord('p') or key == ord('P'):
            self.active_joint = 'J5'
            self.get_logger().info(f"Controlling Wrist Pitch (J5)")
        elif key == ord('y') or key == ord('Y'):
            self.active_joint = 'J6'
            self.get_logger().info(f"Controlling Wrist Yaw (J6)")
        elif key == ord('c') or key == ord('C'):
            # Toggle calibration mode
            self.calibration_mode = not self.calibration_mode
            self.calibration_stage = None
            status = "enabled" if self.calibration_mode else "disabled"
            self.get_logger().info(f"Calibration mode {status}")
        elif key == ord('t') or key == ord('T'):
            # Set top position for calibration
            if self.calibration_mode and self.active_joint:
                self.calibration_stage = 'top'
                self.get_logger().info(f"Move hand to desired top position for {self.active_joint}")
        elif key == ord('d') or key == ord('D'):
            # Set bottom position for calibration
            if self.calibration_mode and self.active_joint:
                self.calibration_stage = 'bottom'
                self.get_logger().info(f"Move hand to desired bottom position for {self.active_joint}")
    
    def handle_calibration(self, landmarks):
        """Handle calibration of joint positions with hand position"""
        if not self.active_joint or not landmarks or not self.calibration_stage:
            return
            
        # Get current joint position
        joint_idx = self.joint_names.index(self.active_joint)
        current_pos = self.current_joint_positions[joint_idx]
        
        # Store the calibration position
        self.calibration_positions[self.calibration_stage][self.active_joint] = current_pos
        
        # Log the calibration
        self.get_logger().info(f"Calibrated {self.calibration_stage} position for {self.active_joint}: {current_pos:.3f}")
        
        # Reset calibration stage to indicate completion
        self.calibration_stage = None
    
    def map_hand_to_joint_angle(self, landmarks):
        """Map hand position directly to joint angle"""
        if not self.active_joint or not landmarks:
            return
            
        # Use the y-coordinate of the index finger tip (landmark 8) for control
        y_pos = landmarks.landmark[8].y
        
        # Clamp y position to control range
        y_pos = max(self.hand_position_min, min(self.hand_position_max, y_pos))
        
        # Normalize the y position to 0-1 range within our control range
        normalized_y = (y_pos - self.hand_position_min) / (self.hand_position_max - self.hand_position_min)
        
        # Invert so that hand up = max angle, hand down = min angle
        normalized_y = 1.0 - normalized_y
        
        # Get joint index and limits
        joint_idx = self.joint_names.index(self.active_joint)
        joint_min = self.joint_limits[self.active_joint][0]
        joint_max = self.joint_limits[self.active_joint][1]
        
        # Check if we have calibration positions for this joint
        if (self.calibration_positions['top'][self.active_joint] is not None and 
            self.calibration_positions['bottom'][self.active_joint] is not None):
            # Use calibrated range instead of joint limits
            top_pos = self.calibration_positions['top'][self.active_joint]
            bottom_pos = self.calibration_positions['bottom'][self.active_joint]
            
            # Interpolate between calibrated positions
            target_pos = bottom_pos + normalized_y * (top_pos - bottom_pos)
            
            # Ensure we don't exceed joint limits regardless of calibration
            target_pos = max(joint_min, min(joint_max, target_pos))
        else:
            # Map directly to joint limits
            target_pos = joint_min + normalized_y * (joint_max - joint_min)
        
        # Apply smoothing filter
        alpha = 0.3  # Smoothing factor (higher = less smoothing)
        smooth_pos = self.current_joint_positions[joint_idx] * (1 - alpha) + target_pos * alpha
        
        # Update the position
        self.current_joint_positions[joint_idx] = smooth_pos
    
    def create_default_trajectory(self):
        """Create a default trajectory at the neutral position"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        
        # Use middle of joint ranges as neutral position
        neutral_positions = [(self.joint_limits[joint][0] + self.joint_limits[joint][1]) / 2.0 
                           for joint in self.joint_names]
        
        # Initialize previous and current positions
        self.prev_joint_positions = list(neutral_positions)
        self.current_joint_positions = list(neutral_positions)
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = neutral_positions
        point.velocities = [0.0] * len(self.joint_names)
        
        # Set time from start
        point.time_from_start = Duration(sec=0, nanosec=500000000)  # 0.5 seconds
        
        # Initialize points list and add the point
        trajectory.points = [point]
        return trajectory
    
    def control_loop(self):
        """Main control loop that runs at controller update rate"""
        if not self.publishing_enabled:
            return  # Skip if publishing is disabled
            
        dt = 1.0 / self.control_rate  # Time step
        
        # Apply velocity limiting to current joint positions
        limited_angles = self.limit_velocity(self.current_joint_positions, dt)
        
        # Create trajectory message
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = limited_angles
        point.velocities = [0.0] * len(self.joint_names)
        
        # Set time from start
        point.time_from_start = Duration(sec=0, nanosec=500000000)  # 0.5 seconds
        
        # Set points list
        trajectory.points = [point]
        
        # Publish trajectory
        self.trajectory_pub.publish(trajectory)
        
        # Save last valid trajectory
        self.last_valid_trajectory = trajectory
    
    def limit_velocity(self, target_positions, dt):
        """Limit the velocity of joint movements"""
        limited_positions = []
        
        for i in range(len(self.prev_joint_positions)):
            curr = self.prev_joint_positions[i]
            target = target_positions[i]
            
            # Calculate desired change
            delta = target - curr
            
            # Calculate current velocity
            velocity = delta / dt
            
            # Limit velocity
            if abs(velocity) > self.max_velocity:
                # Scale back the movement
                velocity = np.sign(velocity) * self.max_velocity
                delta = velocity * dt
            
            # Apply the limited movement
            limited_positions.append(curr + delta)
        
        # Update previous positions for next iteration
        self.prev_joint_positions = list(limited_positions)
        
        return limited_positions

def main(args=None):
    rclpy.init(args=args)
    node = HandTrackingArmController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        if hasattr(node, 'cap'):
            node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()