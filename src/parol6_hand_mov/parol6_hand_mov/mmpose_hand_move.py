#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from threading import Lock
from builtin_interfaces.msg import Duration

# Import MMPose
try:
    from mmpose.apis import init_model, inference_top_down_pose_model, vis_pose_result
    from mmpose.structures import merge_data_samples
    from mmdet.apis import init_detector, inference_detector
except ImportError as e:
    print(f"Error: MMPose/MMDetection not found. {e}")
    print("Please install MMPose with: pip install -U openmim && mim install mmengine mmcv mmpose mmdet")
    import sys
    sys.exit(-1)

class MMPoseArmController(Node):
    def __init__(self):
        super().__init__('mmpose_arm_controller')
        
        # Robot configuration
        self.joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        self.joint_limits = {
            'J1': [-1.083 + 0.0698132, 2.148 - 0.0698132],  # 3 degree offset in limits for safety
            'J2': [-0.960 + 0.0698132, 0.855 - 0.0698132],
            'J3': [-1.259 + 0.0698132, 0.995 - 0.0698132],
            'J4': [-1.841 + 0.0698132, 1.841 - 0.0698132],
            'J5': [-1.571 + 0.0698132, 1.571 - 0.0698132],
            'J6': [-3.142 + 0.0698132, 3.142 - 0.0698132]
        }
        
        # Publisher for joint trajectory command
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10)
        
        # Camera subscriber and bridge
        self.cv_bridge = CvBridge()
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10)
        
        # Initialize MMPose models
        self.init_mmpose()
        
        # Control parameters
        self.control_rate = 20  # Hz
        self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        self.lock = Lock()
        
        # State variables
        self.keypoints = None
        self.person_detected = False
        self.last_valid_trajectory = self.create_default_trajectory()
        
        # Keypoint indices for HRNet model with COCO keypoint definition
        # COCO format: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, 
        #              right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
        #              left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        
        # Tracking config
        self.track_right_arm = True  # Set to False to track left arm instead
        
        # Smoothing filter parameters
        self.filter_alpha = 0.2
        self.filtered_joints = [0.0] * 6
        
        # Previous joint positions for velocity limiting
        self.prev_joint_positions = [0.0] * 6
        self.current_joint_positions = [0.0] * 6
        
        # Maximum velocity and acceleration limits
        self.max_velocity = 0.5  # Maximum joint velocity in radians/second
        self.max_accel = 0.2     # Maximum joint acceleration in radians/second^2
        
        # Time factor to slow down movements
        self.time_factor = 5.0   # Higher value = slower movements
        
        # Control flags
        self.publishing_enabled = False
        self.use_direct_camera = True
        self.camera_device_id = 0
        
        # Active control mode
        self.control_mode = "ALL"  # "ALL", "SHOULDER", "ELBOW", "WRIST", "BASE"
        
        if self.use_direct_camera:
            self.setup_direct_camera()
            
        self.get_logger().info('MMPose arm controller initialized')
    
    def init_mmpose(self):
        """Initialize MMPose and MMDetection models"""
        try:
            # Define model paths - update these to your actual paths
            det_config = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
            det_checkpoint = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
            pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
            pose_checkpoint = 'checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
            
            # Initialize detector model
            self.get_logger().info("Initializing MMDetection model...")
            self.detector = init_detector(det_config, det_checkpoint, device='cuda:0')
            
            # Initialize pose model
            self.get_logger().info("Initializing MMPose model...")
            self.pose_model = init_model(pose_config, pose_checkpoint, device='cuda:0')
            
            self.get_logger().info("MMPose and MMDetection models initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize MMPose/MMDetection models: {str(e)}")
            raise
    
    def setup_direct_camera(self):
        """Set up direct camera capture"""
        self.cap = cv2.VideoCapture(self.camera_device_id)
        
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
        """Process the camera image with MMPose"""
        try:
            # Detect persons using MMDetection
            det_results = inference_detector(self.detector, image)
            
            # Get person detections
            pred_instances = det_results.pred_instances
            person_results = []
            
            for idx in range(len(pred_instances)):
                if pred_instances.labels[idx] == 0:  # 0 is the COCO class ID for person
                    # Get bounding box in xyxy format
                    bbox = pred_instances.bboxes[idx].cpu().numpy().tolist()
                    person_results.append({'bbox': bbox})
            
            # Reset detection status
            self.person_detected = False
            
            if len(person_results) > 0:
                # We detected at least one person
                self.person_detected = True
                
                # Perform pose estimation on detected people
                pose_results = inference_top_down_pose_model(
                    self.pose_model,
                    image,
                    person_results,
                    bbox_thr=0.3,
                    format='xyxy',
                    dataset='coco'
                )
                
                # Visualize pose result
                vis_result = vis_pose_result(
                    self.pose_model,
                    image,
                    pose_results,
                    radius=4,
                    thickness=2,
                    show=False
                )
                
                # Extract keypoints from the first detected person
                if pose_results and len(pose_results) > 0:
                    # Get keypoints from the first person
                    keypoints = pose_results[0].pred_instances.keypoints[0].cpu().numpy()
                    
                    with self.lock:
                        self.keypoints = keypoints
                        
                        # Process keypoints for robot control if publishing is enabled
                        if self.publishing_enabled:
                            self.process_arm_control()
                
                # Get the output image for visualization
                output_image = vis_result
            else:
                output_image = image.copy()
            
            # Draw additional information on the image
            self.draw_status_info(output_image)
            
            # Display the image
            cv2.imshow("Arm Controller", output_image)
            key = cv2.waitKey(1) & 0xFF
            
            # Process keyboard input
            self.handle_keyboard_input(key)
            
        except Exception as e:
            self.get_logger().error(f"Error in process_image: {str(e)}")
    
    def process_arm_control(self):
        """Process arm keypoints to control robot joints"""
        if self.keypoints is None:
            return
        
        # Determine which arm to track
        if self.track_right_arm:
            shoulder = self.keypoints[self.RIGHT_SHOULDER]
            elbow = self.keypoints[self.RIGHT_ELBOW]
            wrist = self.keypoints[self.RIGHT_WRIST]
            nose = self.keypoints[self.NOSE]
        else:
            shoulder = self.keypoints[self.LEFT_SHOULDER]
            elbow = self.keypoints[self.LEFT_ELBOW]
            wrist = self.keypoints[self.LEFT_WRIST]
            nose = self.keypoints[self.NOSE]
        
        # Check if essential keypoints are detected (confidence > 0)
        if (shoulder[2] <= 0 or elbow[2] <= 0 or wrist[2] <= 0):
            self.get_logger().warn("Some keypoints are not detected with sufficient confidence")
            return
        
        # Calculate joint angles
        
        # J1 (Base) - Control with horizontal position of shoulder
        # Map x-coordinate (0-image_width) to joint range
        img_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        normalized_shoulder_x = shoulder[0] / img_width
        j1_min = self.joint_limits['J1'][0]
        j1_max = self.joint_limits['J1'][1]
        j1_target = j1_min + normalized_shoulder_x * (j1_max - j1_min)
        
        # J2 (Shoulder) - Calculate angle between nose-shoulder and shoulder-elbow
        nose_to_shoulder = np.array([shoulder[0] - nose[0], shoulder[1] - nose[1]])
        shoulder_to_elbow = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
        
        # Normalize vectors
        if np.linalg.norm(nose_to_shoulder) > 0 and np.linalg.norm(shoulder_to_elbow) > 0:
            nose_to_shoulder = nose_to_shoulder / np.linalg.norm(nose_to_shoulder)
            shoulder_to_elbow = shoulder_to_elbow / np.linalg.norm(shoulder_to_elbow)
            
            # Calculate angle
            shoulder_angle = np.arccos(np.clip(np.dot(nose_to_shoulder, shoulder_to_elbow), -1.0, 1.0))
            
            # Map angle to robot's J2 range
            # Typical human shoulder range is around 0-180 degrees
            # Map this to the robot's J2 range
            normalized_shoulder_angle = shoulder_angle / np.pi
            j2_min = self.joint_limits['J2'][0]
            j2_max = self.joint_limits['J2'][1]
            j2_target = j2_min + normalized_shoulder_angle * (j2_max - j2_min)
        else:
            j2_target = self.current_joint_positions[1]  # Keep current position
        
        # J3 (Elbow) - Calculate angle between shoulder-elbow and elbow-wrist
        shoulder_to_elbow = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
        elbow_to_wrist = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
        
        # Normalize vectors
        if np.linalg.norm(shoulder_to_elbow) > 0 and np.linalg.norm(elbow_to_wrist) > 0:
            shoulder_to_elbow = shoulder_to_elbow / np.linalg.norm(shoulder_to_elbow)
            elbow_to_wrist = elbow_to_wrist / np.linalg.norm(elbow_to_wrist)
            
            # Calculate angle
            elbow_angle = np.arccos(np.clip(np.dot(shoulder_to_elbow, elbow_to_wrist), -1.0, 1.0))
            
            # Map angle to robot's J3 range
            # Typical human elbow range is around 0-145 degrees
            normalized_elbow_angle = elbow_angle / (np.pi * 0.8)  # 145 degrees = 0.8 * pi
            j3_min = self.joint_limits['J3'][0]
            j3_max = self.joint_limits['J3'][1]
            j3_target = j3_min + normalized_elbow_angle * (j3_max - j3_min)
        else:
            j3_target = self.current_joint_positions[2]  # Keep current position
        
        # J4 (Wrist Roll) - For simplicity, use horizontal position of wrist
        normalized_wrist_x = wrist[0] / img_width
        j4_min = self.joint_limits['J4'][0]
        j4_max = self.joint_limits['J4'][1]
        j4_target = j4_min + normalized_wrist_x * (j4_max - j4_min)
        
        # J5 (Wrist Pitch) - Use vertical position of wrist
        img_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        normalized_wrist_y = wrist[1] / img_height
        j5_min = self.joint_limits['J5'][0]
        j5_max = self.joint_limits['J5'][1]
        j5_target = j5_min + normalized_wrist_y * (j5_max - j5_min)
        
        # Apply smoothing and update joint positions based on control mode
        alpha = self.filter_alpha
        
        if self.control_mode == "ALL":
            self.current_joint_positions[0] = self.current_joint_positions[0] * (1 - alpha) + j1_target * alpha
            self.current_joint_positions[1] = self.current_joint_positions[1] * (1 - alpha) + j2_target * alpha
            self.current_joint_positions[2] = self.current_joint_positions[2] * (1 - alpha) + j3_target * alpha
            self.current_joint_positions[3] = self.current_joint_positions[3] * (1 - alpha) + j4_target * alpha
            self.current_joint_positions[4] = self.current_joint_positions[4] * (1 - alpha) + j5_target * alpha
        elif self.control_mode == "BASE":
            self.current_joint_positions[0] = self.current_joint_positions[0] * (1 - alpha) + j1_target * alpha
        elif self.control_mode == "SHOULDER":
            self.current_joint_positions[1] = self.current_joint_positions[1] * (1 - alpha) + j2_target * alpha
        elif self.control_mode == "ELBOW":
            self.current_joint_positions[2] = self.current_joint_positions[2] * (1 - alpha) + j3_target * alpha
        elif self.control_mode == "WRIST":
            self.current_joint_positions[3] = self.current_joint_positions[3] * (1 - alpha) + j4_target * alpha
            self.current_joint_positions[4] = self.current_joint_positions[4] * (1 - alpha) + j5_target * alpha
    
    def draw_status_info(self, image):
        """Draw status information on the image"""
        h, w = image.shape[:2]
        
        # Draw status text
        status_color = (0, 255, 0) if self.publishing_enabled else (0, 0, 255)
        cv2.putText(image, f"Publishing: {'ON' if self.publishing_enabled else 'OFF'}", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Show control mode
        mode_text = f"Control Mode: {self.control_mode}"
        cv2.putText(image, mode_text, (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show tracking status
        tracking_arm = "Right Arm" if self.track_right_arm else "Left Arm"
        cv2.putText(image, f"Tracking: {tracking_arm}", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show person detection status
        detection_status = "Person detected" if self.person_detected else "No person detected"
        cv2.putText(image, detection_status, (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 255, 0) if self.person_detected else (0, 0, 255), 2)
        
        # Draw current joint positions
        for i, joint in enumerate(self.joint_names):
            value = self.current_joint_positions[i]
            pct = (value - self.joint_limits[joint][0]) / (self.joint_limits[joint][1] - self.joint_limits[joint][0])
            pct = max(0, min(1, pct))  # Clamp between 0 and 1
            
            # Set color based on control mode
            color = (0, 255, 255)
            if (self.control_mode == "ALL" or 
                (self.control_mode == "BASE" and i == 0) or
                (self.control_mode == "SHOULDER" and i == 1) or
                (self.control_mode == "ELBOW" and i == 2) or
                (self.control_mode == "WRIST" and (i == 3 or i == 4))):
                color = (0, 255, 255)  # Highlight active joints
            else:
                color = (200, 200, 200)  # Inactive joints
            
            # Draw joint name and value
            cv2.putText(image, f"{joint}: {value:.2f}", (w - 180, 30 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw control instructions
        instructions = [
            "SPACE: Start/Stop Publishing",
            "A: Control ALL Joints",
            "B: Control Base (J1)",
            "S: Control Shoulder (J2)",
            "E: Control Elbow (J3)",
            "W: Control Wrist (J4-J5)",
            "T: Toggle Arm (Left/Right)",
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
        elif key == ord('a') or key == ord('A'):
            self.control_mode = "ALL"
            self.get_logger().info("Controlling ALL joints")
        elif key == ord('b') or key == ord('B'):
            self.control_mode = "BASE"
            self.get_logger().info("Controlling Base (J1)")
        elif key == ord('s') or key == ord('S'):
            self.control_mode = "SHOULDER"
            self.get_logger().info("Controlling Shoulder (J2)")
        elif key == ord('e') or key == ord('E'):
            self.control_mode = "ELBOW"
            self.get_logger().info("Controlling Elbow (J3)")
        elif key == ord('w') or key == ord('W'):
            self.control_mode = "WRIST"
            self.get_logger().info("Controlling Wrist (J4-J5)")
        elif key == ord('t') or key == ord('T'):
            self.track_right_arm = not self.track_right_arm
            arm = "right" if self.track_right_arm else "left"
            self.get_logger().info(f"Tracking {arm} arm")
    
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
    node = MMPoseArmController()
    
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