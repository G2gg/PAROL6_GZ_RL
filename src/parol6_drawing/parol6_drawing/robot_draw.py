#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import xml.etree.ElementTree as ET
import re
import math
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState, PlanningSceneComponents
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import os
from ament_index_python.packages import get_package_share_directory
import time


class SVGPathParser:
    """Parse SVG path data and convert to 3D coordinates."""
    
    def __init__(self):
        # Regular expression to match path commands and coordinates
        self.path_regex = re.compile(r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)')
        self.coords_regex = re.compile(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?')

    def _approximate_elliptical_arc(self, start, rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, end, num_segments=3):
        """Approximate an elliptical arc with line segments."""
        # Convert angles from degrees to radians
        x_axis_rotation = math.radians(x_axis_rotation)
        
        # If the radii are zero, treat as a straight line
        if rx == 0 or ry == 0:
            return [end]
        
        # Ensure radii are positive
        rx, ry = abs(rx), abs(ry)
        
        # Step 1: Transform to origin
        dx = (start[0] - end[0]) / 2
        dy = (start[1] - end[1]) / 2
        
        # Rotate to align with ellipse axes
        cos_phi = math.cos(x_axis_rotation)
        sin_phi = math.sin(x_axis_rotation)
        x1 = cos_phi * dx + sin_phi * dy
        y1 = -sin_phi * dx + cos_phi * dy
        
        # To nsure radii are large enough
        lambda_value = (x1 ** 2) / (rx ** 2) + (y1 ** 2) / (ry ** 2)
        if lambda_value > 1:
            rx *= math.sqrt(lambda_value)
            ry *= math.sqrt(lambda_value)
        
        # Step 2: Compute center parameters
        sign = -1 if large_arc_flag != sweep_flag else 1
        sq = ((rx ** 2) * (ry ** 2) - (rx ** 2) * (y1 ** 2) - (ry ** 2) * (x1 ** 2)) / ((rx ** 2) * (y1 ** 2) + (ry ** 2) * (x1 ** 2))
        sq = max(0, sq)  # Ensure non-negative
        coef = sign * math.sqrt(sq)
        cx1 = coef * ((rx * y1) / ry)
        cy1 = coef * -((ry * x1) / rx)
        
        # Transform center back
        cx = cos_phi * cx1 - sin_phi * cy1 + (start[0] + end[0]) / 2
        cy = sin_phi * cx1 + cos_phi * cy1 + (start[1] + end[1]) / 2
        
        # Step 3: Compute start and sweep angles
        ux = (x1 - cx1) / rx
        uy = (y1 - cy1) / ry
        vx = (-x1 - cx1) / rx
        vy = (-y1 - cy1) / ry
        
        # Start angle
        start_angle = math.atan2(uy, ux)
        
        # Compute angle delta
        n = math.sqrt((ux ** 2 + uy ** 2) * (vx ** 2 + vy ** 2))
        p = ux * vx + uy * vy
        d = p / n
        d = max(-1, min(1, d))  # Clamp to avoid precision errors
        delta_angle = math.acos(d)
        
        if (ux * vy - uy * vx) < 0:
            delta_angle = -delta_angle
        
        if sweep_flag == 0 and delta_angle > 0:
            delta_angle -= 2 * math.pi
        elif sweep_flag == 1 and delta_angle < 0:
            delta_angle += 2 * math.pi
        
        # Generate points along the arc
        points = []
        for i in range(1, num_segments + 1):
            theta = start_angle + delta_angle * i / num_segments
            x = cx + rx * math.cos(theta) * cos_phi - ry * math.sin(theta) * sin_phi
            y = cy + rx * math.cos(theta) * sin_phi + ry * math.sin(theta) * cos_phi
            points.append([x, y])
        
        return points

    def _approximate_bezier_curve(self, start, cp1, cp2, end, num_segments=3):
        """Approximate a cubic Bezier curve with line segments."""
        points = []
        for i in range(1, num_segments + 1):
            t = i / num_segments
            # Cubic Bezier formula
            x = (1-t)**3 * start[0] + 3*(1-t)**2*t * cp1[0] + 3*(1-t)*t**2 * cp2[0] + t**3 * end[0]
            y = (1-t)**3 * start[1] + 3*(1-t)**2*t * cp1[1] + 3*(1-t)*t**2 * cp2[1] + t**3 * end[1]
            points.append([x, y])
        return points

    def parse_path(self, path_data):
        """Parse SVG path data into a list of commands and coordinates."""
        path_commands = []
        current_pos = [0, 0]  # Current cursor position
        
        for command, params in self.path_regex.findall(path_data):
            coords = [float(c) for c in self.coords_regex.findall(params)]
            
            if command == 'M':  # Move to (absolute)
                for i in range(0, len(coords), 2):
                    current_pos = [coords[i], coords[i+1]]
                    path_commands.append(('move', current_pos[:]))
                    
            elif command == 'm':  # Move to (relative)
                for i in range(0, len(coords), 2):
                    current_pos[0] += coords[i]
                    current_pos[1] += coords[i+1]
                    path_commands.append(('move', current_pos[:]))
                    
            elif command == 'L':  # Line to (absolute)
                for i in range(0, len(coords), 2):
                    current_pos = [coords[i], coords[i+1]]
                    path_commands.append(('line', current_pos[:]))
                    
            elif command == 'l':  # Line to (relative)
                for i in range(0, len(coords), 2):
                    current_pos[0] += coords[i]
                    current_pos[1] += coords[i+1]
                    path_commands.append(('line', current_pos[:]))
                    
            elif command == 'H':  # Horizontal line (absolute)
                for coord in coords:
                    current_pos[0] = coord
                    path_commands.append(('line', current_pos[:]))
                    
            elif command == 'h':  # Horizontal line (relative)
                for coord in coords:
                    current_pos[0] += coord
                    path_commands.append(('line', current_pos[:]))
                    
            elif command == 'V':  # Vertical line (absolute)
                for coord in coords:
                    current_pos[1] = coord
                    path_commands.append(('line', current_pos[:]))
                    
            elif command == 'v':  # Vertical line (relative)
                for coord in coords:
                    current_pos[1] += coord
                    path_commands.append(('line', current_pos[:]))
                    
            elif command in ('C', 'c'):  # Cubic Bezier curve
                absolute = (command == 'C')
                for i in range(0, len(coords), 6):
                    cp1 = [coords[i], coords[i+1]]
                    cp2 = [coords[i+2], coords[i+3]]
                    end = [coords[i+4], coords[i+5]]
                    
                    if not absolute:
                        cp1[0] += current_pos[0]
                        cp1[1] += current_pos[1]
                        cp2[0] += current_pos[0]
                        cp2[1] += current_pos[1]
                        end[0] += current_pos[0]
                        end[1] += current_pos[1]
                    
                    # Approximate the curve with multiple line segments
                    curve_points = self._approximate_bezier_curve(current_pos, cp1, cp2, end)
                    for point in curve_points:
                        path_commands.append(('line', point))
                    
                    current_pos = end[:]
                    
            elif command == 'Z' or command == 'z':  # Close path
                path_commands.append(('close', None))
            
            elif command in ('S', 's'):  # Smooth cubic Bezier curve
                absolute = (command == 'S')
                for i in range(0, len(coords), 4):
                    # Reflection of the previous control point
                    if path_commands and path_commands[-1][0] == 'line' and hasattr(self, '_last_control_point'):
                        cp1 = [2 * current_pos[0] - self._last_control_point[0],
                            2 * current_pos[1] - self._last_control_point[1]]
                    else:
                        cp1 = current_pos[:]
                        
                    cp2 = [coords[i], coords[i+1]]
                    end = [coords[i+2], coords[i+3]]
                    
                    if not absolute:
                        cp2[0] += current_pos[0]
                        cp2[1] += current_pos[1]
                        end[0] += current_pos[0]
                        end[1] += current_pos[1]
                    
                    # Store last control point for potential next S/s command
                    self._last_control_point = cp2[:]
                    
                    # Approximate the curve with multiple line segments
                    curve_points = self._approximate_bezier_curve(current_pos, cp1, cp2, end)
                    for point in curve_points:
                        path_commands.append(('line', point))
                    
                    current_pos = end[:]

            elif command in ('Q', 'q'):  # Quadratic Bezier curve
                absolute = (command == 'Q')
                for i in range(0, len(coords), 4):
                    cp = [coords[i], coords[i+1]]
                    end = [coords[i+2], coords[i+3]]
                    
                    if not absolute:
                        cp[0] += current_pos[0]
                        cp[1] += current_pos[1]
                        end[0] += current_pos[0]
                        end[1] += current_pos[1]
                    
                    # Store control point for potential next T/t command
                    self._last_quadratic_control = cp[:]
                    
                    # Convert quadratic Bezier to cubic for consistent handling
                    cp1 = [current_pos[0] + 2/3 * (cp[0] - current_pos[0]),
                        current_pos[1] + 2/3 * (cp[1] - current_pos[1])]
                    cp2 = [end[0] + 2/3 * (cp[0] - end[0]),
                        end[1] + 2/3 * (cp[1] - end[1])]
                    
                    # Approximate the curve with multiple line segments
                    curve_points = self._approximate_bezier_curve(current_pos, cp1, cp2, end)
                    for point in curve_points:
                        path_commands.append(('line', point))
                    
                    current_pos = end[:]

            elif command in ('T', 't'):  # Smooth quadratic Bezier curve
                absolute = (command == 'T')
                for i in range(0, len(coords), 2):
                    # Reflection of the previous control point
                    if hasattr(self, '_last_quadratic_control'):
                        cp = [2 * current_pos[0] - self._last_quadratic_control[0],
                            2 * current_pos[1] - self._last_quadratic_control[1]]
                    else:
                        cp = current_pos[:]
                        
                    end = [coords[i], coords[i+1]]
                    
                    if not absolute:
                        end[0] += current_pos[0]
                        end[1] += current_pos[1]
                    
                    # Store control point for potential next T/t command
                    self._last_quadratic_control = cp[:]
                    
                    # Convert quadratic Bezier to cubic for consistent handling
                    cp1 = [current_pos[0] + 2/3 * (cp[0] - current_pos[0]),
                        current_pos[1] + 2/3 * (cp[1] - current_pos[1])]
                    cp2 = [end[0] + 2/3 * (cp[0] - end[0]),
                        end[1] + 2/3 * (cp[1] - end[1])]
                    
                    # Approximate the curve with multiple line segments
                    curve_points = self._approximate_bezier_curve(current_pos, cp1, cp2, end)
                    for point in curve_points:
                        path_commands.append(('line', point))
                    
                    current_pos = end[:]

            elif command in ('A', 'a'):  # Elliptical arc
                absolute = (command == 'A')
                for i in range(0, len(coords), 7):
                    rx, ry = coords[i], coords[i+1]
                    x_axis_rotation = coords[i+2]
                    large_arc_flag = int(coords[i+3])
                    sweep_flag = int(coords[i+4])
                    end = [coords[i+5], coords[i+6]]
                    
                    if not absolute:
                        end[0] += current_pos[0]
                        end[1] += current_pos[1]
                    
                    # Approximate the arc with multiple line segments
                    arc_points = self._approximate_elliptical_arc(current_pos, rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, end)
                    
                    for point in arc_points:
                        path_commands.append(('line', point))
                    
                    current_pos = end[:]
                
        return path_commands
    
    def parse_circle(self, cx, cy, r):
        """Convert an SVG circle to a sequence of path commands."""
        path_commands = []
        steps = 24  # Number of segments to approximate the circle
        for i in range(steps + 1):
            angle = 2 * math.pi * i / steps
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            if i == 0:
                path_commands.append(('move', [x, y]))
            else:
                path_commands.append(('line', [x, y]))
        return path_commands


class SVGRobotDrawer(Node):
    """ROS2 node for converting SVG files to robot arm movement commands."""
    
    def __init__(self):
        super().__init__('svg_robot_drawer')
        
        # Publisher for joint trajectory command
        self.pose_publisher = self.create_publisher(Pose, 'robot_target_pose', 10)
        self.trajectory_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)

        # For preventing deadlocks
        self.callback_group = ReentrantCallbackGroup()

        # Set up MoveIt service clients
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik', callback_group=self.callback_group)

        # Wait for service to be available
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for IK service...')

        
        # self.last_position = None

        # Robot specific dimensions in meters
        self.a1 = 0.11050  # 110.50 mm
        self.a2 = 0.02342  # 23.42 mm
        self.a3 = 0.18000  # 180.00 mm
        self.a4 = 0.04350  # 43.50 mm
        self.a5 = 0.17635  # 176.35 mm
        self.a6 = 0.06280  # 62.8 mm
        self.a7 = 0.04525  # 45.25 mm

        self.joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        self.joint_limits = {
            'J1': [-1.083 + 0.0698132, 2.148 - 0.0698132],  # 3 degree offset in limits for safety ~ 0.0698132 radians
            'J2': [-1.221 + 0.0698132, 0.907 - 0.0698132],
            'J3': [-1.883 + 0.0698132, 1.256 - 0.0698132],
            'J4': [-1.841 + 0.0698132, 1.841 - 0.0698132],
            'J5': [-1.571 + 0.0698132, 1.571 - 0.0698132],
            'J6': [-3.142, 3.142]
        }

        self.distance = 0.8 # initialize distance between starting point and next point
        # Parameters
        self.declare_parameter('svg_file', '')
        self.declare_parameter('z_draw_height', 0.001)  
        self.declare_parameter('z_travel_height', 0.025)  
        self.declare_parameter('scale_factor', 0.0002)  
        self.declare_parameter('x_offset', 0.15)       
        self.declare_parameter('y_offset', 0.15)        
        self.declare_parameter('height_of_paper_center', 0.15)          
        self.declare_parameter('command_reduction_factor', 4.0)  
        
        # Get parameters
        self.svg_file = self.get_parameter('svg_file').value
        self.z_draw_height = self.get_parameter('z_draw_height').value
        self.z_travel_height = self.get_parameter('z_travel_height').value
        self.scale_factor = self.get_parameter('scale_factor').value
        self.x_offset = self.get_parameter('x_offset').value
        self.y_offset = self.get_parameter('y_offset').value
        self.height_of_paper_center = self.get_parameter('height_of_paper_center').value
        self.command_reduction_factor = self.get_parameter('command_reduction_factor').value

        
        
        self.timer = self.create_timer(1.0, self.process_svg)
        
        # Initialize SVG parser
        self.path_parser = SVGPathParser()
        
        self.get_logger().info('SVG Robot Drawer node initialized')
    

    def compute_ik(self, target_pose):
        """Call MoveIt IK service to compute inverse kinematics."""
        request = GetPositionIK.Request()
        
        # Set up the IK request
        ik_request = PositionIKRequest()
        ik_request.group_name = "arm"  
        ik_request.pose_stamped.header.frame_id = "base_link"  
        ik_request.pose_stamped.pose = target_pose
        
        robot_state = RobotState()
        joint_state = JointState()
        joint_state.name = self.joint_names
        joint_state.position = [0.0] * len(self.joint_names)  # Default position or current position
        robot_state.joint_state = joint_state
        ik_request.robot_state = robot_state
        
        ik_request.avoid_collisions = True
        
        request.ik_request = ik_request
        
        future = self.ik_client.call_async(request)
        
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0) #wait for the result
        
        # Process the result
        if future.result() is not None:
            response = future.result()
            if response.error_code.val == 1:  # SUCCESS
                # Extract joint positions from the solution
                joint_positions = []
                for joint_name in self.joint_names:
                    if joint_name in response.solution.joint_state.name:
                        index = response.solution.joint_state.name.index(joint_name)
                        joint_positions.append(response.solution.joint_state.position[index])
                    else:
                        self.get_logger().warn(f"Joint {joint_name} not found in IK solution")
                        return None
                return joint_positions
            else:
                self.get_logger().warn(f"IK failed with error code: {response.error_code.val}")
                return None
        else:
            self.get_logger().error("IK service call failed")
            return None
    

    def process_svg(self):
        """Process the SVG file and start drawing."""
        self.timer.cancel()  # Cancel the timer to avoid repeated processing
        
        if not self.svg_file:
            self.get_logger().error('No SVG file specified. Set the svg_file parameter.')
            return
        
        self.get_logger().info(f'Processing SVG file: {self.svg_file}')
        
        try:
            # Parse the SVG file
            tree = ET.parse(self.svg_file)
            root = tree.getroot()
            
            ns = {'svg': 'http://www.w3.org/2000/svg'} # Namespace
            
            # Extract viewBox information
            viewbox = root.get('viewBox')
            if viewbox:
                vb_values = [float(v) for v in viewbox.split()]
                orig_width, orig_height = vb_values[2], vb_values[3]
            else:
                orig_width = float(root.get('width', '500').replace('px', '').replace('%', ''))
                orig_height = float(root.get('height', '500').replace('px', '').replace('%', ''))

            # Set target width and calculate proportional height
            target_width = 500.0
            scale_factor = target_width / orig_width
            target_height = orig_height * scale_factor

            # Update the SVG attributes
            if viewbox:
                #  scaling width and height
                root.set('viewBox', f"{vb_values[0]} {vb_values[1]} {target_width} {target_height}")
            else:
                root.set('viewBox', f"0 0 {target_width} {target_height}")

            # Update width and height attributes
            root.set('width', f"{target_width}px")
            root.set('height', f"{target_height}px")

            
            self.get_logger().info(f'SVG dimensions: {target_width} x {target_height}')
            
           # Process all paths in the SVG
            all_commands = []
            
            # Process path elements
            for path in root.findall('.//svg:path', ns):
                path_data = path.get('d')
                if path_data:
                    commands = self.path_parser.parse_path(path_data)
                    all_commands.extend(commands)
            
            # Process circle elements
            for circle in root.findall('.//svg:circle', ns):
                cx = float(circle.get('cx', '0'))
                cy = float(circle.get('cy', '0'))
                r = float(circle.get('r', '0'))
                commands = self.path_parser.parse_circle(cx, cy, r)
                all_commands.extend(commands)

            # Handle <rect> elements
            for rect in root.findall('.//svg:rect', ns):
                x = float(rect.get('x', '0'))
                y = float(rect.get('y', '0'))
                w = float(rect.get('width', '0'))
                h = float(rect.get('height', '0'))

                commands = [
                    ('move', [x, y]),
                    ('line', [x + w, y]),
                    ('line', [x + w, y + h]),
                    ('line', [x, y + h]),
                    ('line', [x, y]),
                    ('close', None)
                ]
                all_commands.extend(commands)

            # Handle <line> elements
            for line in root.findall('.//svg:line', ns):
                x1 = float(line.get('x1', '0'))
                y1 = float(line.get('y1', '0'))
                x2 = float(line.get('x2', '0'))
                y2 = float(line.get('y2', '0'))

                commands = [
                    ('move', [x1, y1]),
                    ('line', [x2, y2])
                ]
                all_commands.extend(commands)

            # Handle <polyline> elements
            for polyline in root.findall('.//svg:polyline', ns):
                points = polyline.get('points', '').strip()
                coords = [float(num) for num in re.findall(r'[-+]?[0-9]*\.?[0-9]+', points)]
                if len(coords) >= 2:
                    commands = [('move', [coords[0], coords[1]])]
                    for i in range(2, len(coords), 2):
                        commands.append(('line', [coords[i], coords[i+1]]))
                    all_commands.extend(commands)

            # Handle <polygon> elements
            for polygon in root.findall('.//svg:polygon', ns):
                points = polygon.get('points', '').strip()
                coords = [float(num) for num in re.findall(r'[-+]?[0-9]*\.?[0-9]+', points)]
                if len(coords) >= 2:
                    commands = [('move', [coords[0], coords[1]])]
                    for i in range(2, len(coords), 2):
                        commands.append(('line', [coords[i], coords[i+1]]))
                    commands.append(('line', [coords[0], coords[1]]))  # close shape
                    commands.append(('close', None))
                    all_commands.extend(commands)
            
            # Handle <ellipse> elements
            for ellipse in root.findall('.//svg:ellipse', ns):
                cx = float(ellipse.get('cx', '0'))
                cy = float(ellipse.get('cy', '0'))
                rx = float(ellipse.get('rx', '0'))
                ry = float(ellipse.get('ry', '0'))
                
                # Approximate the ellipse with a series of Bezier curves
                # A common technique is to use 4 cubic Bezier curves for a complete ellipse
                # Magic number: 0.551784 gives a good approximation of a quarter circle
                k = 0.551784
                
                # Define control points and path commands
                commands = [
                    # Move to rightmost point
                    ('move', [cx + rx, cy]),
                    
                    # Top-right quarter
                    ('curve', [cx + rx, cy - k*ry, cx + k*rx, cy - ry, cx, cy - ry]),  
                    
                    # Top-left quarter
                    ('curve', [cx - k*rx, cy - ry, cx - rx, cy - k*ry, cx - rx, cy]),
                    
                    # Bottom-left quarter
                    ('curve', [cx - rx, cy + k*ry, cx - k*rx, cy + ry, cx, cy + ry]),
                    
                    # Bottom-right quarter
                    ('curve', [cx + k*rx, cy + ry, cx + rx, cy + k*ry, cx + rx, cy]),
                    
                    # Close the ellipse
                    ('close', None)
                ]
                all_commands.extend(commands)

            # Handle <text> elements
            for text_elem in root.findall('.//svg:text', ns):
                x = float(text_elem.get('x', '0'))
                y = float(text_elem.get('y', '0'))
                content = text_elem.text or ''
                
                # For text, we'll just record the position and content
                # Actual rendering would depend on your drawing system
                font_size = float(text_elem.get('font-size', '12').replace('px', '').replace('pt', ''))
                
                # Simplified approach: add a move command to the text position
                # and store text content as metadata
                commands = [
                    ('move', [x, y]),
                    ('text', {
                        'content': content,
                        'font_size': font_size,
                        'x': x,
                        'y': y
                    })
                ]
                all_commands.extend(commands)
                
                # Also handle any tspan elements inside the text
                for tspan in text_elem.findall('.//svg:tspan', ns):
                    tx = float(tspan.get('x', str(x)))
                    ty = float(tspan.get('y', str(y)))
                    tcontent = tspan.text or ''
                    
                    tcommands = [
                        ('move', [tx, ty]),
                        ('text', {
                            'content': tcontent,
                            'font_size': font_size,  # Inherit from parent or override if specified
                            'x': tx,
                            'y': ty
                        })
                    ]
                    all_commands.extend(tcommands)

            # Handle <image> elements
            for image in root.findall('.//svg:image', ns):
                x = float(image.get('x', '0'))
                y = float(image.get('y', '0'))
                width = float(image.get('width', '0').replace('px', ''))
                height = float(image.get('height', '0').replace('px', ''))
                href = image.get('{http://www.w3.org/1999/xlink}href', '')
                
                # Create a rectangular outline for the image
                commands = [
                    ('move', [x, y]),
                    ('line', [x + width, y]),
                    ('line', [x + width, y + height]),
                    ('line', [x, y + height]),
                    ('line', [x, y]),
                    ('close', None),
                    # Store image metadata
                    ('image', {
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'href': href
                    })
                ]
                all_commands.extend(commands)
            # Start drawing
            self.execute_drawing_commands(all_commands, target_width, target_height)
            
        except Exception as e:
            self.get_logger().error(f'Error processing SVG file: {str(e)}')
    
    def execute_drawing_commands(self, commands, width, height):
        """Execute the drawing commands by sending target positions to the robot."""
        self.get_logger().info(f'Starting to execute {len(commands)} drawing commands')

        commands = self.filter_points(commands, min_distance=self.command_reduction_factor)  # Adjust min_distance as needed
    
        # self.get_logger().info(f'Starting to execute {len(commands)} drawing commands (after filtering)')
        
        current_z = self.y_offset + self.z_draw_height
        pen_down = False
        last_successful_position = None
        consecutive_failures = 0
        max_failures = 3
        self.is_first_move = True
        
        for i, (cmd_type, coords) in enumerate(commands):
            try:
                if cmd_type == 'move':
                    # Lift pen for move commands
                    if pen_down:
                        # First: Lift pen at current XY position
                        if hasattr(self, 'last_position'):
                            # Convert last robot position back to SVG coordinates
                            last_svg_x = (self.last_position[0] - self.x_offset) / self.scale_factor
                            last_svg_y = height - (self.last_position[2] - self.height_of_paper_center) / self.scale_factor
                            
                            # Lift pen at current position
                            success = self.move_to_point(last_svg_x, last_svg_y, 
                                                         self.y_offset - self.z_travel_height, width, height, "slow")
                              # Wait for lift to complete
                        
                        current_z = self.y_offset - self.z_travel_height
                        pen_down = False
                    
                        # Then: Move horizontally to new position with pen up
                        success = self.move_to_point(coords[0], coords[1], current_z, width, height, "slow")
                        # time.sleep(movement_time)
                        # success = self.move_to_point(coords[0], coords[1], self.y_offset + self.z_draw_height, width, height)
                        # time.sleep(0.1)
                    else:
                        # Already up, just move
                        success = self.move_to_point(coords[0], coords[1], current_z, width, height, "slow")
                        
                        if success:
                            last_successful_position = coords
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                    self.is_first_move = False  # After first move, set to False
                        
                elif cmd_type == 'line':
                    # Put pen down for line commands if not already down
                    if not pen_down:
                        current_z = self.y_offset + self.z_draw_height
                        if hasattr(self, 'last_position'):
                            # Convert last robot position back to SVG coordinates
                            last_svg_x = (self.last_position[0] - self.x_offset) / self.scale_factor
                            last_svg_y = height - (self.last_position[2] - self.height_of_paper_center) / self.scale_factor
                            
                            # Down pen at current position
                            success = self.move_to_point(last_svg_x, last_svg_y, current_z, width, height, "slow")
                        # time.sleep(0.5)
                        if success:
                            current_z = self.y_offset + self.z_draw_height
                            pen_down = True
                            last_successful_position = coords
                            consecutive_failures = 0
                            success = self.move_to_point(coords[0], coords[1], current_z, width, height, "normal")
                        else:
                            consecutive_failures += 1
                    
                    success = self.move_to_point(coords[0], coords[1], current_z, width, height, "normal")
                    # time.sleep(0.1)
                    if success:
                        last_successful_position = coords
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        
                elif cmd_type == 'close':
                    # Lift pen at the end of a closed path
                    current_z = self.y_offset - self.z_travel_height
                    if hasattr(self, 'last_position'):
                            # Convert last robot position back to SVG coordinates
                            last_svg_x = (self.last_position[0] - self.x_offset) / self.scale_factor
                            last_svg_y = height - (self.last_position[2] - self.height_of_paper_center) / self.scale_factor
                            
                            # Lift pen at current position
                            success = self.move_to_point(last_svg_x, last_svg_y, current_z, width, height, "slow")
                    pen_down = False
                
                # If we've had too many consecutive failures, try to recover
                if consecutive_failures >= max_failures and last_successful_position:
                    self.get_logger().warn(f'Too many consecutive IK failures, trying to recover')
                    # Try to move to the last successful position with pen up
                    current_z = self.y_offset - self.z_travel_height
                    pen_down = False
                    recovery_success = self.move_to_point(last_successful_position[0], last_successful_position[1], 
                                                current_z, width, height, "normal")
                    if recovery_success:
                        consecutive_failures = 0
                        self.get_logger().info(f'Recovery successful')
                    else:
                        self.get_logger().error(f'Recovery failed, aborting drawing')
                        break
                        
                # Add a small delay to maintain a reasonable rate
                time.sleep(0.05)
                
                if i % 50 == 0:
                    self.get_logger().info(f'Executed {i}/{len(commands)} commands')
                    
            except Exception as e:
                self.get_logger().error(f'Error executing command: {str(e)}')
                consecutive_failures += 1
                
        # Ensure pen is up at the end
        if pen_down and last_successful_position:
            current_z = self.y_offset - self.z_travel_height
            self.move_to_point(last_successful_position[0], last_successful_position[1], current_z, width, height, "normal")
        
        self.get_logger().info('Drawing completed')

        self.return_to_home_position()
    
    def move_to_point(self, x, y, z, svg_width, svg_height, movement_time): 
        """Convert SVG coordinates to robot coordinates and send command using MoveIt IK."""
    
        
        # # To these lines (if paper is on YZ plane facing the robot):
        # robot_x = self.x_offset  # Fixed distance from robot to paper
        # robot_y = self.y_offset + x * self.scale_factor  # SVG x maps to robot y
        # robot_z = self.height_of_paper_center + (svg_height - y) * self.scale_factor  # SVG y maps to robot z (flipped)
        
        # Or these lines (if paper is on XZ plane to the side of robot):
        robot_x = self.x_offset + x * self.scale_factor  # SVG x maps to robot x
        robot_y = z  # Fixed distance from robot to paper
        robot_z = self.height_of_paper_center + (svg_height - y) * self.scale_factor  # SVG y maps to robot z (flipped)
        
        # Create and publish pose message
        pose = Pose()
        pose.position.x = robot_x
        pose.position.y = robot_y
        pose.position.z = robot_z
        
        
        # # Use this for paper on YZ plane (facing the robot):
        # pose.orientation.x = 0.0
        # pose.orientation.y = 0.0
        # pose.orientation.z = 0.0
        # pose.orientation.w = 1.0  # Pointing straight forward
        
        # Or this for paper on XZ plane (to the side of robot):
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.707  # 90 degrees around Z
        pose.orientation.w = 0.707

        self.pose_publisher.publish(pose)
        
        # Call MoveIt IK service
        joint_positions = self.compute_ik(pose)
        
        if joint_positions:
            # Create and publish trajectory message
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.joint_names
            
            point = JointTrajectoryPoint()
            point.positions = joint_positions
            point.time_from_start = Duration(sec=1, nanosec=0)
            # Use distance-based timing for trajectory execution
            if hasattr(self, 'last_position'):
                distance = math.sqrt((robot_x - self.last_position[0])**2 + 
                                    (robot_y - self.last_position[1])**2 + 
                                    (robot_z - self.last_position[2])**2)
                
                if movement_time == "slow" and distance <= 0.01:
                    trajectory_time = 1.5
                elif movement_time == "slow" and distance <= 0.02:
                    trajectory_time = 2.0
                elif movement_time == "slow" and distance <= 0.04:
                    trajectory_time = 3.0
                elif movement_time == "slow" and distance > 0.04:
                    trajectory_time = 5.0
                else:
                    trajectory_time = max(0.3, min(0.3, distance * 2.0))
            else:
                trajectory_time = 3.0 if self.is_first_move else 0.4
                
            point.time_from_start = Duration(sec=int(trajectory_time), 
                                            nanosec=int((trajectory_time % 1) * 1e9))
            
            trajectory_msg.points.append(point)
            self.trajectory_pub.publish(trajectory_msg)
                
            time.sleep(trajectory_time)
            self.last_position = (robot_x, robot_y, robot_z)

            self.get_logger().info(f'Published trajectory: X={robot_x:.4f}, Y={robot_y:.4f}, Z={robot_z:.4f}')
            return True
        else:
            self.get_logger().warn(f'Failed to compute IK for X={robot_x:.4f}, Y={robot_y:.4f}, Z={robot_z:.4f}')
            return False
    
    def return_to_home_position(self):
        """Return the robot to its home position with all joint angles set to 0."""
        self.get_logger().info('Returning robot to home position (all joints at 0)')
        
        # Create a joint trajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names
        
        # Create a point with all zeros for the joint positions
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6DOF robot with all joints at 0
        point.time_from_start = Duration(sec=3, nanosec=0)  # Give it more time to safely return home
        
        trajectory_msg.points.append(point)
        
        # Publish the trajectory
        self.trajectory_pub.publish(trajectory_msg)
        
        # Wait for the movement to complete
        self.get_logger().info('Waiting for robot to reach home position')
        time.sleep(3.5)  # Wait a bit longer than the trajectory duration
        self.get_logger().info('Robot returned to home position')

    def filter_points(self, commands, min_distance=None):
        """Filter out points that are too close together to reduce unnecessary movements."""
        if min_distance is None:
            min_distance = self.command_reduction_factor
        if not commands:
            return commands
        
        filtered = [commands[0]]  # Always keep the first command
        last_point = None
        
        for cmd_type, coords in commands[1:]:
            if cmd_type in ['move', 'line'] and coords:
                if last_point is None:
                    filtered.append((cmd_type, coords))
                    last_point = coords
                else:
                    # Calculate distance from last point
                    distance = math.sqrt((coords[0] - last_point[0])**2 + (coords[1] - last_point[1])**2)
                    if distance >= min_distance:  # Only add if point is far enough
                        filtered.append((cmd_type, coords))
                        last_point = coords
                    # else: skip this point as it's too close
            else:
                filtered.append((cmd_type, coords))  # Keep non-coordinate commands
        
        return filtered

    # def keyboard_listener(self):
    #     """Listen for keyboard input to stop drawing and return home."""
    #     while rclpy.ok():
    #         try:
    #             user_input = input()
    #             if user_input.lower() in ['h', 'home', 'stop']:
    #                 self.get_logger().info('Home command received! Stopping current operation and returning to home position...')
    #                 self.stop_drawing = True
    #                 self.return_to_home_position()
    #                 break
    #         except EOFError:
    #             break
    #         except KeyboardInterrupt:
    #             break
    

def main(args=None):
    rclpy.init(args=args)
    node = SVGRobotDrawer()
    
    # Use a multithreaded executor for the service clients
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by keyboard interrupt')
    except Exception as e:
        node.get_logger().error(f'Node error: {str(e)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()