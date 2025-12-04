#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
import numpy as np
import cv2
import math

# ----------------- Utility Functions (Modified for new logic) -----------------
Y_START = -4.92 
Y_LENGTH = 1.35
LANE_1_X = 0.26
LANE_2_X = -1.48 # Using waypoint value of -1.48
LANE_3_X = -3.17

def get_plant_id_modified(x, y, relative_y, robot_vy):
    """
    Determines the plant_ID (1-8) based on world coordinates (x, y)
    AND the specific logic for Lane 2 based on relative shape position.
    
    relative_y: The Y-coordinate of the shape in the robot's frame (positive=left, negative=right).
    robot_vy: The robot's linear velocity in the Y direction (to infer direction of travel).
    """
    # 1. Determine Lane (X-coordinate)
    lane = 0
    if abs(x - LANE_1_X) <= 0.40: # Lane 1 (IDs 1-4)
        lane = 1
    elif abs(x - LANE_2_X) <= 0.40: # Lane 2 (IDs 5-8, or 1-8 based on logic)
        lane = 2
    elif abs(x - LANE_3_X) <= 0.40: # Lane 3 (IDs 5-8)
        lane = 3
    
    if lane == 0:
        return 0 # Not in a valid lane for plant ID calculation

    # 2. Determine ID Block (Y-coordinate)
    y_rel = y - Y_START
    block = -1 # 0-indexed block (0, 1, 2, 3)
    # The block calculation remains the same for all lanes
    if 0 <= y_rel < Y_LENGTH: block = 0
    elif Y_LENGTH <= y_rel < 2 * Y_LENGTH: block = 1
    elif 2 * Y_LENGTH <= y_rel < 3 * Y_LENGTH: block = 2
    elif 3 * Y_LENGTH <= y_rel < 4 * Y_LENGTH: block = 3
    
    if block == -1:
        return 0 # Outside Y-range

    # 3. Calculate final ID based on Lane and Directional Logic
    plant_id = 0
    
    # Simple lanes: IDs are fixed based on lane and block
    if lane == 1: # Lane 1: Always IDs 1-4
        plant_id = block + 1
    elif lane == 3: # Lane 3: Always IDs 5-8
        plant_id = block + 5
        
    # Complex Lane: Lane 2 (The prompt defines this as the "mid line")
    elif lane == 2:
        # Determine Direction of Travel in Y-Axis:
        y_direction = math.sin(math.atan2(math.sin(robot_vy), math.cos(robot_vy))) 
        y_direction_sign = math.sin(math.atan2(math.sin(robot_vy), math.cos(robot_vy)))
        
        # Determine Side: relative_y > 0 means LEFT side of the robot frame
        is_left_side = relative_y > 0.01 
        
        # --- Custom Logic Implementation (Retained from original code) ---
        
        # Case 1: Y Increasing (Moving towards positive Y (North/up))
        if y_direction_sign > 0:
            if is_left_side: # Left side: IDs 5-8
                plant_id = block + 5
            else: # Right side: IDs 1-4
                plant_id = block + 1
                
        # Case 2: Y Decreasing (Moving towards negative Y)
        elif y_direction_sign < 0:
            if is_left_side: # Left side: IDs 1-4
                plant_id = block + 1
            else: # Right side: IDs 5-8
                plant_id = block + 5
        else:
            plant_id = block + 5 # Defaulting to the lane 2 standard (IDs 5-8)
            
    return plant_id

# TARGET CHANGE: Set to 135 degrees (2.356 radians) for Triangle stability
TARGET_ANGLE_TRIANGLE = 2.356  # 135 degrees in radians 
TOLERANCE_TRIANGLE = math.radians(15.0) 

# 90 degrees = pi / 2 ≈ 1.5708 radians
TARGET_ANGLE_SQUARE = 1.5708
TOLERANCE_SQUARE = math.radians(20.0) 

# ADDED: Pentagon interior angle 108 degrees, corresponding exterior angle 72 degrees (1.2566 radians)
TARGET_ANGLE_PENTAGON = math.radians(72.0)
TOLERANCE_PENTAGON = math.radians(15.0) 


class ShapeDetectorRANSAC(Node):
    def __init__(self):
        super().__init__('shape_detector_task3b_modified')
        
        # --- Publishers/Subscribers ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.active_sub = self.create_subscription(Bool, '/detection_active_signal', self.active_signal_callback, 10) 
        
        # Subscriber for Nav Node completion signal
        self.report_ready_sub = self.create_subscription(
            Bool, '/advance_complete_signal', self.report_ready_callback, 10
        )
        
        self.detection_pub = self.create_publisher(String, '/detection_status', 10)
        self.nav_command_pub = self.create_publisher(String, '/nav_command', 10)
        
        # --- State Management ---
        self.robot_x, self.robot_y, self.robot_yaw = 0.0, 0.0, 0.0
        self.robot_vy_inferred = 0.0 # Store a proxy for direction
        self.detection_active = False 
        
        # ADDED: Flag for initial dock detection (Pentagon/Triangle)
        self.initial_dock_detected = False
        
        # Buffer to hold shape data until reporting is complete
        self.pending_report = {'shape': None, 'initial_center': None, 'relative_y': 0.0} 
        
        # --- RANSAC/Clustering Parameters (TUNED) ---
        self.ransac_threshold = 0.025      
        self.ransac_iterations = 500       
        self.min_line_points = 6           
        self.max_lines = 6
        self.cluster_eps = 0.10
        self.cluster_min_samples = 6
        self.min_line_length = 0.1
        self.max_line_length = 0.40        
        self.detection_range = 1.75
        
        # --- Detection history ---
        self.detected_shapes = [] # Stores (x, y) of INITIAL detection point
        self.last_detection_time = 0.0
        self.scan_count = 0
        
        self.get_logger().info('=== RANSAC Shape Detector Initialized (Task 3B Custom Logic) ===')

    # ==================== CORE CALLBACKS ====================
    def odom_callback(self, msg):
        """Update robot pose and store y-velocity for direction inference."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Storing the yaw angle as a proxy for the direction of motion
        self.robot_vy_inferred = self.robot_yaw 
    
    def report_ready_callback(self, msg: Bool):
        """Publishes the final status report with the new PLANT_ID using custom logic."""
        if msg.data and self.pending_report['shape'] is not None:
            shape = self.pending_report['shape']
            
            # --- NEW: Handle Initial Dock Advance Completion ---
            if shape == 'INITIAL_DOCK':
                # The robot has advanced towards the dock. Now trigger the actual dock logic.
                dock_msg = String(data='DOCK_FOUND')
                self.nav_command_pub.publish(dock_msg)
                self.get_logger().warn("✅ Reached Dock Station. Triggering DOCK_FOUND sequence.")
                
                # Cleanup
                self.detected_shapes.append(self.pending_report['initial_center'])
                self.pending_report = {'shape': None, 'initial_center': None, 'relative_y': 0.0} 
                self.last_detection_time = self.get_clock().now().nanoseconds / 1e9
                return
            
            # --- CALCULATE PLANT ID using MODIFIED LOGIC ---
            current_x, current_y = self.robot_x, self.robot_y
            relative_y = self.pending_report['relative_y']
            robot_yaw_at_detection = self.robot_vy_inferred
            
            plant_id = get_plant_id_modified(current_x, current_y, relative_y, robot_yaw_at_detection)
            
            # --- PUBLISH FINAL REPORT ---
            status_map = {
                'TRIANGLE': 'FERTILIZER_REQUIRED', 
                'SQUARE': 'BAD_HEALTH',
                'PENTAGON': 'UNKNOWN_SHAPE' # PENTAGON is only used for initial dock trigger
            }
            status = status_map.get(shape, 'UNKNOWN')
            
            report_msg = String()
            report_msg.data = f"{status},{current_x:.2f},{current_y:.2f},{plant_id}"
            self.detection_pub.publish(report_msg)
            
            self.get_logger().info(f'✅ FINAL REPORT @ NEW POSE: {report_msg.data} (ID: {plant_id}, RelY: {relative_y:.2f})')
            
            self.detected_shapes.append(self.pending_report['initial_center'])
            self.pending_report = {'shape': None, 'initial_center': None, 'relative_y': 0.0} 
            self.last_detection_time = self.get_clock().now().nanoseconds / 1e9

    def active_signal_callback(self, msg: Bool):
        """Handle signal from the Nav Node to activate detection."""
        new_state = msg.data
        if new_state and not self.detection_active:
            self.detection_active = True
            self.get_logger().warn("📢 LiDAR Detection ACTIVATED (Yaw-based control).")
        elif not new_state and self.detection_active:
            self.detection_active = False
            self.get_logger().warn("LiDAR Detection DEACTIVATED (Yaw-based control).")

    def scan_callback(self, msg):
        """Main processing pipeline (MODIFIED for initial dock detection)"""
        self.scan_count += 1
        points = self.scan_to_points(msg)
        best_shape = None
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        time_cooldown_active = (current_time - self.last_detection_time < 1.5) 
        pending_cooldown = self.pending_report['shape'] is not None
        cooldown_active = time_cooldown_active or pending_cooldown

        # Only process if active and not on cooldown
        if self.detection_active and not pending_cooldown and len(points) >= 10:
            
            clusters = self.cluster_points(points)

            for cluster in clusters:
                if len(cluster) < self.cluster_min_samples:
                    continue
                
                lines = self.fit_lines_ransac(cluster)
                
                if len(lines) < 2:
                    continue
                
                shape = self.classify_shape(lines, cluster) 
                
                if shape:
                    best_shape = shape
                    
                    # Calculate detection centroid in ROBOT frame
                    centroid = np.mean(cluster, axis=0)
                    relative_x, relative_y = centroid[0], centroid[1] 
                    
                    # Calculate world coordinates of the detection centroid
                    world_x = self.robot_x + relative_x * math.cos(self.robot_yaw) - relative_y * math.sin(self.robot_yaw)
                    world_y = self.robot_y + relative_x * math.sin(self.robot_yaw) + relative_y * math.cos(self.robot_yaw)
                    
                    if not time_cooldown_active and self.should_publish((world_x, world_y)):
                        
                        # --- NEW: INITIAL DOCK STATION CHECK (Pentagon or Triangle) ---
                        if not self.initial_dock_detected and (shape == 'PENTAGON' or shape == 'TRIANGLE'):
                            self.get_logger().warn(f"🚨 Detected DOCK SHAPE ({shape})! Initiating approach.")
                            self.initial_dock_detected = True # Mark as detected
                            # Send command to Navigator to advance first, then trigger dock sequence
                            self.send_nav_command(command='DOCK_FOUND', shape=shape, initial_center=(world_x, world_y)) 
                            cooldown_active = True
                            return # Stop processing, action taken
                            
                        # --- NORMAL PLANT DETECTION CHECK (Square or Triangle) ---
                        elif shape == 'SQUARE' or shape == 'TRIANGLE':
                            # Send command and buffer data for later reporting (Plant detection)
                            self.send_nav_command(command='STOP', shape=shape, initial_center=(world_x, world_y), relative_y=relative_y)
                            cooldown_active = True 
                        
                    break

    # ==================== RANSAC, CLASSIFICATION, and GEOMETRY (Modified classify_shape) ====================
    # (Other geometry functions omitted for brevity, they are assumed correct from the fetch)

    def scan_to_points(self, msg):
        """Convert LaserScan to cartesian points"""
        points = []
        angle = msg.angle_min
        
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max and r < self.detection_range:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append([x, y])
            angle += msg.angle_increment
        
        return np.array(points) if len(points) > 0 else np.array([]).reshape(0, 2)
    
    def cluster_points(self, points):
        """Custom DBSCAN clustering without sklearn"""
        # Implementation assumed correct from original
        if len(points) == 0: return []
        n_points = len(points)
        visited = np.zeros(n_points, dtype=bool)
        labels = -np.ones(n_points, dtype=int)
        cluster_id = 0
        
        for i in range(n_points):
            if visited[i]: continue
            visited[i] = True
            neighbors = self.region_query(points, i, self.cluster_eps)
            if len(neighbors) < self.cluster_min_samples: continue
            labels[i] = cluster_id
            seed_set = list(neighbors)
            for j in seed_set:
                if not visited[j]:
                    visited[j] = True
                    new_neighbors = self.region_query(points, j, self.cluster_eps)
                    if len(new_neighbors) >= self.cluster_min_samples:
                        seed_set.extend(new_neighbors)
                if labels[j] == -1: labels[j] = cluster_id
            cluster_id += 1
        
        clusters = []
        for cid in range(cluster_id):
            cluster_points = points[labels == cid]
            if len(cluster_points) >= self.cluster_min_samples:
                clusters.append(cluster_points)
        return clusters

    def region_query(self, points, idx, eps):
        """Find all points within eps distance of points[idx]"""
        distances = np.linalg.norm(points - points[idx], axis=1)
        neighbors = np.where(distances < eps)[0]
        return neighbors.tolist()

    def fit_lines_ransac(self, points):
        """Fit multiple lines using RANSAC"""
        lines = []
        remaining = points.copy()
        
        for _ in range(self.max_lines):
            if len(remaining) < self.min_line_points: break
            line, inliers = self.ransac_line(remaining)
            if line is None or len(inliers) < self.min_line_points: break
            inlier_points = remaining[inliers]
            line_length, start, end = self.get_line_endpoints(inlier_points, line)
            if self.min_line_length <= line_length <= self.max_line_length:
                lines.append({'params': line, 'start': start, 'end': end, 'length': line_length})
            remaining = remaining[~inliers]
        return lines

    def ransac_line(self, points):
        """RANSAC algorithm to fit a single line"""
        if len(points) < 2: return None, None
        n_points = len(points)
        best_line, best_inliers, max_inliers = None, None, 0
        
        for _ in range(self.ransac_iterations):
            idx = np.random.choice(n_points, 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]
            line = self.fit_line_two_points(p1, p2)
            if line is None: continue
            distances = self.point_to_line_distance(points, line)
            inliers = distances < self.ransac_threshold
            n_inliers = np.sum(inliers)
            if n_inliers > max_inliers:
                max_inliers = n_inliers
                best_line = line
                best_inliers = inliers
        return best_line, best_inliers

    def fit_line_two_points(self, p1, p2):
        """Fit line ax + by + c = 0 through two points"""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.sqrt(dx**2 + dy**2)
        if length < 1e-6: return None
        a = -dy / length
        b = dx / length
        c = -(a * p1[0] + b * p1[1])
        return (a, b, c)

    def point_to_line_distance(self, points, line):
        """Calculate perpendicular distance from points to line"""
        a, b, c = line
        return np.abs(a * points[:, 0] + b * points[:, 1] + c)

    def get_line_endpoints(self, points, line):
        """Find the two extreme points along the line"""
        a, b, c = line
        line_dir = np.array([b, -a])
        projections = np.dot(points, line_dir)
        min_idx, max_idx = np.argmin(projections), np.argmax(projections)
        start, end = points[min_idx], points[max_idx]
        length = np.linalg.norm(end - start)
        return length, start, end

    def check_angles(self, lines, target_angle, tolerance, required_count):
        """Checks if a required number of angle pairs meet a target angle with tolerance."""
        angles = self.calculate_line_angles(lines)
        valid_angles = [a for a in angles if abs(a - target_angle) <= tolerance]
        return len(valid_angles) >= required_count, valid_angles
        
    def classify_shape(self, lines, cluster):
        """
        Classify shape based on the count of detected lines and angle constraints.
        MODIFIED to include PENTAGON detection logic.
        """
        n_lines = len(lines)
        if n_lines < 2: return None
        
        # Bounding box filtering 
        x_min, y_min = np.min(cluster, axis=0)
        x_max, y_max = np.max(cluster, axis=0)
        perimeter = 2 * ((x_max - x_min) + (y_max - y_min))
        area = (x_max - x_min) * (y_max - y_min)
        if perimeter < 0.25 or perimeter > 2.0 or area < 0.01 or area > 0.25:
            return None
        
        # Check for PENTAGON (5 sides, checking for 4 or more detected lines)
        if n_lines >= 4:
            # Check for angles near 72 degrees (180 - 108)
            is_pentagon, _ = self.check_angles(
                lines, TARGET_ANGLE_PENTAGON, TOLERANCE_PENTAGON, required_count=2
            )
            if is_pentagon:
                return 'PENTAGON'

        # Check for TRIANGLE (3 sides, checking for 2 strong lines)
        if n_lines == 2:
            is_triangle, _ = self.check_angles(
                lines, TARGET_ANGLE_TRIANGLE, TOLERANCE_TRIANGLE, required_count=1
            )
            if is_triangle:
                return 'TRIANGLE'

        # Check for SQUARE (4 sides, checking for 3 or more detected lines)
        elif n_lines >= 3:
            is_square, _ = self.check_angles(
                lines, TARGET_ANGLE_SQUARE, TOLERANCE_SQUARE, required_count=2
            )
            if is_square:
                return 'SQUARE'
                
        return None

    def calculate_line_angles(self, lines):
        """Calculate angles between ALL line directions, including supplementary angle checks for obtuse targets."""
        angles = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                angle = self.angle_between_lines(lines[i], lines[j])
                angles.append(angle)
                
                # Check the supplementary angle for the 135-degree target (Triangle)
                supplementary_angle_tri = math.pi - angle
                if abs(supplementary_angle_tri - TARGET_ANGLE_TRIANGLE) <= TOLERANCE_TRIANGLE:
                    angles.append(supplementary_angle_tri)
                
                # Check the supplementary angle for 72/108 degree target (Pentagon)
                supplementary_angle_pent = math.pi - angle
                if abs(supplementary_angle_pent - TARGET_ANGLE_PENTAGON) <= TOLERANCE_PENTAGON:
                    angles.append(supplementary_angle_pent)
        
        return angles

    def angle_between_lines(self, line1, line2):
        """Calculate the corner angle between two lines (via normal vectors)"""
        a1, b1, _ = line1['params']
        a2, b2, _ = line2['params']
        
        n1, n2 = np.array([a1, b1]), np.array([a2, b2])
        cos_angle = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return math.acos(cos_angle) 
    
    def should_publish(self, center):
        """Check if detection should be published (spatial filtering)"""
        # Checks if the new detection is too close to a previously published detection (0.5m)
        for prev_x, prev_y in self.detected_shapes:
            dist = math.sqrt((center[0] - prev_x)**2 + (center[1] - prev_y)**2)
            if dist < 0.5: 
                return False
        
        return True

    def send_nav_command(self, command='STOP', shape=None, initial_center=None, relative_y=0.0):
        """
        Sends navigation command, either 'DOCK_FOUND' or 'STOP'.
        """
        if command == 'DOCK_FOUND':
            # --- MODIFIED: Advance towards the Dock Shape first ---
            advance_dist = 0.80 # Advance distance for Dock (similar to Triangle)
            
            # Store special state to trigger dock sequence later
            self.pending_report['shape'] = 'INITIAL_DOCK'
            self.pending_report['initial_center'] = initial_center
            
            # Send STOP command with advance distance (Navigator will handle advance)
            stop_msg = String(data=f'STOP,{advance_dist:.2f}')
            self.nav_command_pub.publish(stop_msg)
            self.get_logger().warn(f'🚦 Detected DOCK ({shape}). Advancing {advance_dist:.2f}m before reporting.')
            
            # Record the initial detection point (only once)
            # We add it here (or in callback) to prevent re-detection during advance
            # Ideally, `should_publish` filter handles this if we don't move too far away.
            # But just to be safe:
            # Note: We actually add it in the report_ready_callback usually, but doing it here prevents re-trigger.
            # However, if we advance 0.8m, we might be far enough.
            
        elif command == 'STOP' and shape:
            # Plant detection
            # Determine required advance distance (meters) based on shape type
            advance_dist = 0.0
            if shape == 'SQUARE':
                advance_dist = 0.65 
            elif shape == 'TRIANGLE':
                advance_dist = 0.80
            
            # 1. Store detection data until the Nav Node signals completion
            self.pending_report['shape'] = shape
            self.pending_report['initial_center'] = initial_center
            self.pending_report['relative_y'] = relative_y # Store the relative Y position
            
            # 2. Send STOP command + advance distance to Nav Node
            stop_msg = String(data=f'STOP,{advance_dist:.2f}') 
            self.nav_command_pub.publish(stop_msg)
            self.get_logger().warn(f'🚦 Sent STOP command with advance: {advance_dist:.2f}m. WAITING FOR REPORT SIGNAL.')
        
def main(args=None):
    rclpy.init(args=args)
    detector = ShapeDetectorRANSAC()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()