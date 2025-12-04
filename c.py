#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
# from std_msgs.msg import String  <-- REMOVED: No longer needed
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped, PointStamped, Quaternion
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import math

try:
    from tf2_geometry_msgs import do_transform_point
except ImportError:
    # Fallback/placeholder if tf2_geometry_msgs is not installed or import fails
    def do_transform_point(point, transform):
        raise NotImplementedError("tf2_geometry_msgs is required for do_transform_point")

SHOW_IMAGE = True
TEAM_ID = 2203 

# Marker IDs and Frame Names for ArUco
FERTILIZER_ARUCO_ID = 3 
FERTILIZER_BASE_ARUCO_ID = 6 
FRAME_NAMES = {
    FERTILIZER_ARUCO_ID: f'{TEAM_ID}_fertiliser_1',     # ID 3 is renamed to *_can
    FERTILIZER_BASE_ARUCO_ID: f'{TEAM_ID}_ebot_base' # ID 6 is renamed to *_ebot_base
}

# Configuration for Robust Locking
CONFIDENCE_THRESHOLD = 5 # Number of consecutive detections required to lock the ArUco TF
POSITION_THRESHOLD = 0.10 # 10 cm proximity threshold for fruit matching

class VisionAndTFPublisher(Node):
    """
    ROS 2 Node to:
    1. Detect ArUco markers, store their 'base_link' TF after a confidence threshold,
       and continuously broadcast the locked TF.
    2. Detect 'bad fruits' using color/contour, find their 3D position using depth 
       data, and continuously broadcast their TF.
    """
    def __init__(self):
        super().__init__('vision_and_tf_publisher')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None 
        
        # --- MODIFIED: Removed is_detection_enabled flag (Always True now) ---

        # Camera Intrinsics (Hardcoded values are typically loaded from a sensor_msgs/CameraInfo topic)
        self.centerCamX = 642.724365234375
        self.centerCamY = 361.9780578613281
        self.focalX = 915.3003540039062
        self.focalY = 914.0320434570312
        self.camera_matrix = np.array([[self.focalX, 0, self.centerCamX], [0, self.focalY, self.centerCamY], [0, 0, 1]])
        self.dist_coeffs = np.zeros((4, 1))
        self.marker_size = 0.13 # Using 0.13 from ArUco code
        
        # ArUco Setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cb_group = ReentrantCallbackGroup() 

        # TF Locking Mechanism (for ArUco)
        self.locked_tfs = {id: None for id in FRAME_NAMES.keys()}
        self.detection_confidence_counter = {id: 0 for id in FRAME_NAMES.keys()}
        
        # TF Locking Mechanism (for Bad Fruits)
        self.locked_fruit_tfs = {}  # {fruit_id: TransformStamped}
        self.fruit_detection_confidence = {}  # {fruit_id: counter}
        self.fruit_detection_positions = {}  # {fruit_id: [(x, y, z), ...]} - Stores 'camera_link' positions pre-lock
        
        # Subscriptions 
        self.create_subscription(
            Image, '/camera/image_raw', self.colorimagecb, 10, callback_group=self.cb_group
        )
        self.create_subscription(
            Image, '/camera/depth/image_raw', self.depthimagecb, 10, callback_group=self.cb_group
        )

        # --- MODIFIED: Removed /detection_status subscription ---
        
        # Main loop timer 
        self.create_timer(0.1, self.process_vision_and_tf, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow('Vision and TF Publisher', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Vision and TF Publisher', 1280, 720)
        
        self.get_logger().info(f"VisionAndTFPublisher node started. Team ID: {TEAM_ID}")
        self.get_logger().info("Vision System Active (Continuous Mode)")

    ## --- CALLBACKS ---
    
    # --- MODIFIED: Removed detection_status_cb ---

    def colorimagecb(self, data):
        """Callback function for color image topic."""
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting color image: {e}')

    def depthimagecb(self, data):
        """Callback function for depth image topic (converts 16UC1 to meters)."""
        try:
            depth_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            if depth_img.dtype == np.uint16:
                self.depth_image = depth_img.astype(np.float32) / 1000.0 
            else:
                self.depth_image = depth_img
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')
    
    # --- HELPER METHODS ---
    
    def rvec_tvec_to_transform(self, rvec, tvec):
        """Converts rvec/tvec to a TransformStamped message (optical frame). (UNMODIFIED)"""
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        q = Quaternion()
        
        # Quaternion Calculation logic from original code 1
        trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            q.w = 0.25 * s
            q.x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            q.y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            q.z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
            q.w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            q.x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            q.y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            q.z = 0.25 * s
        
        transform = TransformStamped()
        transform.transform.translation.x = float(tvec[0][0])
        transform.transform.translation.y = float(tvec[0][1])
        transform.transform.translation.z = float(tvec[0][2])
        transform.transform.rotation = q
        return transform

    def detect_aruco_markers(self, image):
        """Detects ArUco markers and estimates their pose. (UNMODIFIED)"""
        detected_markers = []
        if image is None: return detected_markers
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            if ids is None or len(ids) == 0: return detected_markers
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            for i in range(len(ids)):
                detected_markers.append({
                    'id': int(ids[i][0]),
                    'corners': corners[i],
                    'rvec': rvecs[i],
                    'tvec': tvecs[i]
                })
        except Exception as e:
            self.get_logger().error(f'Error in ArUco detection: {e}')
        return detected_markers
        
    def draw_axes(self, image, rvec, tvec, length=0.07):
        """Draws the 3D axes on the image for visualization. (UNMODIFIED)"""
        try:
            cv2.drawFrameAxes(
                image, self.camera_matrix, self.dist_coeffs, rvec, tvec, length, thickness=6
            )
        except Exception as e:
            self.get_logger().error(f'Error drawing axes: {e}') 
            
    def get_depth_at_point(self, x, y):
        """Retrieves the depth value (in meters) at a specific pixel (x, y). (UNMODIFIED)"""
        if self.depth_image is None:
            return None
            
        try:
            height, width = self.depth_image.shape
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            
            depth_value = self.depth_image[y, x]

            if depth_value <= 0.0 or np.isnan(depth_value):
                return None
            
            return float(depth_value)
                
        except Exception as e:
            self.get_logger().error(f'Error getting depth: {e}')
            return None
            
    def compute_3d_position(self, pixel_x, pixel_y, depth):
        """Converts 2D pixel coordinates and depth to 3D in 'camera_link' frame. (UNMODIFIED)"""
        z_depth = float(depth)
        
        # 1. Standard Pinhole Model (Optical Frame: X_opt=Right, Y_opt=Down, Z_opt=Depth)
        x_optical = z_depth * (pixel_x - self.centerCamX) / self.focalX
        y_optical = z_depth * (pixel_y - self.centerCamY) / self.focalY
        
        # 2. Map Optical to Link Frame (Standard ROS camera convention)
        pos_x_cam = z_depth    # X_link = Z_optical (Depth)
        pos_y_cam = -x_optical # Y_link = -X_optical (Invert Right to get Left)
        pos_z_cam = -y_optical # Z_link = -Y_optical (Invert Down to get Up)
        
        return pos_x_cam, pos_y_cam, pos_z_cam

    def create_fruit_mask(self, image):
        """Creates an ROI mask focused on the conveyor belt/tray. (UNMODIFIED)"""
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Define a general ROI for the area of interest
        tray_x_start = 50
        tray_x_end = width // 2
        tray_y_start = height // 4
        tray_y_end = 3 * height // 4
        
        mask[tray_y_start:tray_y_end, tray_x_start:tray_x_end] = 255
        
        return mask

    def bad_fruit_detection(self, rgb_image):
        """Detects bad fruits using a robust color mask and contour detection. (UNMODIFIED)"""
        bad_fruits = []
        if rgb_image is None:
            return bad_fruits

        try:
            tray_mask = self.create_fruit_mask(rgb_image)
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            
            # Color masking for 'bad' colors (brown/grey/dark)
            brown_mask = cv2.inRange(hsv_image, np.array([5, 50, 20]), np.array([25, 255, 100]))
            grey_mask = cv2.inRange(hsv_image, np.array([0, 0, 100]), np.array([180, 50, 255]))
            dark_mask = cv2.inRange(hsv_image, np.array([0, 0, 0]), np.array([180, 255, 60]))
            
            combined_mask = cv2.bitwise_or(brown_mask, grey_mask)
            combined_mask = cv2.bitwise_or(combined_mask, dark_mask)
            final_mask = cv2.bitwise_and(combined_mask, tray_mask)
            
            # Morphological operations for noise reduction
            kernel = np.ones((5, 5), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            fruit_id = 1
            for contour in contours:
                area = cv2.contourArea(contour)
                # Area and aspect ratio filtering
                if 800 < area < 25000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 0.5 < aspect_ratio < 2.0:
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Get depth for the fruit center
                        depth = self.get_depth_at_point(center_x, center_y)
                        
                        if depth is None or depth == 0:
                            # Fallback depth if sensor data is bad
                            depth = 0.5 
                        
                        fruit_info = {
                            'id': fruit_id, # This ID is only local to this frame, will be matched/reassigned later
                            'center': (center_x, center_y),
                            'depth': depth,
                            'bbox': (x, y, w, h),
                            'contour': contour,
                        }
                        bad_fruits.append(fruit_info)
                        fruit_id += 1
            
        except Exception as e:
            self.get_logger().error(f'Error in bad fruit detection: {e}')
        
        return bad_fruits

    # --- TF LOCKING HELPER METHODS ---
    
    # 🌟 CORRECTED METHOD: Must check against both pre-locked and locked TFs.
    def match_fruit_to_tracked(self, pixel_x, pixel_y, depth):
        """
        Matches a detected fruit to an existing tracked (pre-lock) or locked fruit 
        based on 3D position proximity in the 'camera_link' frame.
        """
        # Compute 3D position of the current detection in 'camera_link' frame
        pos_x, pos_y, pos_z = self.compute_3d_position(pixel_x, pixel_y, depth)
        
        # A. Check against currently tracked (pre-locked) fruits
        for fruit_id, positions in self.fruit_detection_positions.items():
            if positions:
                # Use the last known position for comparison
                last_pos = positions[-1] 
                distance = math.sqrt(
                    (pos_x - last_pos[0])**2 + 
                    (pos_y - last_pos[1])**2 + 
                    (pos_z - last_pos[2])**2
                )
                
                if distance < POSITION_THRESHOLD:
                    return fruit_id  # Match found (pre-locked)

        # B. Check against currently locked fruits
        for fruit_id, tf_msg in self.locked_fruit_tfs.items():
            # If already locked, its TF is in 'base_link'. We must transform it back 
            # to 'camera_link' for comparison with the new detection.
            try:
                base_pt = PointStamped()
                base_pt.header.frame_id = tf_msg.header.frame_id # 'base_link'
                # Use Time(0) to get the latest available transform
                base_pt.header.stamp = rclpy.time.Time().to_msg() 
                base_pt.point.x = tf_msg.transform.translation.x
                base_pt.point.y = tf_msg.transform.translation.y
                base_pt.point.z = tf_msg.transform.translation.z
                
                # Lookup transform from 'base_link' to 'camera_link'
                base_to_camera = self.tf_buffer.lookup_transform(
                    'camera_link', 'base_link', rclpy.time.Time()
                )
                
                # Transform the locked point to 'camera_link'
                cam_pt_locked = do_transform_point(base_pt, base_to_camera)
                
                # Compare distance in 'camera_link' frame
                dist_x = pos_x - cam_pt_locked.point.x
                dist_y = pos_y - cam_pt_locked.point.y
                dist_z = pos_z - cam_pt_locked.point.z
                distance = math.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
                
                if distance < POSITION_THRESHOLD:
                    return fruit_id # Match found (locked)
                    
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                # Silently fail if TF lookup is temporarily unavailable
                pass
            except Exception as e:
                self.get_logger().error(f'Error during locked fruit matching: {e}')
                
        return None  # New fruit
    
    def lock_fruit_tf(self, fruit_id, pos_x_cam, pos_y_cam, pos_z_cam):
        """Locks a bad fruit's TF in base_link frame after confidence threshold. (UNMODIFIED)"""
        try:
            current_stamp = self.get_clock().now().to_msg()
            frame_name = f'{TEAM_ID}_bad_fruit_{fruit_id}'
            
            # Transform to base_link
            cam_pt = PointStamped()
            cam_pt.header.frame_id = 'camera_link'
            cam_pt.header.stamp = current_stamp
            cam_pt.point.x = pos_x_cam
            cam_pt.point.y = pos_y_cam
            cam_pt.point.z = pos_z_cam
            
            # Lookup transform (rclpy.time.Time() for latest available)
            camera_to_base = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
            base_pt = do_transform_point(cam_pt, camera_to_base)
            
            # Create and store the locked TF
            tf_msg = TransformStamped()
            tf_msg.header.frame_id = 'base_link'
            tf_msg.child_frame_id = frame_name
            tf_msg.transform.translation.x = base_pt.point.x
            tf_msg.transform.translation.y = base_pt.point.y
            tf_msg.transform.translation.z = base_pt.point.z
            
            # Orientation for downward-pointing gripper
            tf_msg.transform.rotation.x = 0.0
            tf_msg.transform.rotation.y = 1.0
            tf_msg.transform.rotation.z = 0.0
            tf_msg.transform.rotation.w = 0.0
            
            self.locked_fruit_tfs[fruit_id] = tf_msg
            
            self.get_logger().info(
                f'✅ LOCKED {frame_name.upper()} after {CONFIDENCE_THRESHOLD} detections '
                f'at X:{base_pt.point.x:.3f}, Y:{base_pt.point.y:.3f}, Z:{base_pt.point.z:.3f}'
            )
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Waiting for base_link TF to lock fruit {fruit_id}: {e}', throttle_duration_sec=1.0)
            self.fruit_detection_confidence[fruit_id] = 0
        except Exception as e:
            self.get_logger().error(f'Error locking fruit TF for fruit_{fruit_id}: {e}')
            self.fruit_detection_confidence[fruit_id] = 0

    # --- TF LOCKING (for ArUco) ---
    def lock_marker_tf(self, marker):
        """Calculates and stores the ArUco marker's TF in the base_link frame. (UNMODIFIED)"""
        marker_id = marker['id']
        rvec = marker['rvec']
        tvec = marker['tvec']
        frame_name = FRAME_NAMES[marker_id]
        
        try:
            current_stamp = self.get_clock().now().to_msg()
            
            # 1. Convert translation from Optical (ArUco output) to Link Frame 
            aruco_tf_optical = self.rvec_tvec_to_transform(rvec, tvec)
            x_opt = aruco_tf_optical.transform.translation.x
            y_opt = aruco_tf_optical.transform.translation.y
            z_opt = aruco_tf_optical.transform.translation.z
            
            # Optical to Link Frame (ROS convention: X=Fwd, Y=Lft, Z=Up)
            x_link = z_opt  
            y_link = -x_opt 
            z_link = -y_opt 
            
            # 2. Transform position to 'base_link'
            cam_pt = PointStamped()
            cam_pt.header.frame_id = 'camera_link'
            cam_pt.header.stamp = current_stamp
            cam_pt.point.x, cam_pt.point.y, cam_pt.point.z = x_link, y_link, z_link

            # Lookup transform (rclpy.time.Time() for latest available)
            camera_to_base = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
            base_pt = do_transform_point(cam_pt, camera_to_base)
            
            # 3. Define Fixed Rotation (to match the object's expected frame)
            q_fixed = Quaternion()
            if marker_id == FERTILIZER_ARUCO_ID:
                q_fixed = Quaternion(x=0.5, y=0.5, z=-0.5, w=0.5) 
            elif marker_id == FERTILIZER_BASE_ARUCO_ID:
                q_fixed = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0) 
            
            # 4. Create and Store the TransformStamped object
            tf_msg_base = TransformStamped()
            tf_msg_base.header.frame_id = 'base_link'
            tf_msg_base.child_frame_id = frame_name
            tf_msg_base.transform.translation.x = base_pt.point.x
            tf_msg_base.transform.translation.y = base_pt.point.y
            tf_msg_base.transform.translation.z = base_pt.point.z 
            tf_msg_base.transform.rotation = q_fixed
            
            self.locked_tfs[marker_id] = tf_msg_base
            
            self.get_logger().info(
                f'✅ LOCKED {frame_name.upper()} (ID: {marker_id}) after {CONFIDENCE_THRESHOLD} detections '
                f'at X:{base_pt.point.x:.3f}, Y:{base_pt.point.y:.3f}, Z:{base_pt.point.z:.3f}'
            )
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Waiting for base_link TF to lock {frame_name} (ID {marker_id}): {e}', throttle_duration_sec=1.0)
            self.detection_confidence_counter[marker_id] = 0 # Reset counter if TF lookup fails
        except Exception as e:
            self.get_logger().error(f'Error locking ArUco TF for {frame_name}: {e}')
            self.detection_confidence_counter[marker_id] = 0 # Reset counter on error

    # --- MAIN PROCESS LOOP ---
    def process_vision_and_tf(self):
        # --- MODIFIED: Removed 'if not self.is_detection_enabled: return' logic ---
        
        if self.cv_image is None:
            return

        current_stamp = self.get_clock().now().to_msg()
        display_image = self.cv_image.copy()
        
        # --- 1. ArUco Detection and Locking (Original Code 1 Logic) ---
        detected_ids_in_frame = set()
        aruco_markers = self.detect_aruco_markers(self.cv_image)
        
        for marker in aruco_markers:
            marker_id = marker['id']
            
            if marker_id not in FRAME_NAMES:
                continue

            detected_ids_in_frame.add(marker_id)
            
            # Visualization
            if display_image is not None:
                cv2.aruco.drawDetectedMarkers(display_image, [marker['corners']], borderColor=(0, 255, 255))
                self.draw_axes(display_image, marker['rvec'], marker['tvec'], length=0.07)
            
            # Locking Logic
            if self.locked_tfs[marker_id] is None:
                self.detection_confidence_counter[marker_id] = min(
                    CONFIDENCE_THRESHOLD, self.detection_confidence_counter[marker_id] + 1
                )
                
                if self.detection_confidence_counter[marker_id] >= CONFIDENCE_THRESHOLD:
                    self.lock_marker_tf(marker)
        
        # Reset confidence for non-detected, unlocked markers
        for marker_id in self.locked_tfs:
            if self.locked_tfs[marker_id] is None and marker_id not in detected_ids_in_frame:
                self.detection_confidence_counter[marker_id] = 0
                
        # Continuous Broadcasting of Locked ArUco TFs
        for marker_id, tf_msg in self.locked_tfs.items():
            if tf_msg is not None:
                tf_msg.header.stamp = current_stamp
                self.tf_broadcaster.sendTransform(tf_msg)
                
        # --- 2. Bad Fruit Detection and TF Locking (With Confidence Mechanism) ---
        
        # MODIFIED: Stop bad fruit detection if ANY fruit is already locked.
        if len(self.locked_fruit_tfs) == 0:
            bad_fruits = self.bad_fruit_detection(self.cv_image)
            detected_fruit_ids_in_frame = set()
            
            for fruit in bad_fruits:
                center_x, center_y = fruit['center']
                depth = fruit['depth']
                x, y, w, h = fruit['bbox']
                contour = fruit['contour']
                
                # Compute 3D position in 'camera_link' frame
                pos_x_cam, pos_y_cam, pos_z_cam = self.compute_3d_position(center_x, center_y, depth)
                
                # Match to existing tracked fruit or assign new ID
                matched_fruit_id = self.match_fruit_to_tracked(center_x, center_y, depth)
                
                if matched_fruit_id is None:
                    # Check all currently tracked AND locked IDs
                    all_ids = list(self.fruit_detection_confidence.keys()) + list(self.locked_fruit_tfs.keys())
                    
                    # Filter out duplicates and find the maximum ID
                    unique_ids = [int(i) for i in set(all_ids)]

                    # Assign the next available ID
                    new_fruit_id = max(unique_ids) + 1 if unique_ids else 1
                    
                    matched_fruit_id = new_fruit_id
                    self.fruit_detection_confidence[matched_fruit_id] = 0
                    self.fruit_detection_positions[matched_fruit_id] = []
                
                detected_fruit_ids_in_frame.add(matched_fruit_id)
                
                # Store position history for stability tracking (only for pre-locked fruits)
                if matched_fruit_id not in self.locked_fruit_tfs:
                    self.fruit_detection_positions[matched_fruit_id].append((pos_x_cam, pos_y_cam, pos_z_cam))
                    if len(self.fruit_detection_positions[matched_fruit_id]) > 10:
                        self.fruit_detection_positions[matched_fruit_id].pop(0)  # Keep last 10 positions
                
                # Visualization
                frame_name = f'{TEAM_ID}_bad_fruit_{matched_fruit_id}'
                cv2.drawContours(display_image, [contour], -1, (0, 255, 255), 2)
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Show if locked or detecting
                if matched_fruit_id in self.locked_fruit_tfs:
                    label = f'bad fruit {matched_fruit_id} (LOCKED)'
                    color = (0, 255, 255)  # Yellow for locked
                else:
                    conf_count = self.fruit_detection_confidence.get(matched_fruit_id, 0)
                    label = f'bad fruit {matched_fruit_id} ({conf_count}/{CONFIDENCE_THRESHOLD})'
                    color = (0, 255, 0)  # Green for detecting
                
                cv2.putText(display_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Locking Logic
                if matched_fruit_id not in self.locked_fruit_tfs:
                    self.fruit_detection_confidence[matched_fruit_id] = min(
                        CONFIDENCE_THRESHOLD, 
                        self.fruit_detection_confidence[matched_fruit_id] + 1
                    )
                    
                    if self.fruit_detection_confidence[matched_fruit_id] >= CONFIDENCE_THRESHOLD:
                        self.lock_fruit_tf(matched_fruit_id, pos_x_cam, pos_y_cam, pos_z_cam)
            
            # Reset confidence for non-detected, unlocked fruits
            for fruit_id in list(self.fruit_detection_confidence.keys()):
                if fruit_id not in self.locked_fruit_tfs and fruit_id not in detected_fruit_ids_in_frame:
                    # Decay confidence faster if fruit is missed
                    self.fruit_detection_confidence[fruit_id] = max(0, self.fruit_detection_confidence[fruit_id] - 2)
                    if self.fruit_detection_confidence[fruit_id] == 0:
                        # Remove tracking if confidence drops to 0
                        del self.fruit_detection_confidence[fruit_id]
                        if fruit_id in self.fruit_detection_positions:
                            del self.fruit_detection_positions[fruit_id]
        
        # Continuous Broadcasting of Locked Fruit TFs
        for fruit_id, tf_msg in self.locked_fruit_tfs.items():
            if tf_msg is not None:
                tf_msg.header.stamp = current_stamp
                self.tf_broadcaster.sendTransform(tf_msg)
            
        # --- 3. Display Image ---
        if SHOW_IMAGE and display_image is not None:
            cv2.imshow('Vision and TF Publisher', display_image)
            cv2.waitKey(1)

# --- Main Execution ---
def main(args=None):
    rclpy.init(args=args)
    node = VisionAndTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
