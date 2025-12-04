#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Quaternion
from std_msgs.msg import String # Used for the new publication
from tf2_ros import TransformException, TransformStamped, Buffer, TransformListener
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from linkattacher_msgs.srv import AttachLink, DetachLink
import math
import time
from typing import List, Dict

# --- Constants ---
LINEAR_P_GAIN = 1.0   
ANGULAR_P_GAIN = 1.2 

MAX_LINEAR_VEL = 0.7  
MAX_ANGULAR_VEL = 1.0 
POS_TOLERANCE = 0.05  
ORI_TOLERANCE = 0.15  
WAYPOINT_HOLD_TIME = 1.0 

EE_LINK = "wrist_3_link" # Use wrist_3_link as the magnetic gripper
BASE_LINK = "base_link"
TEAM_ID = 2203 # IMPORTANT: Replace with your actual Team ID

# Target TF Frame Names (MUST match your detection script)
FERTILISER_FRAME = f'{TEAM_ID}_can'
EBOT_DROP_FRAME = f'{TEAM_ID}_ebot_base'
BAD_FRUIT_1_FRAME = f'{TEAM_ID}_bad_fruit_1' 
BAD_FRUIT_2_FRAME = f'{TEAM_ID}_bad_fruit_2'
BAD_FRUIT_3_FRAME = f'{TEAM_ID}_bad_fruit_3'
TRASHBIN_DROP_FRAME = 'trash_bin_drop_point' 

# --- Waypoint Definitions (Updated for three fruits) ---

WAYPOINTS = [
    # 1. Home/Safe Pose (P1) - ATTACH FERTILIZER HERE
    { 'name': 'P1', 'pos': [-0.218, -0.547, 0.632], 'ori': [0.707, 0.028, 0.034, 0.707], 'action': 'ATTACH_FERTILISER' },
    
    # 2. Retreat from P1 after picking fertilizer
    { 'name': 'P1_RETREAT', 'pos': [0.2, -0.200, 0.632], 'ori': [0.707, 0.028, 0.034, 0.707], 'action': 'NONE' },
    
    # --- UPDATED SECTION: Added x_offset and y_offset ---
    # 3. Approach eBot (higher approach)
    { 
        'name': 'EBOT_APPROACH', 
        'target_frame': EBOT_DROP_FRAME, 
        'x_offset': 0.0, # <--- Edit this value for X offset
        'y_offset': 0.0, # <--- Edit this value for Y offset
        'z_offset': 0.30, 
        'action': 'NONE'
    },
    
    # 4. Drop fertilizer can on eBot
    { 
        'name': 'EBOT_DROP', 
        'target_frame': EBOT_DROP_FRAME, 
        'x_offset': 0.10, # <--- Edit this value for X offset
        'y_offset': 0.00, # <--- Edit this value for Y offset
        'z_offset': 0.15, 
        'action': 'DETACH_FERTILISER' # The action we are tracking
    },
    # ----------------------------------------------------

    # ------------------- BAD FRUIT 1 SEQUENCE -------------------
    # 5. Approach above Bad Fruit 1
    { 'name': 'FRUIT_1_GRASP', 'target_frame': BAD_FRUIT_1_FRAME, 'z_offset': 0.02, 'action': 'ATTACH_FRUIT'},
    # 6. Retreat with Fruit 1
    { 'name': 'FRUIT_1_RETREAT', 'target_frame': BAD_FRUIT_1_FRAME, 'z_offset': 0.18, 'action': 'NONE'},

    # 7. Place Fruit 1 in Trashbin
    { 'name': 'TRASHBIN_DROP_1', 'pos': [-0.806, 0.010, 0.182], 'ori': [-0.684, 0.726, 0.05, 0.008], 'action': 'DETACH_FRUIT'},

    # 8. Move to intermediate safe position (P2)
    { 'name': 'P2_INTERMEDIATE_1', 'pos': [-0.159, 0.501, 0.415], 'ori': [0.029, 0.997, 0.045, 0.033], 'action': 'NONE'},
    
    # ------------------- BAD FRUIT 2 SEQUENCE -------------------
    # 9. Approach above Bad Fruit 2
    { 'name': 'FRUIT_2_GRASP', 'target_frame': BAD_FRUIT_2_FRAME, 'z_offset': 0.02, 'action': 'ATTACH_FRUIT'},
    # 10. Retreat with Fruit 2
    { 'name': 'FRUIT_2_RETREAT', 'target_frame': BAD_FRUIT_2_FRAME, 'z_offset': 0.18, 'action': 'NONE'},

    # 11. Place Fruit 2 in Trashbin
    { 'name': 'TRASHBIN_DROP_2', 'pos': [-0.806, 0.010, 0.182], 'ori': [-0.684, 0.726, 0.05, 0.008], 'action': 'DETACH_FRUIT'},
    
    # 12. Move to intermediate safe position (P2)
    { 'name': 'P2_INTERMEDIATE_2', 'pos': [-0.159, 0.501, 0.415], 'ori': [0.029, 0.997, 0.045, 0.033], 'action': 'NONE'},

    # ------------------- BAD FRUIT 3 SEQUENCE -------------------
    # 13. Approach above Bad Fruit 3
    { 'name': 'FRUIT_3_GRASP', 'target_frame': BAD_FRUIT_3_FRAME, 'z_offset': 0.02, 'action': 'ATTACH_FRUIT'},
    # 14. Retreat with Fruit 3
    { 'name': 'FRUIT_3_RETREAT', 'target_frame': BAD_FRUIT_3_FRAME, 'z_offset': 0.18, 'action': 'NONE'},

    # 15. Place Fruit 3 in Trashbin
    { 'name': 'TRASHBIN_DROP_3', 'pos': [-0.806, 0.010, 0.182], 'ori': [-0.684, 0.726, 0.05, 0.008], 'action': 'DETACH_FRUIT'},
    
]


class WaypointServoingNode(Node):
    
    def __init__(self):
        super().__init__('ur5_waypoint_servo_node')
        self.cb_group = ReentrantCallbackGroup() 
        self.get_logger().info("UR5 Waypoint Servoing Node started")

        self.is_started = False

        # Publisher for Twist commands
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        
        # --- NEW: Publisher for the fertilizer loaded signal ---
        self.fertilizer_pub = self.create_publisher(String, '/fertilizer_can_received', 10)
        # -----------------------------------------------------

        # TF2 Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Service Clients for Gripper Control
        self.attach_client = self.create_client(AttachLink, '/attach_link', callback_group=self.cb_group)
        self.detach_client = self.create_client(DetachLink, '/detach_link', callback_group=self.cb_group)
        while not self.attach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Attach service not available, waiting again...')
        while not self.detach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Detach service not available, waiting again...')
        
        self.create_subscription(
            String, '/detection_status', self.detection_status_cb, 10, callback_group=self.cb_group
        )

        self.timer_period = 0.02
        self.timer = self.create_timer(self.timer_period, self.control_loop, callback_group=self.cb_group)

        self.waypoints = WAYPOINTS
        self.current_waypoint_index = 0
        self.is_holding = False
        self.hold_start_time = 0.0
        
        self.get_logger().info("Waiting for /detection_status trigger...")

    # --- Gripper Control Methods (Unchanged) ---
    # ... (detection_status_cb, call_attach_service, call_detach_service are the same) ...
    def detection_status_cb(self, msg):
        """Enable manipulation only when 'DOCK_STATION' is received."""
        if "DOCK_STATION" in msg.data:
            if not self.is_started:
                self.is_started = True
                self.get_logger().info(f"Received Trigger: {msg.data}. Manipulation System STARTED.")

    def call_attach_service(self, model_name):
        """Calls the /attach_link service."""
        request = AttachLink.Request()
        request.model1_name = model_name
        request.link1_name = 'body'
        request.model2_name = 'ur5'
        request.link2_name = EE_LINK 

        future = self.attach_client.call_async(request)
        timeout = 2.0
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            time.sleep(0.01)
        
        if future.done():
            if future.result() is not None and future.result().success:
                self.get_logger().info(f'✅ Successfully ATTACHED {model_name}.')
            else:
                self.get_logger().error(f'❌ Failed to ATTACH {model_name}. Service call failed.')
        else:
            self.get_logger().error(f'❌ Timeout waiting for ATTACH service for {model_name}.')

    def call_detach_service(self, model_name):
        """Calls the /detach_link service."""
        request = DetachLink.Request()
        request.model1_name = model_name
        request.link1_name = 'body'
        request.model2_name = 'ur5'
        request.link2_name = EE_LINK 

        future = self.detach_client.call_async(request)
        timeout = 2.0
        start = time.time()
        while not future.done() and (time.time() - start) < timeout:
            time.sleep(0.01)
        
        if future.done():
            if future.result() is not None and future.result().success:
                self.get_logger().info(f'✅ Successfully DETACHED {model_name}.')
            else:
                self.get_logger().error(f'❌ Failed to DETACH {model_name}. Service call failed.')
        else:
            self.get_logger().error(f'❌ Timeout waiting for DETACH service for {model_name}.')

    # --- Servoing and Waypoint Management (get_target_pose is the same) ---
    def get_target_pose(self, waypoint: Dict) -> tuple[List[float], List[float]]:
        """
        Calculates the target position and orientation. 
        Uses TF lookup with offsets if 'target_frame' is present.
        Otherwise, it uses the fixed 'pos' and 'ori'.
        """
        if 'target_frame' in waypoint:
            target_frame = waypoint['target_frame']
            # --- UPDATED: Get x, y, z offsets (default to 0.0) ---
            x_offset = waypoint.get('x_offset', 0.0)
            y_offset = waypoint.get('y_offset', 0.0)
            z_offset = waypoint.get('z_offset', 0.0)
            
            try:
                # Lookup the transform from BASE_LINK to the target TF frame
                target_tf: TransformStamped = self.tf_buffer.lookup_transform(
                    BASE_LINK, target_frame, rclpy.time.Time()
                )
                
                # Apply offsets
                pos = [
                    target_tf.transform.translation.x + x_offset,
                    target_tf.transform.translation.y + y_offset,
                    target_tf.transform.translation.z + z_offset
                ]
                # Use the rotation from the target TF
                q = target_tf.transform.rotation
                ori = [q.x, q.y, q.z, q.w]
                
                return pos, ori

            except TransformException as ex:
                self.get_logger().warn(f'Could not lookup transform {BASE_LINK} to {target_frame}: {ex}', throttle_duration_sec=1.0)
                return None, None 

        else:
            # Fixed Waypoint
            return waypoint['pos'], waypoint['ori']


    def control_loop(self):
        if not self.is_started:
            return

        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info("All waypoints reached. Node shutting down.")
            self.timer.cancel()
            return

        target_waypoint = self.waypoints[self.current_waypoint_index]
        
        # 1. Stop and wait state management
        if self.is_holding:
            time_elapsed = self.get_clock().now().nanoseconds / 1e9 - self.hold_start_time
            if time_elapsed < WAYPOINT_HOLD_TIME:
                self.publish_zero_twist()
                return
            else:
                # Holding finished, move to next waypoint
                self.is_holding = False
                
                # --- NEW: Publish signal after holding for EBOT_DROP ---
                if target_waypoint['name'] == 'EBOT_DROP':
                    self.get_logger().info("📣 FERTILIZER_LOADED signal published to /fertilizer_can_received.")
                    msg = String()
                    msg.data = 'FERTILIZER_LOADED'
                    self.fertilizer_pub.publish(msg)
                # ----------------------------------------------------
                
                self.current_waypoint_index += 1
                if self.current_waypoint_index < len(self.waypoints):
                    next_target_name = self.waypoints[self.current_waypoint_index]['name']
                    self.get_logger().info(f"✅ Hold complete. Moving to next waypoint: {next_target_name}")
                else:
                    self.get_logger().info("✅ All waypoints completed! Stopping.")
                return

        # 2. Get Target Pose (either fixed or via TF lookup)
        target_p, target_q_list = self.get_target_pose(target_waypoint)
        if target_p is None:
            self.publish_zero_twist()
            return # Wait for TF

        # 3. Get current end-effector pose via TF
        current_pose_tf: TransformStamped = None
        try:
            current_pose_tf = self.tf_buffer.lookup_transform(
                BASE_LINK, EE_LINK, rclpy.time.Time()
            )
        except TransformException as ex:
            # This is the external TF error you were seeing.
            self.get_logger().warn(f'Could not transform {BASE_LINK} to {EE_LINK}: {ex}', throttle_duration_sec=1.0)
            self.publish_zero_twist()
            return

        # 4. Calculate Error 
        # Position Error (Linear)
        current_p = current_pose_tf.transform.translation
        pos_error_vector = [
            target_p[0] - current_p.x,
            target_p[1] - current_p.y,
            target_p[2] - current_p.z
        ]
        pos_error_magnitude = math.sqrt(sum(e**2 for e in pos_error_vector))

        # Orientation Error (Angular)
        current_q = current_pose_tf.transform.rotation
        target_q = Quaternion(
            x=target_q_list[0], y=target_q_list[1], z=target_q_list[2], w=target_q_list[3]
        )
        ori_error_vector, ori_error_angle = self.quat_error_to_rotation_vector(current_q, target_q)
        
        # 5. Check Goal Reached 
        current_pos_tol = POS_TOLERANCE
        current_ori_tol = ORI_TOLERANCE
        
        # Relax tolerances for drop operations 
        if target_waypoint['name'] == 'EBOT_DROP' or target_waypoint['name'].startswith('TRASHBIN_DROP'):
            current_pos_tol = 0.086
            current_ori_tol = 0.30
        
        if pos_error_magnitude <= current_pos_tol and ori_error_angle <= current_ori_tol:
            self.get_logger().info(
                f"✅ Waypoint {target_waypoint['name']} reached! Pos Error: {pos_error_magnitude:.3f}m"
            )
            self.publish_zero_twist()
            
            # --- ACTION EXECUTION (Pick/Place) ---
            action = target_waypoint.get('action', 'NONE')
            if action == 'ATTACH_FERTILISER':
                self.get_logger().info("🔧 Calling ATTACH service for fertiliser_can")
                self.call_attach_service('fertiliser_can')
            elif action == 'DETACH_FERTILISER':
                self.get_logger().info("🔧 Calling DETACH service for fertiliser_can")
                self.call_detach_service('fertiliser_can')
            elif action == 'ATTACH_FRUIT':
                self.get_logger().info("🔧 Calling ATTACH service for bad_fruit")
                self.call_attach_service('bad_fruit')
            elif action == 'DETACH_FRUIT':
                self.get_logger().info("🔧 Calling DETACH service for bad_fruit")
                self.call_detach_service('bad_fruit')
            
            # Initiate hold time
            self.is_holding = True
            self.hold_start_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(f"🔒 Starting hold period for {WAYPOINT_HOLD_TIME}s at waypoint {target_waypoint['name']}")
            return

        # 6. Command Velocity (P-Control and Clamping)
        twist_cmd = Twist()
        
        # Linear Velocity (Translational Control)
        linear_scale = LINEAR_P_GAIN
        twist_cmd.linear.x = pos_error_vector[0] * linear_scale
        twist_cmd.linear.y = pos_error_vector[1] * linear_scale
        twist_cmd.linear.z = pos_error_vector[2] * linear_scale

        # Angular Velocity (Rotational Control)
        angular_scale = ANGULAR_P_GAIN
        twist_cmd.angular.x = ori_error_vector[0] * angular_scale
        twist_cmd.angular.y = ori_error_vector[1] * angular_scale
        twist_cmd.angular.z = ori_error_vector[2] * angular_scale

        # Clamp linear velocity to maximum limits
        linear_vel_mag = math.sqrt(twist_cmd.linear.x**2 + twist_cmd.linear.y**2 + twist_cmd.linear.z**2)
        if linear_vel_mag > MAX_LINEAR_VEL:
            ratio = MAX_LINEAR_VEL / linear_vel_mag
            twist_cmd.linear.x *= ratio
            twist_cmd.linear.y *= ratio
            twist_cmd.linear.z *= ratio

        # Clamp angular velocity to maximum limits
        angular_vel_mag = math.sqrt(twist_cmd.angular.x**2 + twist_cmd.angular.y**2 + twist_cmd.angular.z**2)
        if angular_vel_mag > MAX_ANGULAR_VEL:
            ratio = MAX_ANGULAR_VEL / angular_vel_mag
            twist_cmd.angular.x *= ratio
            twist_cmd.angular.y *= ratio
            twist_cmd.angular.z *= ratio
        
        self.pub.publish(twist_cmd)
        
    # ... (Quaternion Helper Methods are the same) ...
    def quaternion_normalize(self, q: Quaternion) -> Quaternion:
        """Normalizes a quaternion."""
        norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        if norm == 0.0:
            return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return Quaternion(
            x=q.x / norm, y=q.y / norm, z=q.z / norm, w=q.w / norm
        )

    def quaternion_inverse(self, q: Quaternion) -> Quaternion:
        """Computes the inverse (conjugate for unit quaternion) of a quaternion."""
        return Quaternion(x=-q.x, y=-q.y, z=-q.z, w=q.w)

    def quaternion_multiply(self, q1: Quaternion, q2: Quaternion) -> Quaternion:
        """Multiplies two quaternions (q1 * q2)."""
        x1, y1, z1, w1 = q1.x, q1.y, q1.z, q1.w
        x2, y2, z2, w2 = q2.x, q2.y, q2.z, q2.w 
        
        return Quaternion(
            x=w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            y=w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            z=w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w=w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        )

    def quat_error_to_rotation_vector(self, current_q: Quaternion, target_q: Quaternion) -> tuple[List[float], float]:
       
        # 1. Calculate the error quaternion: Q_error = Q_target * Q_current_inverse
        q_curr_inv = self.quaternion_inverse(current_q)
        q_error = self.quaternion_multiply(target_q, q_curr_inv)
        q_error = self.quaternion_normalize(q_error)

        # 2. Convert Q_error (x, y, z, w) to rotation vector (axis * angle)
        w = q_error.w
        
        if abs(w) > 1.0:
            w = math.copysign(1.0, w) 
            
        angle = 2.0 * math.acos(w)
        
        if angle > math.pi:
            angle = 2.0 * math.pi - angle
            w = math.cos(angle / 2.0)
            q_error.x = -q_error.x
            q_error.y = -q_error.y
            q_error.z = -q_error.z
            
        sin_half_angle = math.sqrt(1.0 - w*w)

        if sin_half_angle < 1e-6:
            return [0.0, 0.0, 0.0], 0.0
        
        scale_factor = angle / sin_half_angle
        
        rotation_vector = [
            q_error.x * scale_factor,
            q_error.y * scale_factor,
            q_error.z * scale_factor
        ]
        
        return rotation_vector, angle
        
    def publish_zero_twist(self):
        """Sends a zero Twist command to stop the arm."""
        zero_twist = Twist()
        self.pub.publish(zero_twist)


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor() 
    node = WaypointServoingNode()
    executor.add_node(node)
    
    try:
        node.get_logger().info("Starting UR5 Pick and Place Servoing Node.")
        executor.spin()
    except SystemExit:
        pass
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt received. Shutting down.')
    finally:
        if rclpy.ok():
            node.publish_zero_twist()
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()