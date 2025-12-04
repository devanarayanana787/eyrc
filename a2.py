#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool
import math
import time

# ----------------- Utility Functions -----------------
def quat_to_yaw(x, y, z, w):
    """Convert quaternion orientation to yaw angle."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def clamp_angle(theta):
    """Normalize angle into [-pi, pi]."""
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta

# ----------------- Main Class -----------------
class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('ebot_nav_task2a')

        # Publishers / Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.detection_status_pub = self.create_publisher(String, '/detection_status', 10) 
        self.detector_signal_pub = self.create_publisher(Bool, '/detection_active_signal', 10) 
        self.advance_complete_pub = self.create_publisher(Bool, '/advance_complete_signal', 10)

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_subscription(String, '/nav_command', self.command_callback, 10) 
        self.create_subscription(String, '/fertilizer_can_received', self.fertilizer_loaded_callback, 10) 
        
        # ------------------- WAYPOINTS (x, y, final_yaw) -------------------
        self.goals = [
            (0.26, -5.40, 0.781),    # Intermediate A
            (0.26, -1.95, 1.57),     # P1 (Dock Station) - Stop point (P1_INDEX = 1)
            (0.26, 1.3, 1.57),       # Intermediate B
            (-1.40, 1.3, -2.10),     # Intermediate C 
            (-1.48, -0.67, -1.57),   # P2 
            (-1.48, -5.7, -1.57),
            (-3.17, -5.7, -3.14),
            (-3.17, 1.2, 1.57),
            (-1.65, 1.2, -1.00),
            (-1.53, -6.61, -1.57),    # P1 (Dock Station) - Stop point (This is the *second* visit to P1)
        ]
        
        self.current_goal_idx = 0
        self.P1_INDEX = 1 
        self.P1_SECOND_INDEX = 9 # Assuming the last one is the second visit

        # Robot Pose and Sensor Data
        self.robot_x, self.robot_y, self.robot_yaw = 0.0, 0.0, 0.0
        self.lidar_ready = False
        self.laser_ranges = []

        # ------------------- STATE MANAGEMENT -------------------
        self.state = 'NAVIGATING' 
        self.detection_active = False    
        self.P1_reported = False         
        self.P1_second_reported = False
        self.detection_cooldown_timer = None 
        self.COOLDOWN_DURATION = 2.0     
        
        # --- Variables for the ADVANCE logic ---
        self.advance_distance = 0.0
        self.initial_x = 0.0
        self.initial_y = 0.0
        
        # ------------------- CONTROL PARAMETERS -------------------
        self.Kp_lin, self.Kp_ang = 3.0, 4.5
        self.max_v, self.max_w = 1.0, 2.0
        self.dist_threshold = 0.2      
        self.angle_threshold = math.radians(10) 

        # Obstacle avoidance params
        self.obs_front_limit = 0.55
        self.obs_clear_limit = 0.9
        self.avoid_mode = False
        self.avoid_side = 0

        # Main loop (30 Hz)
        self.create_timer(1.0 / 30.0, self.update_control)
        self.get_logger().info("WaypointNavigator initialized. Starting navigation...")
        
        self.detector_signal_pub.publish(Bool(data=self.detection_active))

    # ---------- Callbacks ----------

    def odom_callback(self, msg: Odometry):
        """Update robot's current pose."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

    def lidar_callback(self, msg: LaserScan):
        """Store LiDAR data."""
        self.laser_ranges = list(msg.ranges)
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment
        self.lidar_ready = True
        
    def command_callback(self, msg: String):
        """
        Handle STOP/RESUME commands from Detector.
        Format: "STOP,0.40"
        """
        parts = msg.data.upper().split(',')
        command = parts[0]
        
        try:
            self.advance_distance = float(parts[1]) if len(parts) > 1 else 0.0
        except ValueError:
            self.advance_distance = 0.0
            self.get_logger().error(f"Invalid advance distance in command: {msg.data}")

        if command == 'STOP' and self.state == 'NAVIGATING':
            self.state = 'ADVANCE_TO_SHAPE' 
            self.initial_x = self.robot_x
            self.initial_y = self.robot_y
            self.publish_twist(0.0, 0.0)
            self.get_logger().warn(f"⚠️ Received STOP command. Preparing to advance {self.advance_distance:.2f}m.")
    
    def fertilizer_loaded_callback(self, msg: String):
        """
        Handles the FERTILIZER_LOADED confirmation from the other node.
        """
        if self.state == 'WAITING_FOR_FERTILIZER' and msg.data == 'FERTILIZER_LOADED':
            self.get_logger().info("✅ Received FERTILIZER_LOADED. Resuming navigation.")
            
            # Update the flag for the correct P1 visit
            if self.current_goal_idx == self.P1_INDEX:
                self.P1_reported = True
                # Activate detection signal for the Shape Detector after first P1 visit
                self.detection_active = True
                self.detector_signal_pub.publish(Bool(data=self.detection_active))
                self.get_logger().info("✅ LiDAR Detections ACTIVATED. Moving to next goal.")
            
            elif self.current_goal_idx == self.P1_SECOND_INDEX:
                self.P1_second_reported = True
            
            self.current_goal_idx += 1
            self.state = 'NAVIGATING'
            
        elif self.state == 'WAITING_FOR_FERTILIZER':
            self.get_logger().warn(f"Received unexpected message: {msg.data} while waiting.")


    def _resume_external_stop(self):
        """
        Resumes motion after an external (LiDAR) stop and PUBLISHES THE COMPLETION SIGNAL.
        """
        if self.detection_cooldown_timer:
            self.detection_cooldown_timer.cancel()
            self.detection_cooldown_timer = None
            
        self.advance_complete_pub.publish(Bool(data=True))
        self.get_logger().info("📢 Advance/Hold Complete. Publishing /advance_complete_signal.")

        self.state = 'NAVIGATING'
        self.get_logger().info("✅ Resuming navigation.")

    # --- MODIFIED _handle_p1_dock_sequence ---
    def _handle_p1_dock_sequence(self):
        """
        Handles the mandatory stop, report, and wait for FERTILIZER_LOADED at P1.
        Uses the exact goal coordinates for reporting, as requested.
        """
        is_first_visit = self.current_goal_idx == self.P1_INDEX
        is_second_visit = self.current_goal_idx == self.P1_SECOND_INDEX
        
        if is_first_visit and self.P1_reported:
            return
        if is_second_visit and self.P1_second_reported:
            return

        # 1. Get the exact, pre-defined coordinates of the current P1 goal
        report_x, report_y, _ = self.goals[self.current_goal_idx]
        
        msg = String()
        # Format: STATUS,PLANT_ID,X,Y (using 0 as placeholder for Plant ID)
        # Using fixed coordinates as requested (e.g., DOCK_STATION,0,0.26,-1.95)
        msg.data = f"DOCK_STATION,0,{report_x:.2f},{report_y:.2f}" 
        self.detection_status_pub.publish(msg)
        self.get_logger().info(f'📢 PUBLISHED DOCK_STATION (FIXED COORDS): {msg.data}')
        
        # 2. Stop and wait for the confirmation message
        self.state = 'WAITING_FOR_FERTILIZER' 
        self.publish_twist(0.0, 0.0) 
        self.get_logger().warn("🛑 Stopped at Dock Station. WAITING for FERTILIZER_LOADED confirmation...")
        

    # ---------- LiDAR Processing (for Obstacle Avoidance) ----------
    def sector_distance(self, angle_center, angle_width):
        """Get minimum distance in a LiDAR sector."""
        if not self.lidar_ready: return float('inf')
        N = len(self.laser_ranges)
        
        idx_center = int((angle_center - self.angle_min) / self.angle_increment)
        half_range = int(angle_width / abs(self.angle_increment) / 2) 
        start = max(0, idx_center - half_range)
        end = min(N - 1, idx_center + half_range)

        valid = [r for r in self.laser_ranges[start:end + 1] 
                 if r > 0.1 and not math.isinf(r) and not math.isnan(r)]
                 
        return min(valid) if valid else float('inf')

    def front_distance(self):
        """Get minimum distance in the front sector (40 degrees wide)."""
        return self.sector_distance(0.0, math.radians(40)) 

    def choose_side(self):
        """Pick left or right based on which side is more open."""
        left_clear = self.sector_distance(math.radians(45), math.radians(45))
        right_clear = self.sector_distance(-math.radians(45), math.radians(45))
        return 1 if left_clear > right_clear else -1 

    # ---------- Control algorithm ----------
    def update_control(self):
        """Main control loop for navigation, state checking, and obstacle avoidance."""
        # Add new state to the early exit check
        if self.robot_x is None or self.state in ['STOP_WAYPOINT', 'WAITING_FOR_FERTILIZER', 'STOP_EXTERNAL']:
            self.publish_twist(0.0, 0.0)
            return

        # ------------------- ADVANCE LOGIC -------------------
        if self.state == 'ADVANCE_TO_SHAPE':
            current_distance_moved = math.hypot(self.robot_x - self.initial_x, self.robot_y - self.initial_y)

            if current_distance_moved < self.advance_distance:
                self.publish_twist(0.2, 0.0) 
                return
            else:
                self.get_logger().info("✅ Advance complete. Initiating 2-second status hold.")
                
                self.state = 'STOP_EXTERNAL' 
                self.publish_twist(0.0, 0.0)

                if self.detection_cooldown_timer:
                    self.detection_cooldown_timer.cancel()
                self.detection_cooldown_timer = self.create_timer(self.COOLDOWN_DURATION, self._resume_external_stop)
                return
        
        # End of mission check
        if self.current_goal_idx >= len(self.goals):
            self.publish_twist(0.0, 0.0)
            self.state = 'STOP_WAYPOINT' 
            self.get_logger().info("🏁 All waypoints reached. Final stop.")
            return

        gx, gy, gyaw = self.goals[self.current_goal_idx]
        dx, dy = gx - self.robot_x, gy - self.robot_y
        dist_error = math.hypot(dx, dy)
        final_yaw_err = clamp_angle(gyaw - self.robot_yaw)
        
        is_P1_visit = self.current_goal_idx == self.P1_INDEX
        is_P1_second_visit = self.current_goal_idx == self.P1_SECOND_INDEX

        # 1. Waypoint Reached Logic
        if dist_error <= self.dist_threshold:
            if abs(final_yaw_err) > self.angle_threshold:
                self.publish_twist(0.0, self.Kp_ang * final_yaw_err)
                return
            else:
                self.publish_twist(0.0, 0.0)
                
                # Check for the Dock Station (P1)
                # Note: The logic for reaching P1 is still based on dist_threshold, 
                # but the subsequent reporting uses fixed coordinates.
                if (is_P1_visit and not self.P1_reported) or \
                   (is_P1_second_visit and not self.P1_second_reported):
                    self._handle_p1_dock_sequence()
                    return 
                
                elif self.current_goal_idx == len(self.goals) - 1:
                    self.state = 'STOP_WAYPOINT'
                    self.get_logger().info("🏁 Final waypoint reached. Stopping.")
                    return 
                
                else:
                    self.current_goal_idx += 1
                    self.get_logger().info(f"Reached waypoint {self.current_goal_idx}. Moving to next.")
                return 

        # 2. Obstacle Avoidance State Transition
        d_front = self.front_distance()
        if d_front < self.obs_front_limit and not self.avoid_mode:
            self.avoid_mode = True
            self.avoid_side = self.choose_side()
            self.get_logger().warn(f"🚧 Obstacle detected at {d_front:.2f} m. Initiating avoidance.")
        elif self.avoid_mode and d_front > self.obs_clear_limit:
            self.avoid_mode = False
            self.get_logger().info("✨ Obstacle cleared, resuming waypoint path.")

        # 3. PID (P-Control) Calculations
        v_cmd = self.Kp_lin * dist_error
        v_cmd = max(0.0, min(self.max_v, v_cmd))

        heading_error = clamp_angle(math.atan2(dy, dx) - self.robot_yaw)
        w_cmd = self.Kp_ang * heading_error
        w_cmd = max(-self.max_w, min(self.max_w, w_cmd))

        # 4. Obstacle Override Logic
        if self.avoid_mode:
            turn_power = (self.obs_clear_limit - d_front) / (self.obs_clear_limit - self.obs_front_limit)
            w_cmd = self.avoid_side * self.max_w * turn_power
            v_cmd = min(v_cmd, 0.18)

        self.publish_twist(v_cmd, w_cmd)

    # ---------- Command Helpers ----------
    def publish_twist(self, v, w):
        """Publishes the linear and angular velocity commands."""
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_vel_pub.publish(msg)

# ----------------- Main -----------------
def main(args=None):
    rclpy.init(args=args)
    nav_node = WaypointNavigator()
    try:
        rclpy.spin(nav_node)
    except KeyboardInterrupt:
        pass
    finally:
        nav_node.publish_twist(0.0, 0.0)
        nav_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()