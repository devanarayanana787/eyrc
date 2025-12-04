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

# Note: The complex get_plant_id function is REMOVED from the Nav Node 
# because it is only calculated in the Detector Node now. 
# Only the Dock Station ID is handled here.

# ----------------- Main Class -----------------
class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('ebot_nav_task3b_modified')

        # Publishers / Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.detection_status_pub = self.create_publisher(String, '/detection_status', 10) 
        self.detector_signal_pub = self.create_publisher(Bool, '/detection_active_signal', 10) 
        self.advance_complete_pub = self.create_publisher(Bool, '/advance_complete_signal', 10)

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_subscription(String, '/nav_command', self.command_callback, 10) 
        
        # ------------------- WAYPOINTS (x, y, final_yaw) -------------------
        self.goals = [
            (0.26, -5.40, 0.781),    # Intermediate A (Towards Dock)
            (0.36, -1.95, 1.57),     # P1 (Dock Station) - STOP POINT
            (0.26, 1.3, 1.57),       # Intermediate B (End of Lane 1)
            (-1.40, 1.3, -2.10),     # Intermediate C (Towards Lane 2 Start) 
            (-1.48, -0.67, -1.57),   # Lane 2 Middle (Mid-point for detection)
            (-1.48, -5.1, -1.57),    # Lane 2 End
            (-3.17, -5.1, -3.14),    # Intermediate (Towards Lane 3)
            (-3.17, 1.2, 1.57),      # Lane 3 End
            (-1.65, 1.2, -1.00),     # Intermediate (Final Dock Approach)
            (-1.53, -6.61, -1.57),   # P3 (Final Dock Station/Stop Point)
        ]
        
        self.current_goal_idx = 0
        self.P1_INDEX = 1 
        
        # Robot Pose and Sensor Data
        self.robot_x, self.robot_y, self.robot_yaw = 0.0, 0.0, 0.0
        self.lidar_ready = False
        self.laser_ranges = []

        # ------------------- STATE MANAGEMENT -------------------
        self.state = 'NAVIGATING' 
        self.detection_active = False    
        self.P1_reported = False         
        self.detection_cooldown_timer = None 
        self.P1_timer = None                 
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
        """
        parts = msg.data.upper().split(',')
        command = parts[0]
        
        # Extract advance distance
        try:
            self.advance_distance = float(parts[1]) if len(parts) > 1 else 0.0
        except ValueError:
            self.advance_distance = 0.0
            self.get_logger().error(f"Invalid advance distance in command: {msg.data}")

        if command == 'STOP' and self.state == 'NAVIGATING':
            # Stop immediately and switch to ADVANCE state
            self.state = 'ADVANCE_TO_SHAPE' 
            self.initial_x = self.robot_x
            self.initial_y = self.robot_y
            self.publish_twist(0.0, 0.0)
            self.get_logger().warn(f"⚠️ Received STOP command. Preparing to advance {self.advance_distance:.2f}m.")

    def _resume_external_stop(self):
        """
        Resumes motion after an external (LiDAR) stop and PUBLISHES THE COMPLETION SIGNAL.
        """
        if self.detection_cooldown_timer:
            self.detection_cooldown_timer.cancel()
            self.detection_cooldown_timer = None
            
        # 1. Signal the Detector that the advance and hold is complete
        self.advance_complete_pub.publish(Bool(data=True))
        self.get_logger().info("📢 Advance/Hold Complete. Publishing /advance_complete_signal.")

        # 2. Resume Navigation
        self.state = 'NAVIGATING'
        self.get_logger().info("✅ Resuming navigation.")

    def _handle_p1_dock_sequence(self):
        """
        Handles the mandatory stop, report, and resume at P1 (Dock Station).
        """
        if self.P1_reported:
            return

        # 1. Publish the DOCK_STATION status: STATUS,X,Y,PLANT_ID=0
        report_x, report_y = self.robot_x, self.robot_y
        plant_id = 0 
        
        msg = String()
        msg.data = f"DOCK_STATION,{report_x:.2f},{report_y:.2f},{plant_id}" 
        self.detection_status_pub.publish(msg)
        self.get_logger().info(f'📢 PUBLISHED DOCK_STATION (P1): {msg.data}')
        
        self.P1_reported = True
        self.state = 'STOP_WAYPOINT' 
        self.publish_twist(0.0, 0.0) 

        # 2. Start the 2-second hold timer 
        self.get_logger().warn("🛑 Starting 2-second hold for DOCK_STATION...")
        self.P1_timer = self.create_timer(self.COOLDOWN_DURATION, self._resume_p1_dock_sequence)


    def _resume_p1_dock_sequence(self):
        """Resumes motion after P1 hold and activates LiDAR detection."""
        
        if self.P1_timer:
            self.P1_timer.cancel()
            self.P1_timer = None

        self.current_goal_idx += 1
        self.state = 'NAVIGATING'
        
        # 3. Activate detection signal for the Shape Detector
        self.detection_active = True
        self.detector_signal_pub.publish(Bool(data=self.detection_active))
        self.get_logger().info("✅ P1 hold complete. LiDAR Detections ACTIVATED. Moving to next goal.")

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
        # Halt conditions
        if self.robot_x is None or self.state in ['STOP_WAYPOINT']:
            self.publish_twist(0.0, 0.0)
            return

        # ------------------- ADVANCE LOGIC -------------------
        if self.state == 'ADVANCE_TO_SHAPE':
            # Calculate distance moved based on initial position
            current_distance_moved = math.hypot(self.robot_x - self.initial_x, self.robot_y - self.initial_y)

            if current_distance_moved < self.advance_distance:
                # Keep moving forward slowly
                self.publish_twist(0.2, 0.0) 
                return
            else:
                # Target distance reached, transition to the required 2-second hold
                self.get_logger().info("✅ Advance complete. Initiating 2-second status hold.")
                
                # Transition to STOP_EXTERNAL state 
                self.state = 'STOP_EXTERNAL' 
                self.publish_twist(0.0, 0.0)

                # Start the 2-second hold timer 
                if self.detection_cooldown_timer:
                    self.detection_cooldown_timer.cancel()
                self.detection_cooldown_timer = self.create_timer(self.COOLDOWN_DURATION, self._resume_external_stop)
                return
        
        elif self.state == 'STOP_EXTERNAL':
            # Robot is stopped and timer is running, waiting for _resume_external_stop 
            self.publish_twist(0.0, 0.0)
            return
        # ------------------- END ADVANCE LOGIC -------------------
        
        # The rest of the logic only runs if state is 'NAVIGATING'
        
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

        # 1. Waypoint Reached Logic
        if dist_error <= self.dist_threshold:
            if abs(final_yaw_err) > self.angle_threshold:
                self.publish_twist(0.0, self.Kp_ang * final_yaw_err)
                return
            else:
                self.publish_twist(0.0, 0.0)
                
                # Handle P1 stop (Dock Station)
                if self.current_goal_idx == self.P1_INDEX and not self.P1_reported: 
                    self._handle_p1_dock_sequence()
                    return 
                
                # Handle Final stop (last index)
                elif self.current_goal_idx == len(self.goals) - 1:
                    self.state = 'STOP_WAYPOINT'
                    self.get_logger().info("🏁 Final waypoint reached. Stopping.")
                    return 
                
                # Handle all other waypoints 
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