#!/usr/bin/env python3

# ROS2 module imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration

# Python module imports
import queue
import time

# ===================
# PID Controller
# ===================
class PIDController:
    """
    PID controller class
    """
    def __init__(self, kP, kI, kD, kS):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.kS = kS
        self.err_int = 0
        self.err_prev = 0
        self.err_hist = queue.Queue(self.kS)
        self.t_prev = 0

    def control(self, err, t):
        '''
        Generate PID controller output.
        :param err: Instantaneous error in control variable w.r.t. setpoint
        :param t  : Current timestamp
        :return u: PID controller output
        '''
        dt = t - self.t_prev # Timestep
        if dt > 0.0:
            self.err_hist.put(err) # Update error history
            self.err_int += err # Integrate error
            if self.err_hist.full(): # Jacketing logic to prevent integral windup
                self.err_int -= self.err_hist.get() # Rolling FIFO buffer
            self.err_dif = (err - self.err_prev) # Error difference
            u = (self.kP * err) + (self.kI * self.err_int * dt) + (self.kD * self.err_dif / dt) # PID control law
            self.err_prev = err # Update previos error term
            self.t_prev = t # Update timestamp
            return u # Control signal


# ===================
# Robot Controller
# ===================
class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=10
        )

        self.robot_scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.robot_laserscan_callback,
            qos_profile
        )
        self.robot_ctrl_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            qos_profile
        )
        self.lin_vel_pub = self.create_publisher(
            Float64,
            '/lin_vel',
            qos_profile
        )
        self.ang_vel_pub = self.create_publisher(
            Float64,
            '/ang_vel',
            qos_profile
        )
        timer_period = 0.001
        self.timer = self.create_timer(timer_period, self.robot_controller_callback)

        self.laserscan = []
        self.ctrl_msg = Twist()
        self.lin_vel_msg = Float64()
        self.ang_vel_msg = Float64()
        self.start_time = self.get_clock().now()

        # ==================== PART 1: Initialize your PID controller ====================
        # TODO: Set kP, kI, kD, kS appropriately
        self.pid_lat = PIDController(
            0.2, 0.0, 0.0, 10  # TODO: Fill in lateral kP, kI, kD
        )
        self.pid_lon = PIDController(
            0.1, 0.0, 0.0, 10  # TODO: Fill in longitudinal kP, kI, kD
        )

    def robot_laserscan_callback(self, msg):
        # Capture most recent laserscan
        self.laserscan = msg.ranges

    def robot_controller_callback(self):
        DELAY = 4.0
        warm_up_time = self.get_clock().now() - self.start_time

        if warm_up_time > Duration(seconds=DELAY):
            # =============== PART 2: Laser Scan Cropping ===============
            # Select front-left (e.g. 0–29 degrees), front-right (e.g. 330–359 degrees),
            # rear-left (e.g. 150–179 degrees), and rear-right (e.g. 180–209 degrees) scan sectors
            front_left_scan_min = min(self.laserscan[0:-1])
            front_right_scan_min = min(self.laserscan[0:-1])
            rear_left_scan_min = min(self.laserscan[0:-1])
            rear_right_scan_min = min(self.laserscan[0:-1])
            tstamp = time.time()

            LIN_VEL = 0.0
            ANG_VEL = 0.0

            # # PID Tuning and Debugging Example - Uncomment below lines for test
            # LIN_VEL = self.pid_lon.control(min(3.5, self.laserscan[0]), tstamp)
            # self.lin_vel_msg.data = LIN_VEL
            # self.lin_vel_pub.publish(self.lin_vel_msg)

            # ANG_VEL = self.pid_lat.control(0.2, tstamp)
            # self.ang_vel_msg.data = ANG_VEL
            # self.ang_vel_pub.publish(self.ang_vel_msg)

            # =============== PART 3: Avoidance Logic Design ===============
            # Guidelines for students:
            # 1. Compute the minimum obstacle distance on the left and right sides.
            # 2. Favor turning away from the side with closer obstacles.
            # 3. If one side is much clearer (>0.5m) than the other, go straight and steer away from the close side.
            # 4. If both sides are close, slow or stop the robot and rotate toward the more open side.
            # 5. If both sides are clear (>0.5m), drive straight.
            # 6. Use the PID controllers to smoothly adjust linear and angular velocities.

            self.ctrl_msg.linear.x = LIN_VEL
            self.ctrl_msg.angular.z = ANG_VEL
            self.robot_ctrl_pub.publish(self.ctrl_msg)
            print('Distance to closest obstacle:', round(min(front_left_scan_min, front_right_scan_min), 4))
        else:
            left_time = Duration(seconds=DELAY).to_msg().sec - warm_up_time.to_msg().sec
            print(f'Warming up... {left_time} seconds left')


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
