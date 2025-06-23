#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration

import queue
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class PIDController:
    def __init__(self, kP, kI, kD, kS):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.kS = kS
        self.err_int = 0
        self.err_dif = 0
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


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=10
        )
        self.robot_image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.robot_image_callback, qos_profile
        )
        self.robot_ctrl_pub = self.create_publisher(
            Twist, '/cmd_vel', qos_profile
        )
        timer_period = 0.001
        self.timer = self.create_timer(timer_period, self.robot_controller_callback)
        self.cv_bridge = CvBridge()
        self.cv_image = None
        self.ctrl_msg = Twist()
        self.start_time = self.get_clock().now()

        # ==================== PART 1: Initialize your PID controller ====================
        # TODO: Set kP, kI, kD, kS appropriately
        self.pid_controller = PIDController(
            kP=None, kI=None, kD=None, kS=None
        )

    def robot_image_callback(self, msg):
        try:
            self.cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as error:
            print(error)

    def robot_controller_callback(self):
        DELAY = 4.0
        warm_up_time = self.get_clock().now() - self.start_time

        if warm_up_time > Duration(seconds=DELAY):
            width, height, channels = self.cv_image.shape
            # ==================== PART 2: Implement image cropping and masking ====================
            # TODO: Crop the image to focus on the lane area
            # HINT: crop using array slicing with width and height
            crop = self.cv_image[...]  # Complete this

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([50, 0, 0])
            upper_yellow = np.array([70, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            m = cv2.moments(mask, False)

            # ==================== PART 3: Implement lane-center error calculation ====================
            try:
                # TODO: Complete cx and cy calculation using image moments
                cx = ...  
                cy = ...
            except ZeroDivisionError:
                cx, cy = width/2, height/2

            cv2.circle(mask, (int(cx), int(cy)), 5, (255, 100, 255), 3)
            cv2.imshow("Camera Frame", self.cv_image)
            cv2.imshow("Cropped Frame", crop)
            cv2.imshow("Masked Frame", mask)
            cv2.waitKey(1)

            # ==================== PART 4: Implement error and velocity commands ====================
            # TODO: Implement lateral error and pass it into PID controller
            error = ...  # Lateral deviation calculation
            tstamp = time.time()

            LIN_VEL = 0.06
            ANG_VEL = self.pid_controller.control(error, tstamp)

            self.ctrl_msg.linear.x = LIN_VEL
            self.ctrl_msg.angular.z = ANG_VEL

            self.robot_ctrl_pub.publish(self.ctrl_msg)
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
