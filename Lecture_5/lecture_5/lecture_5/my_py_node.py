#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node 

def rotation(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)]
    ])

class MyNode(Node):

    def __init__(self):
        super().__init__('example_node')
        self.create_timer(0.2, self.timer_callback)
        self._angle = np.pi/180

    def timer_callback(self):
        self.get_logger().info(f'Rotation Matrix : {rotation(self._angle)}')
        self._angle += np.pi/180

def main(args=None):
    rclpy.init(args=args)

    node = MyNode()

    rclpy.spin(node)
    node.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()