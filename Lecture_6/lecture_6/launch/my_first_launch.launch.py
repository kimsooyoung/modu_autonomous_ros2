# Copyright 2025 @Modulabs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is basic example for launch file.
launches the turtlesim_node and the turtle_teleop_key node 
so you can control the turtle with your keyboard.
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # Start the turtlesim simulator
    turtlesim_node = Node(
        package="turtlesim",
        executable="turtlesim_node",
        name="turtlesim_node",
    )

    # Start the second turtlesim simulator
    turtlesim_node_2 = Node(
        package="turtlesim",
        executable="turtlesim_node",
        name="turtlesim_node_2",
    )

    return LaunchDescription([
        turtlesim_node,
        turtlesim_node_2
    ])