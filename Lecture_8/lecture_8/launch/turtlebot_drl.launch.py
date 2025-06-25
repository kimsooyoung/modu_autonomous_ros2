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
This is a Gazebo environment launch file for a Deep Reinforcement Learning example using TurtleBot.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.actions import TimerAction

def generate_launch_description():

    if 'TURTLEBOT3_MODEL' not in os.environ:
        os.environ['TURTLEBOT3_MODEL'] = "waffle"
    else:
        pass

    if 'GAZEBO_MODEL_PATH' not in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] = os.path.join(get_package_share_directory('lecture_8'), 'models')
    else:
        os.environ['GAZEBO_MODEL_PATH'] += ":" + os.path.join(get_package_share_directory('lecture_8'), 'models')

    # gazebo
    pkg_gazebo_ros = FindPackageShare(package='gazebo_ros').find('gazebo_ros')   
    turtlebot3_gazebo = os.path.join(get_package_share_directory('turtlebot3_gazebo'))
    pkg_path = os.path.join(get_package_share_directory('lecture_8'))
    world_path = os.path.join(pkg_path, 'worlds', 'drl_world.world')

    # Start Gazebo server
    start_gazebo_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world_path}.items()
    )

    # Start Gazebo client    
    start_gazebo_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py'))
    )

    # turtlebot robot_state_publisher
    tb_robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(turtlebot3_gazebo, 'launch', 'robot_state_publisher.launch.py'))
    )

    # launch RViz
    rviz_config_file = os.path.join(pkg_path, 'rviz', 'rl_gazebo.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([
        start_gazebo_server_cmd,
        start_gazebo_client_cmd,
        tb_robot_state_publisher,

        TimerAction(    
            period=3.0,
            actions=[rviz]
        ),
    ])