import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess
from launch.actions import TimerAction

def generate_launch_description():

    # pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    # pkg_share = get_package_share_directory('assignment_3b')
    # IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([turtlebot3_gazebo_launch, '/robot_state_publisher.launch.py']),
    #     launch_arguments={'use_sim_time': use_sim_time}.items(),
    # ),

    use_sim_time = LaunchConfiguration('use_sim_time', default='True')

    if 'TURTLEBOT3_MODEL' not in os.environ:
        os.environ['TURTLEBOT3_MODEL'] = "burger"
    else:
        pass

    # gazebo
    pkg_gazebo_ros = FindPackageShare(package='gazebo_ros').find('gazebo_ros')   
    turtlebot3_gazebo = os.path.join(get_package_share_directory('turtlebot3_gazebo'))
    pkg_path = os.path.join(get_package_share_directory('lecture_5'))
    world_path = os.path.join(pkg_path, 'worlds', 'lane_keeping.sdf')

    # Start Gazebo server
    start_gazebo_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world_path}.items()
    )

    # Start Gazebo client    
    start_gazebo_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py'))
    )

    start_gazebo_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py'))
    )

    # turtlebot robot_state_publisher
    tb_robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(turtlebot3_gazebo, 'launch', 'robot_state_publisher.launch.py'))
    )

    # launch RViz
    rviz_config_file = os.path.join(pkg_path, 'rviz', 'lane_keeping.rviz')
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

        # # use simulation time parameter setup
        # ExecuteProcess(
        #     cmd=['ros2', 'param', 'set', '/gazebo', 'use_sim_time', use_sim_time],
        #     output='screen'
        # ),

        # DeclareLaunchArgument(
        #     'world',
        #     default_value=[os.path.join(pkg_share, 'worlds', 'lane_keeping.sdf')],
        #     description='Simulation Description Format (SDFormat/SDF) for Describing Robot and Environment',
        # ),
        # gazebo,
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([turtlebot3_gazebo_launch, '/robot_state_publisher.launch.py']),
        #     launch_arguments={'use_sim_time': use_sim_time}.items(),
        # ),

        # Node(
        #     package='assignment_3b',
        #     executable='lane_keeping',
        #     name='lane_keeping_node',
        #     emulate_tty=True,
        #     output='screen',
        # ),
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='odometry_rviz',
        #     arguments=['-d', [FindPackageShare("assignment_3b"), '/rviz', '/lane_keeping.rviz',]]
        # ),
    ])