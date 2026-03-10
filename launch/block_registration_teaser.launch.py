from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("concrete_block_registration_teaser")
    params_file = os.path.join(pkg_share, "config", "block_registration.yaml")

    return LaunchDescription(
        [
            Node(
                package="concrete_block_registration_teaser",
                executable="block_registration_teaser_node",
                output="screen",
                parameters=[params_file],
            )
        ]
    )
